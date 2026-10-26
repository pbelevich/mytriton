"""Benchmark mytriton.autotune on Tensor Core GEMM."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from benchmark_matmul import (
    correctness_summary,
    cuda_event_samples,
    device_summary,
    find_raw_kernel,
    find_threads_per_block,
    kernel_resource_summary,
    launch_raw_kernel,
    matmul_kernel,
    tflops,
    timing_summary,
)

import mytriton as triton
from mytriton.cuda_utils import cuda_module

AUTOTUNE_CONFIGS = (
    triton.Config({"BM": 64, "BK": 16, "BN": 64}, num_warps=4),
    triton.Config({"BM": 64, "BK": 32, "BN": 64}, num_warps=4),
    triton.Config({"BM": 128, "BK": 16, "BN": 128}, num_warps=4),
    triton.Config({"BM": 128, "BK": 32, "BN": 128}, num_warps=4),
    triton.Config({"BM": 128, "BK": 32, "BN": 128}, num_warps=8),
)


autotuned_matmul_kernel = triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
    warmup=2,
    rep=7,
)(matmul_kernel)


def config_dict(config: triton.Config) -> dict[str, object]:
    return {**config.kwargs, "num_warps": config.num_warps}


def benchmark_size(
    size: int,
    *,
    dtype: torch.dtype,
    dtype_name: str,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    m = n = k = size
    torch.manual_seed(seed + size)
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((k, n), device="cuda", dtype=dtype) * 0.25
    output = torch.empty((m, n), device="cuda", dtype=torch.float32)
    reference = torch.empty_like(output)

    native = timing_summary(
        cuda_event_samples(
            lambda: torch.mm(a, b, out=reference, out_dtype=torch.float32),
            warmup=warmup,
            repeats=repeats,
        )
    )

    def grid(meta):
        return (
            triton.cdiv(m, meta["BM"]),
            triton.cdiv(n, meta["BN"]),
        )

    launcher = autotuned_matmul_kernel[grid]

    tuning_start = time.perf_counter()
    _, _, cuda_source = launcher(a, b, output, m, n, k)
    torch.cuda.synchronize()
    tuning_ms = (time.perf_counter() - tuning_start) * 1.0e3

    selected = autotuned_matmul_kernel.best_config
    if selected is None:
        raise RuntimeError("autotuner did not select a configuration")
    measurements = next(reversed(autotuned_matmul_kernel.timings.values()))

    wrapper = timing_summary(
        cuda_event_samples(
            partial(launcher, a, b, output, m, n, k),
            warmup=warmup,
            repeats=repeats,
        )
    )

    cp = cuda_module()
    raw_kernel = find_raw_kernel(cuda_source)
    threads_per_block = find_threads_per_block(cuda_source)
    selected_grid = (
        triton.cdiv(m, int(selected.kwargs["BM"])),
        triton.cdiv(n, int(selected.kwargs["BN"])),
    )
    raw_arguments = (
        cp.from_dlpack(a.detach()),
        cp.from_dlpack(b.detach()),
        cp.from_dlpack(output.detach()),
        np.int32(m),
        np.int32(n),
        np.int32(k),
    )
    raw = timing_summary(
        cuda_event_samples(
            partial(
                launch_raw_kernel,
                cp,
                raw_kernel,
                selected_grid,
                threads_per_block,
                raw_arguments,
            ),
            warmup=warmup,
            repeats=repeats,
        )
    )

    torch.mm(a, b, out=reference, out_dtype=torch.float32)
    launcher(a, b, output, m, n, k)
    torch.cuda.synchronize()
    correctness = correctness_summary(output, reference)

    return {
        "shape": [m, k, n],
        "dtype": dtype_name,
        "selected_config": config_dict(selected),
        "candidate_kernel_ms": [
            {
                "config": config_dict(config),
                "median_ms": milliseconds,
            }
            for config, milliseconds in measurements.items()
        ],
        "tuning_ms": tuning_ms,
        "cache_entries": len(autotuned_matmul_kernel.cache),
        "torch": {
            **native,
            "tflops": tflops(m, n, k, native["median_ms"]),
        },
        "cached_wrapper": {
            **wrapper,
            "tflops": tflops(m, n, k, wrapper["median_ms"]),
        },
        "raw": {
            **raw,
            "tflops": tflops(m, n, k, raw["median_ms"]),
        },
        "resources": kernel_resource_summary(
            raw_kernel.attributes,
            device_attributes=cp.cuda.Device().attributes,
            threads_per_block=threads_per_block,
        ),
        "correctness": correctness,
    }


def print_result(result: dict[str, object]) -> None:
    m, k, n = result["shape"]
    print(f"\n{m}x{k} @ {k}x{n}")
    print(f"selected: {result['selected_config']}")
    for candidate in result["candidate_kernel_ms"]:
        print(f"  {candidate['config']}: {candidate['median_ms']:.4f} ms")
    print(f"tuning: {result['tuning_ms']:.2f} ms")
    print(
        f"torch: {result['torch']['median_ms']:.4f} ms, "
        f"{result['torch']['tflops']:.2f} TFLOP/s"
    )
    print(
        f"cached wrapper: {result['cached_wrapper']['median_ms']:.4f} ms, "
        f"{result['cached_wrapper']['tflops']:.2f} TFLOP/s"
    )
    print(
        f"raw: {result['raw']['median_ms']:.4f} ms, "
        f"{result['raw']['tflops']:.2f} TFLOP/s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/artifacts/autotune-a100.json"),
    )
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes must contain positive integers")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")

    cp = cuda_module()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    autotuned_matmul_kernel.clear_cache()
    results = [
        benchmark_size(
            size,
            dtype=dtype,
            dtype_name=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        for size in args.sizes
    ]
    report = {
        "device": device_summary(cp),
        "autotune_configs": [config_dict(config) for config in AUTOTUNE_CONFIGS],
        "results": results,
    }
    print(json.dumps(report["device"], indent=2))
    for result in results:
        print_result(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
