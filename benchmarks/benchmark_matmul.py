"""Benchmark mytriton tensor-core matmul against torch.mm.

The benchmark reports three mytriton timings:

* ``wrapper`` measures the normal public launch path;
* ``raw`` launches the already-compiled CuPy RawKernel directly;
* ``graph`` replays that raw launch from a CUDA Graph.

This separation makes it possible to distinguish Python/compiler-wrapper
overhead from time spent in the generated CUDA kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

import mytriton as triton
import mytriton.language as tl
from mytriton.block_shapes import cuda_threads_per_block
from mytriton.cuda_utils import cuda_module

Tile = tuple[int, int, int]

DEFAULT_TILES: tuple[Tile, ...] = (
    (16, 16, 16),
    (16, 16, 32),
    (16, 32, 16),
    (16, 32, 32),
    (32, 16, 16),
    (32, 16, 32),
    (32, 32, 16),
    (32, 32, 32),
)


@triton.jit
def matmul_kernel(
    a,
    b,
    out,
    M,
    N,
    K,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
    offsets_k = tl.arange(0, BK)

    accumulator = tl.zeros((BM, BN), tl.float32)

    for k_base in range(0, K, BK):
        a_rows = offsets_m
        a_columns = k_base + offsets_k[None, :]
        a_values = tl.load(
            a + a_rows * K + a_columns,
            mask=(a_rows < M) & (a_columns < K),
            other=0.0,
        )

        b_rows = k_base + offsets_k[:, None]
        b_columns = offsets_n
        b_values = tl.load(
            b + b_rows * N + b_columns,
            mask=(b_rows < K) & (b_columns < N),
            other=0.0,
        )

        accumulator = accumulator + tl.dot(a_values, b_values)

    output_pointers = out + offsets_m * N + offsets_n
    output_mask = (offsets_m < M) & (offsets_n < N)
    tl.store(output_pointers, accumulator, mask=output_mask)


def parse_tile(value: str) -> Tile:
    try:
        bm, bk, bn = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"expected BMxBKxBN, got {value!r}") from error

    if bm <= 0 or bk <= 0 or bn <= 0:
        raise argparse.ArgumentTypeError("tile dimensions must be positive")
    return bm, bk, bn


def cuda_event_samples(
    operation: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def timing_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def capture_cuda_graph(operation: Callable[[], object]) -> torch.cuda.CUDAGraph:
    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)

    with torch.cuda.stream(capture_stream):
        operation()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        operation()

    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph


def tflops(m: int, n: int, k: int, milliseconds: float) -> float:
    return 2.0 * m * n * k / milliseconds / 1.0e9


def find_raw_kernel(cuda_source: str) -> Any:
    for (source, _name, _device), kernel in matmul_kernel.cuda_cache.items():
        if source == cuda_source:
            return kernel
    raise RuntimeError("compiled CUDA kernel is missing from the launch cache")


def launch_raw_kernel(
    cp,
    raw_kernel,
    grid: tuple[int, ...],
    threads_per_block: int,
    arguments: tuple[object, ...],
) -> None:
    torch_stream = torch.cuda.current_stream()
    with cp.cuda.Stream.from_external(torch_stream):
        raw_kernel(
            grid,
            (threads_per_block,),
            arguments,
        )


def kernel_resource_summary(
    attributes: dict[str, int],
    *,
    device_attributes: dict[str, int],
    threads_per_block: int,
) -> dict[str, int | float]:
    registers = attributes["num_regs"]
    shared_bytes = attributes["shared_size_bytes"]
    reserved_shared_bytes = device_attributes.get(
        "ReservedSharedMemoryPerBlock",
        0,
    )
    effective_shared_bytes = shared_bytes + reserved_shared_bytes

    block_limits = [
        device_attributes["MaxBlocksPerMultiprocessor"],
        device_attributes["MaxThreadsPerMultiProcessor"] // threads_per_block,
    ]
    if registers:
        block_limits.append(
            device_attributes["MaxRegistersPerMultiprocessor"]
            // (registers * threads_per_block)
        )
    if effective_shared_bytes:
        block_limits.append(
            device_attributes["MaxSharedMemoryPerMultiprocessor"]
            // effective_shared_bytes
        )

    active_blocks = min(block_limits)
    warps_per_block = math.ceil(threads_per_block / device_attributes["WarpSize"])
    active_warps = active_blocks * warps_per_block
    maximum_warps = (
        device_attributes["MaxThreadsPerMultiProcessor"]
        // device_attributes["WarpSize"]
    )

    return {
        "registers_per_thread": registers,
        "static_shared_bytes": shared_bytes,
        "reserved_shared_bytes_per_block": reserved_shared_bytes,
        "local_bytes_per_thread": attributes["local_size_bytes"],
        "estimated_active_blocks_per_sm": active_blocks,
        "estimated_occupancy_percent": 100.0 * active_warps / maximum_warps,
    }


def correctness_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    sample_count: int = 8192,
) -> dict[str, float | bool]:
    count = min(sample_count, actual.numel())
    indices = torch.linspace(
        0,
        actual.numel() - 1,
        steps=count,
        device=actual.device,
        dtype=torch.float64,
    ).to(torch.int64)
    actual_sample = actual.flatten()[indices]
    expected_sample = expected.flatten()[indices]
    difference = (actual_sample - expected_sample).abs()

    return {
        "sampled_values": count,
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "close": torch.allclose(
            actual_sample,
            expected_sample,
            rtol=2.0e-2,
            atol=5.0e-2,
        ),
    }


def benchmark_size(
    size: int,
    *,
    dtype: torch.dtype,
    tiles: Sequence[Tile],
    warmup: int,
    repeats: int,
    use_graphs: bool,
    seed: int,
    artifacts_dir: Path | None,
    dtype_name: str,
) -> dict[str, object]:
    m = n = k = size
    torch.manual_seed(seed + size)
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((k, n), device="cuda", dtype=dtype) * 0.25
    reference = torch.empty((m, n), device="cuda", dtype=torch.float32)

    def run_torch() -> torch.Tensor:
        return torch.mm(a, b, out=reference, out_dtype=torch.float32)

    native_samples = cuda_event_samples(
        run_torch,
        warmup=warmup,
        repeats=repeats,
    )
    native = timing_summary(native_samples)

    native_graph = None
    if use_graphs:
        graph = capture_cuda_graph(run_torch)
        native_graph = timing_summary(
            cuda_event_samples(
                graph.replay,
                warmup=warmup,
                repeats=repeats,
            )
        )

    cp = cuda_module()
    device_attributes = cp.cuda.Device().attributes
    tile_results = []

    for bm, bk, bn in tiles:
        out = torch.empty((m, n), device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
        launcher = matmul_kernel[grid]

        run_wrapper = partial(
            launcher,
            a,
            b,
            out,
            m,
            n,
            k,
            BM=bm,
            BK=bk,
            BN=bn,
        )

        _ops, ssa_ops, cuda_source = run_wrapper()
        torch.cuda.synchronize()

        threads_per_block = cuda_threads_per_block(ssa_ops)
        raw_kernel = find_raw_kernel(cuda_source)
        cupy_a = cp.from_dlpack(a.detach())
        cupy_b = cp.from_dlpack(b.detach())
        cupy_out = cp.from_dlpack(out.detach())
        raw_arguments = (
            cupy_a,
            cupy_b,
            cupy_out,
            np.int32(m),
            np.int32(n),
            np.int32(k),
        )

        run_raw = partial(
            launch_raw_kernel,
            cp,
            raw_kernel,
            grid,
            threads_per_block,
            raw_arguments,
        )

        wrapper = timing_summary(
            cuda_event_samples(
                run_wrapper,
                warmup=warmup,
                repeats=repeats,
            )
        )
        raw = timing_summary(
            cuda_event_samples(
                run_raw,
                warmup=warmup,
                repeats=repeats,
            )
        )

        raw_graph = None
        if use_graphs:
            graph = capture_cuda_graph(run_raw)
            raw_graph = timing_summary(
                cuda_event_samples(
                    graph.replay,
                    warmup=warmup,
                    repeats=repeats,
                )
            )

        run_raw()
        torch.cuda.synchronize()
        correctness = correctness_summary(out, reference)
        resources = kernel_resource_summary(
            raw_kernel.attributes,
            device_attributes=device_attributes,
            threads_per_block=threads_per_block,
        )

        if artifacts_dir is not None:
            artifact_stem = f"matmul_{dtype_name}_m{m}_n{n}_k{k}_bm{bm}_bk{bk}_bn{bn}"
            source_path = artifacts_dir / f"{artifact_stem}.cu"
            source_path.write_text(cuda_source)
            metadata_path = artifacts_dir / f"{artifact_stem}.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "shape": [m, k, n],
                        "tile": {"BM": bm, "BK": bk, "BN": bn},
                        "grid": list(grid),
                        "threads_per_block": threads_per_block,
                        "cuda_source_sha256": hashlib.sha256(
                            cuda_source.encode()
                        ).hexdigest(),
                        "raw_kernel_attributes": raw_kernel.attributes,
                        "mma_instructions_in_source": cuda_source.count(
                            "mma.sync.aligned"
                        ),
                        "barriers_in_source": cuda_source.count("__syncthreads()"),
                        "uses_ldmatrix": "ldmatrix" in cuda_source,
                        "uses_cp_async": "cp.async" in cuda_source,
                    },
                    indent=2,
                )
                + "\n"
            )

        raw_ms = raw["median_ms"]
        native_ms = native["median_ms"]
        tile_results.append(
            {
                "tile": {"BM": bm, "BK": bk, "BN": bn},
                "grid": list(grid),
                "grid_blocks": math.prod(grid),
                "threads_per_block": threads_per_block,
                "arithmetic_intensity_flop_per_byte": bm * bn / (bm + bn),
                "mma_instructions_in_source": cuda_source.count("mma.sync.aligned"),
                "barriers_in_source": cuda_source.count("__syncthreads()"),
                "uses_cp_async": "cp.async" in cuda_source,
                "wrapper": wrapper,
                "raw": raw,
                "graph": raw_graph,
                "raw_tflops": tflops(m, n, k, raw_ms),
                "slowdown_vs_torch": raw_ms / native_ms,
                "wrapper_overhead_percent": 100.0
                * (wrapper["median_ms"] - raw_ms)
                / raw_ms,
                "resources": resources,
                "correctness": correctness,
            }
        )

    return {
        "shape": [m, k, n],
        "torch": {
            "eager": native,
            "graph": native_graph,
            "eager_tflops": tflops(m, n, k, native["median_ms"]),
        },
        "mytriton": tile_results,
    }


def print_size_summary(result: dict[str, object]) -> None:
    m, k, n = result["shape"]
    native = result["torch"]
    native_ms = native["eager"]["median_ms"]
    native_tflops = native["eager_tflops"]
    graph = native["graph"]
    graph_suffix = "" if graph is None else f", graph {graph['median_ms']:.4f} ms"

    print()
    print(f"{m}x{k} @ {k}x{n}")
    print(f"torch.mm: {native_ms:.4f} ms, {native_tflops:.2f} TFLOP/s{graph_suffix}")
    print(
        " BM  BK  BN | wrapper ms | raw ms | graph ms | raw TF/s | "
        "gap | regs | smem B | occ est | max error"
    )
    print("-" * 108)

    for item in result["mytriton"]:
        tile = item["tile"]
        resources = item["resources"]
        correctness = item["correctness"]
        graph_result = item["graph"]
        graph_ms = "   n/a  "
        if graph_result is not None:
            graph_ms = f"{graph_result['median_ms']:8.4f}"
        print(
            f"{tile['BM']:3d} {tile['BK']:3d} {tile['BN']:3d} |"
            f" {item['wrapper']['median_ms']:10.4f} |"
            f" {item['raw']['median_ms']:6.4f} |"
            f" {graph_ms} |"
            f" {item['raw_tflops']:8.2f} |"
            f" {item['slowdown_vs_torch']:4.1f}x |"
            f" {resources['registers_per_thread']:4d} |"
            f" {resources['static_shared_bytes']:6d} |"
            f" {resources['estimated_occupancy_percent']:6.1f}% |"
            f" {correctness['max_abs_error']:.3e}"
        )


def device_summary(cp) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "sm_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cupy_version": cp.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument(
        "--tiles",
        nargs="+",
        type=parse_tile,
        default=list(DEFAULT_TILES),
        metavar="BMxBKxBN",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes must contain positive integers")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.artifacts_dir is not None:
        args.artifacts_dir = args.artifacts_dir.resolve()
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUPY_CACHE_DIR"] = str(args.artifacts_dir / "cupy-cache")
        os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"

    cp = cuda_module()
    metadata = {
        "device": device_summary(cp),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cuda_graphs": not args.no_graphs,
        "tiles": [list(tile) for tile in args.tiles],
    }
    print(json.dumps(metadata, indent=2))

    results = []
    for size in args.sizes:
        result = benchmark_size(
            size,
            dtype=dtype,
            tiles=args.tiles,
            warmup=args.warmup,
            repeats=args.repeats,
            use_graphs=not args.no_graphs,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            dtype_name=args.dtype,
        )
        results.append(result)
        print_size_summary(result)

    report = {"metadata": metadata, "results": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
