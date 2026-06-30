from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.mlir_backend import (
    default_gpu_to_nvvm_stages,
    format_mlir_lowering_report,
    run_mlir_pipeline_stages,
)
from mytriton.mlir_codegen import SSAGPUMLIRCodegen
from mytriton.ssa import SSALowering
from mytriton.trace import make_runtime_params, trace


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)
    b = tl.load(y + offs, mask=mask, other=0.0)

    tl.store(out + offs, a + b, mask=mask)


def build_gpu_mlir() -> str:
    n = 1000
    block = 256

    x = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    out = np.empty_like(x)

    bound = add_kernel.signature.bind(x, y, out, n, BLOCK=block)
    params = make_runtime_params(add_kernel.signature, bound.arguments)

    ops, params = trace(
        add_kernel.fn,
        add_kernel.signature,
        bound.arguments,
        runtime_params=params,
    )

    ssa_ops = SSALowering().lower(ops)

    return SSAGPUMLIRCodegen().generate(
        "add_kernel",
        ssa_ops,
        params,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="sm_80")
    parser.add_argument("--features", default="+ptx80")
    parser.add_argument("--out-dir", default="mlir-debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlir_text = build_gpu_mlir()
    (out_dir / "00-input-gpu.mlir").write_text(mlir_text)

    result = run_mlir_pipeline_stages(
        mlir_text,
        default_gpu_to_nvvm_stages(
            chip=args.chip,
            features=args.features,
        ),
    )

    for index, stage in enumerate(result.stages, start=1):
        safe_name = stage.name.replace(" ", "-")
        suffix = "ok" if stage.status.value == "ok" else stage.status.value
        path = out_dir / f"{index:02d}-{safe_name}-{suffix}.mlir"
        path.write_text(stage.output_text)

        if stage.error is not None:
            error_path = out_dir / f"{index:02d}-{safe_name}.error.txt"
            error_path.write_text(stage.error)

    print(format_mlir_lowering_report(result))
    print()
    print(f"Wrote debug files to {out_dir}")


if __name__ == "__main__":
    main()
