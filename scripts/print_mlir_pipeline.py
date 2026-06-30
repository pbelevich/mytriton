from __future__ import annotations

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.mlir_backend import (
    GPU_CLEANUP_PIPELINE,
    GPU_LOWER_TO_NVVM_PIPELINE,
    NVVM_ATTACH_TARGET_PIPELINE,
    run_mlir_pass_pipelines,
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


def main() -> None:
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

    mlir_text = SSAGPUMLIRCodegen().generate(
        "add_kernel",
        ssa_ops,
        params,
    )

    print("=== GPU MLIR ===")
    print(mlir_text)

    print("\n=== After NVVM target attach ===")
    target_mlir = run_mlir_pass_pipelines(
        mlir_text,
        [
            GPU_CLEANUP_PIPELINE,
            NVVM_ATTACH_TARGET_PIPELINE,
        ],
    )
    print(target_mlir)

    print("\n=== After gpu-lower-to-nvvm-pipeline ===")
    lowered = run_mlir_pass_pipelines(
        target_mlir,
        [
            GPU_LOWER_TO_NVVM_PIPELINE,
        ],
    )
    print(lowered)


if __name__ == "__main__":
    main()
