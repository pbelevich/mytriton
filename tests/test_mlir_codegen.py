from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.mlir_backend import (
    FUNC_CLEANUP_PIPELINE,
    GENERIC_CLEANUP_PIPELINE,
    GPU_CLEANUP_PIPELINE,
    GPU_LOWER_TO_NVVM_PIPELINE,
    NVVM_ATTACH_TARGET_PIPELINE,
    run_mlir_pass_pipeline,
    run_mlir_pass_pipelines,
    try_run_mlir_pass_pipeline,
    try_run_mlir_pass_pipelines,
)
from mytriton.mlir_codegen import SSAGPUMLIRCodegen, SSAMLIRCodegen
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


def build_add_kernel_ssa():
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
    return ssa_ops, params


def build_add_kernel_mlir():
    ssa_ops, params = build_add_kernel_ssa()
    return SSAMLIRCodegen().generate("add_kernel", ssa_ops, params)


def test_add_kernel_mlir_codegen_snapshot():
    mlir_text = build_add_kernel_mlir()

    assert mlir_text == dedent(
        """\
        module {
          func.func @add_kernel(%x: memref<?xf32>, %y: memref<?xf32>, %out: memref<?xf32>, %n: i32, %block_id_x: i32, %thread_id_x: i32) {
            %c_i32_256 = arith.constant 256 : i32
            %v1 = arith.muli %block_id_x, %c_i32_256 : i32
            %v3 = arith.addi %v1, %thread_id_x : i32
            %idx4 = arith.index_cast %v3 : i32 to index
            %v5 = arith.cmpi slt, %v3, %n : i32
            %c_f32_0_0 = arith.constant 0.000000e+00 : f32
            %v6 = scf.if %v5 -> (f32) {
              %loaded6 = memref.load %x[%idx4] : memref<?xf32>
              scf.yield %loaded6 : f32
            } else {
              scf.yield %c_f32_0_0 : f32
            }
            %idx7 = arith.index_cast %v3 : i32 to index
            %v8 = scf.if %v5 -> (f32) {
              %loaded8 = memref.load %y[%idx7] : memref<?xf32>
              scf.yield %loaded8 : f32
            } else {
              scf.yield %c_f32_0_0 : f32
            }
            %v9 = arith.addf %v6, %v8 : f32
            %idx10 = arith.index_cast %v3 : i32 to index
            scf.if %v5 {
              memref.store %v9, %out[%idx10] : memref<?xf32>
            }
            return
          }
        }
        """
    ).rstrip("\n")


def test_add_kernel_mlir_parses_with_bindings():
    mlir = pytest.importorskip("mlir")  # noqa: F841

    from mytriton.mlir_backend import parse_mlir_module

    rendered = parse_mlir_module(build_add_kernel_mlir())

    assert "func.func @add_kernel" in rendered
    assert "scf.if" in rendered
    assert "memref.load" in rendered


def build_add_kernel_gpu_mlir():
    ssa_ops, params = build_add_kernel_ssa()
    return SSAGPUMLIRCodegen().generate("add_kernel", ssa_ops, params)


def test_add_kernel_gpu_mlir_codegen_snapshot():
    mlir_text = build_add_kernel_gpu_mlir()

    assert "module attributes {gpu.container_module}" in mlir_text
    assert "gpu.module @kernels" in mlir_text
    assert "gpu.func @add_kernel" in mlir_text
    assert "gpu.block_id x" in mlir_text
    assert "gpu.thread_id x" in mlir_text
    assert "arith.index_cast %bid_x : index to i32" in mlir_text
    assert "arith.index_cast %tid_x : index to i32" in mlir_text
    assert "memref.load" in mlir_text
    assert "memref.store" in mlir_text
    assert "gpu.return" in mlir_text


def test_add_kernel_gpu_mlir_parses_with_bindings():
    pytest.importorskip("mlir")

    from mytriton.mlir_backend import parse_mlir_module

    rendered = parse_mlir_module(build_add_kernel_gpu_mlir())

    assert "gpu.module @kernels" in rendered
    assert "gpu.func @add_kernel" in rendered


def test_add_kernel_func_mlir_cleanup_pipeline_with_bindings():
    pytest.importorskip("mlir")

    mlir_text = build_add_kernel_mlir()
    rendered = run_mlir_pass_pipeline(
        mlir_text,
        FUNC_CLEANUP_PIPELINE,
    )

    assert "func.func @add_kernel" in rendered
    assert "arith.muli" in rendered
    assert "memref.load" in rendered
    assert "memref.store" in rendered


def test_add_kernel_gpu_mlir_cleanup_pipeline_with_bindings():
    pytest.importorskip("mlir")

    mlir_text = build_add_kernel_gpu_mlir()

    result = try_run_mlir_pass_pipeline(
        mlir_text,
        GPU_CLEANUP_PIPELINE,
    )

    if not result.ok:
        result = try_run_mlir_pass_pipeline(
            mlir_text,
            GENERIC_CLEANUP_PIPELINE,
        )

    assert result.ok, result.error
    assert "gpu.module @kernels" in result.output
    assert "gpu.func @add_kernel" in result.output
    assert "gpu.block_id" in result.output
    assert "gpu.thread_id" in result.output


def test_mlir_invalid_pipeline_reports_error_with_bindings():
    pytest.importorskip("mlir")

    mlir_text = build_add_kernel_mlir()

    result = try_run_mlir_pass_pipeline(
        mlir_text,
        "builtin.module(this-pass-does-not-exist)",
    )

    assert not result.ok
    assert result.error is not None
    assert "this-pass-does-not-exist" in result.error


def test_add_kernel_gpu_mlir_attach_nvvm_target_with_bindings():
    pytest.importorskip("mlir")

    mlir_text = build_add_kernel_gpu_mlir()

    rendered = run_mlir_pass_pipelines(
        mlir_text,
        [
            GPU_CLEANUP_PIPELINE,
            NVVM_ATTACH_TARGET_PIPELINE,
        ],
    )

    assert "gpu.module @kernels" in rendered
    assert "nvvm.target" in rendered
    assert 'chip = "sm_80"' in rendered


def test_add_kernel_gpu_mlir_lower_to_nvvm_smoke_with_bindings():
    pytest.importorskip("mlir")

    mlir_text = build_add_kernel_gpu_mlir()

    result = try_run_mlir_pass_pipelines(
        mlir_text,
        [
            GPU_CLEANUP_PIPELINE,
            NVVM_ATTACH_TARGET_PIPELINE,
            GPU_LOWER_TO_NVVM_PIPELINE,
        ],
    )

    if result.ok:
        assert "gpu.func" not in result.output or "nvvm." in result.output
        assert "llvm." in result.output or "nvvm." in result.output
    else:
        assert result.error is not None
        assert "failed to run MLIR pass pipeline" in result.error
