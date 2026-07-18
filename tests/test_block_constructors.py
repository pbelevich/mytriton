from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def full_and_zeros_kernel(out, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.zeros((BLOCK,), tl.float32)
    values = values + tl.full([BLOCK], 2.5, tl.float32)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def runtime_full_kernel(out, n, value, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.full(BLOCK, value, tl.float32)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def empty_kernel(out, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.empty((BLOCK,), tl.float32)
    tl.store(out + offsets, values, mask=mask)


@pytest.mark.codegen
def test_full_and_zeros_lowering_and_cuda_codegen(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    out = np.zeros(5, dtype=np.float32)

    full_and_zeros_kernel.clear_cache()
    _, ssa_ops, cuda_src = full_and_zeros_kernel[(1,)](out, 5, BLOCK=8)

    expected_ssa = dedent(
        """\
        %0 = zeros {shape=(8,), dtype=f32} : vector<8 x f32>
        %1 = full 2.5 {shape=(8,), dtype=f32} : vector<8 x f32>
        %2 = add %0, %1 : vector<8 x f32>
        %3 = arange {start=0, end=8} : vector<8 x i32>
        %4 = addptr out, %3 : vector<8 x ptr<f32>>
        %5 = cmp_lt %3, n : vector<8 x bool>
        store %4, %2, %5
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void full_and_zeros_kernel(float* out, int n) {
            float v0 = 0.0f;
            float v1 = 2.5f;
            float v2 = (v0 + v1);
            int v3 = threadIdx.x;
            bool v5 = (v3 < n);
            if (v5) {
                out[v3] = v2;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


@pytest.mark.codegen
def test_empty_lowers_to_uninitialized_cuda_value(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")
    out = np.zeros(5, dtype=np.float32)

    empty_kernel.clear_cache()
    _, ssa_ops, cuda_src = empty_kernel[(1,)](out, 5, BLOCK=8)

    assert "%0 = empty {shape=(8,), dtype=f32} : vector<8 x f32>" in (
        SSAPrinter().print_ops(ssa_ops)
    )
    assert "float v0;" in cuda_src


@pytest.mark.execution
@pytest.mark.parametrize("value", [3, 1.25])
def test_runtime_full_and_zeros_execute(cp, value):
    n = 37
    block = 64
    out = cp.zeros(n, dtype=cp.float32)

    runtime_full_kernel.clear_cache()
    runtime_full_kernel[(1,)](out, n, value, BLOCK=block)
    cp.cuda.Stream.null.synchronize()

    cp.testing.assert_allclose(out, cp.full(n, value, dtype=cp.float32))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: tl.empty((), tl.float32),
        lambda: tl.zeros((4, 0), tl.float32),
        lambda: tl.full((4, -1), 0.0, tl.float32),
        lambda: tl.zeros((4, True), tl.float32),
    ],
)
def test_block_constructors_reject_invalid_shapes(constructor):
    with pytest.raises(TypeError, match=r"block shape|block dimensions"):
        constructor()


def test_full_rejects_block_fill_value():
    with pytest.raises(TypeError, match="full value must be scalar"):
        from mytriton.type_inference import TypeInference

        TypeInference().infer(
            tl.full((4,), tl.zeros((4,), tl.float32), tl.float32).expr
        )


def test_mlir_rejects_empty(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "mlir")
    out = np.zeros(5, dtype=np.float32)

    empty_kernel.clear_cache()
    with pytest.raises(TypeError, match=r"MLIR MVP does not support tl\.empty"):
        empty_kernel[(1,)](out, 5, BLOCK=8)


@pytest.mark.codegen
def test_mlir_lowers_full_and_zeros(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "mlir")
    out = np.zeros(5, dtype=np.float32)

    full_and_zeros_kernel.clear_cache()
    _, _, mlir_src = full_and_zeros_kernel[(1,)](out, 5, BLOCK=8)

    assert "%c_f32_0 = arith.constant 0.000000e+00 : f32" in mlir_src
    assert "%c_f32_2_5 = arith.constant 2.500000e+00 : f32" in mlir_src
    assert "%v2 = arith.addf %c_f32_0, %c_f32_2_5 : f32" in mlir_src
