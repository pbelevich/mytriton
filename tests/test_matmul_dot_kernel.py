from __future__ import annotations

import importlib

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def matmul_dot_kernel(a, b, c, n_cols, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    k = tl.arange(0, BLOCK)
    k_mask = k < K

    a_values = tl.load(
        a + row * K + k,
        mask=k_mask,
        other=0.0,
    )

    b_values = tl.load(
        b + k * n_cols + col,
        mask=k_mask,
        other=0.0,
    )

    value = tl.dot(a_values, b_values)

    first_lane = k < 1

    tl.store(
        c + row * n_cols + col,
        value,
        mask=first_lane,
    )


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def toy_matmul_dot(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.dtype == b.dtype
    assert str(a.dtype) == "float32"
    assert a.flags.c_contiguous
    assert b.flags.c_contiguous

    m, k = a.shape
    k2, n = b.shape
    assert k == k2

    block = next_power_of_2(k)
    if block > 1024:
        raise ValueError(f"K={k} is too large for one-block dot; BLOCK={block}")

    array_module = importlib.import_module(type(a).__module__.split(".", maxsplit=1)[0])
    c = array_module.empty((m, n), dtype=a.dtype)

    matmul_dot_kernel[(m, n)](
        a,
        b,
        c,
        n,
        k,
        BLOCK=block,
    )

    return c


def test_matmul_dot_kernel_lowering():
    m = 3
    k = 5
    n = 4
    block = 8

    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.empty((m, n), dtype=np.float32)

    _, ssa_ops, cuda_src = matmul_dot_kernel[(m, n)](
        a,
        b,
        c,
        n,
        k,
        BLOCK=block,
    )

    ssa = SSAPrinter().print_ops(ssa_ops)

    assert "dot" in ssa
    assert "%0 = program_id {axis=0} : i32" in ssa
    assert "%1 = program_id {axis=1} : i32" in ssa
    assert "arange {start=0, end=8}" in ssa
    assert "store" in ssa

    assert "dot_product" in cuda_src
    assert "__shared__ float dot_smem" in cuda_src
    assert "if (" in cuda_src


def test_matmul_dot_kernel_cuda_execution(cp):
    m = 7
    k = 11
    n = 9

    a = cp.random.randn(m, k, dtype=cp.float32)
    b = cp.random.randn(k, n, dtype=cp.float32)

    c = toy_matmul_dot(a, b)

    cp.cuda.runtime.deviceSynchronize()

    expected = a @ b

    cp.testing.assert_allclose(
        c,
        expected,
        rtol=1e-4,
        atol=1e-3,
    )
