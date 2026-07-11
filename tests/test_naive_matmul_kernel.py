from __future__ import annotations

from textwrap import dedent
from typing import Any

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter
from mytriton.trace import (
    AddPtr,
    Arange,
    BinOp,
    Const,
    Load,
    Param,
    PointerType,
    ProgramId,
    ScalarType,
    Store,
    trace,
)


@triton.jit
def naive_matmul_kernel(a, b, c, n_cols, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK + tl.arange(0, BLOCK)

    mask = cols < n_cols
    accumulator: Any = 0.0

    a_row_start = row * K

    for k in tl.static_range(0, K):
        a_value = tl.load(a + a_row_start + k)

        b_offsets = k * n_cols + cols
        b_values = tl.load(
            b + b_offsets,
            mask=mask,
            other=0.0,
        )

        accumulator = accumulator + a_value * b_values

    c_offsets = row * n_cols + cols

    tl.store(
        c + c_offsets,
        accumulator,
        mask=mask,
    )


def naive_matmul(a, b, cp, block=256):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects two matrices")

    if a.dtype != cp.float32 or b.dtype != cp.float32:
        raise TypeError("only float32 is supported")

    if not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError("matrices must be C-contiguous")

    m, k = a.shape
    b_k, n = b.shape

    if k != b_k:
        raise ValueError(f"incompatible shapes: {a.shape} and {b.shape}")

    if k == 0:
        raise ValueError("matrices must have at least one reduction column")

    if n == 0:
        raise ValueError("matrix must have at least one output column")

    c = cp.empty((m, n), dtype=cp.float32)

    naive_matmul_kernel[(m, triton.cdiv(n, block))](
        a,
        b,
        c,
        n,
        K=k,
        BLOCK=block,
    )

    return c


def test_naive_matmul_kernel_trace():
    rows = 2
    cols = 5
    K = 3
    BLOCK = 4

    a = np.empty((rows, K), dtype=np.float32)
    b = np.empty((K, cols), dtype=np.float32)
    c = np.empty((rows, cols), dtype=np.float32)

    bound = naive_matmul_kernel.signature.bind(a, b, c, cols, K=K, BLOCK=BLOCK)

    ops, _ = trace(
        naive_matmul_kernel.fn,
        naive_matmul_kernel.signature,
        bound.arguments,
    )

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    i32 = ScalarType(name="i32")
    a_param = Param(name="a", ty=f32_ptr)
    b_param = Param(name="b", ty=f32_ptr)
    c_param = Param(name="c", ty=f32_ptr)
    n_cols_param = Param(name="n_cols", ty=i32)
    row = ProgramId(axis=0)
    col_block = ProgramId(axis=1)
    cols_expr = BinOp(
        op="+",
        lhs=BinOp(op="*", lhs=col_block, rhs=Const(value=4)),
        rhs=Arange(start=0, end=4),
    )
    mask = BinOp(op="<", lhs=cols_expr, rhs=n_cols_param)
    a_row_base = AddPtr(
        base=a_param,
        offset=BinOp(op="*", lhs=row, rhs=Const(value=3)),
    )

    def dot_term(k: int) -> tuple[Load, Load, BinOp]:
        a_value = Load(
            ptr=AddPtr(base=a_row_base, offset=Const(value=k)),
            mask=None,
            other=None,
        )
        b_value = Load(
            ptr=AddPtr(
                base=b_param,
                offset=BinOp(
                    op="+",
                    lhs=BinOp(op="*", lhs=Const(value=k), rhs=n_cols_param),
                    rhs=cols_expr,
                ),
            ),
            mask=mask,
            other=Const(value=0.0),
        )
        return a_value, b_value, BinOp(op="*", lhs=a_value, rhs=b_value)

    a0, b0, term0 = dot_term(0)
    a1, b1, term1 = dot_term(1)
    a2, b2, term2 = dot_term(2)

    accumulator = BinOp(op="+", lhs=Const(value=0.0), rhs=term0)
    accumulator = BinOp(op="+", lhs=accumulator, rhs=term1)
    accumulator = BinOp(op="+", lhs=accumulator, rhs=term2)
    c_offsets = BinOp(
        op="+",
        lhs=BinOp(op="*", lhs=row, rhs=n_cols_param),
        rhs=cols_expr,
    )

    expected_ops = [
        a0,
        b0,
        a1,
        b1,
        a2,
        b2,
        Store(
            ptr=AddPtr(base=c_param, offset=c_offsets),
            value=accumulator,
            mask=mask,
        ),
    ]

    assert ops == expected_ops


def test_naive_matmul_kernel_lowering():
    rows = 2
    cols = 5
    K = 3
    BLOCK = 4

    a = np.empty((rows, K), dtype=np.float32)
    b = np.empty((K, cols), dtype=np.float32)
    c = np.empty((rows, cols), dtype=np.float32)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (rows, triton.cdiv(cols, meta["BLOCK"]))

    _, ssa_ops, cuda_src = naive_matmul_kernel[grid](
        a,
        b,
        c,
        cols,
        K=K,
        BLOCK=BLOCK,
    )

    assert received_meta == {"K": K, "BLOCK": BLOCK}

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 3 : i32
        %2 = addptr a, %1 : ptr<f32>
        %3 = addptr %2, 0 : ptr<f32>
        %4 = load %3, none, none : f32
        %5 = mul 0, n_cols : i32
        %6 = program_id {axis=1} : i32
        %7 = mul %6, 4 : i32
        %8 = arange {start=0, end=4} : vector<4 x i32>
        %9 = add %7, %8 : vector<4 x i32>
        %10 = add %5, %9 : vector<4 x i32>
        %11 = addptr b, %10 : vector<4 x ptr<f32>>
        %12 = cmp_lt %9, n_cols : vector<4 x bool>
        %13 = load %11, %12, 0.0 : vector<4 x f32>
        %15 = addptr %2, 1 : ptr<f32>
        %16 = load %15, none, none : f32
        %17 = mul 1, n_cols : i32
        %18 = add %17, %9 : vector<4 x i32>
        %19 = addptr b, %18 : vector<4 x ptr<f32>>
        %20 = load %19, %12, 0.0 : vector<4 x f32>
        %22 = addptr %2, 2 : ptr<f32>
        %23 = load %22, none, none : f32
        %24 = mul 2, n_cols : i32
        %25 = add %24, %9 : vector<4 x i32>
        %26 = addptr b, %25 : vector<4 x ptr<f32>>
        %27 = load %26, %12, 0.0 : vector<4 x f32>
        %28 = mul %4, %13 : vector<4 x f32>
        %29 = add 0.0, %28 : vector<4 x f32>
        %30 = mul %16, %20 : vector<4 x f32>
        %31 = add %29, %30 : vector<4 x f32>
        %32 = mul %23, %27 : vector<4 x f32>
        %33 = add %31, %32 : vector<4 x f32>
        %34 = mul %0, n_cols : i32
        %35 = add %34, %9 : vector<4 x i32>
        %36 = addptr c, %35 : vector<4 x ptr<f32>>
        store %36, %33, %12
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void naive_matmul_kernel(float* a, float* b, float* c, int n_cols) {
            int v0 = blockIdx.x;
            int v1 = (v0 * 3);
            float v4 = (true ? a[(v1 + 0)] : 0.0f);
            int v5 = (0 * n_cols);
            int v6 = blockIdx.y;
            int v7 = (v6 * 4);
            int v8 = threadIdx.x;
            int v9 = (v7 + v8);
            int v10 = (v5 + v9);
            bool v12 = (v9 < n_cols);
            float v13 = (v12 ? b[v10] : 0.0f);
            float v16 = (true ? a[(v1 + 1)] : 0.0f);
            int v17 = (1 * n_cols);
            int v18 = (v17 + v9);
            float v20 = (v12 ? b[v18] : 0.0f);
            float v23 = (true ? a[(v1 + 2)] : 0.0f);
            int v24 = (2 * n_cols);
            int v25 = (v24 + v9);
            float v27 = (v12 ? b[v25] : 0.0f);
            float v28 = (v4 * v13);
            float v29 = (0.0f + v28);
            float v30 = (v16 * v20);
            float v31 = (v29 + v30);
            float v32 = (v23 * v27);
            float v33 = (v31 + v32);
            int v34 = (v0 * n_cols);
            int v35 = (v34 + v9);
            if (v12) {
                c[v35] = v33;
            }
        }
        """
    ).rstrip("\n")


def test_naive_matmul_kernel_cuda_execution(cp):
    rows = 31
    inner = 17
    cols = 513
    block = 256

    a = cp.random.randn(rows, inner, dtype=cp.float32)
    b = cp.random.randn(inner, cols, dtype=cp.float32)

    actual = naive_matmul(a, b, cp, block=block)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, a @ b, rtol=1e-4, atol=1e-3)


def test_naive_matmul_rejects_incompatible_shapes():
    a = np.empty((2, 3), dtype=np.float32)
    b = np.empty((4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="incompatible shapes"):
        naive_matmul(a, b, np, block=4)


def test_naive_matmul_rejects_zero_reduction_columns():
    a = np.empty((2, 0), dtype=np.float32)
    b = np.empty((0, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one reduction column"):
        naive_matmul(a, b, np, block=4)


def test_naive_matmul_rejects_zero_output_columns():
    a = np.empty((2, 3), dtype=np.float32)
    b = np.empty((3, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one output column"):
        naive_matmul(a, b, np, block=4)
