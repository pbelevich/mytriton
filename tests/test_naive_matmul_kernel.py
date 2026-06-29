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
    col_start = BinOp(op="*", lhs=col_block, rhs=Const(value=4))
    lanes = Arange(start=0, end=4)
    cols_expr = BinOp(
        op="+",
        lhs=col_start,
        rhs=lanes,
    )
    mask = BinOp(op="<", lhs=cols_expr, rhs=n_cols_param)
    a_row_start = BinOp(op="*", lhs=row, rhs=Const(value=3))

    def dot_term(k: int):
        a_row_base = AddPtr(base=a_param, offset=a_row_start)
        a_ptr = AddPtr(base=a_row_base, offset=Const(value=k))
        a_value = Load(ptr=a_ptr, mask=None, other=None)
        b_row_start = BinOp(op="*", lhs=Const(value=k), rhs=n_cols_param)
        b_offsets = BinOp(op="+", lhs=b_row_start, rhs=cols_expr)
        b_ptr = AddPtr(base=b_param, offset=b_offsets)
        b_values = Load(ptr=b_ptr, mask=mask, other=Const(value=0.0))
        product = BinOp(op="*", lhs=a_value, rhs=b_values)
        return [
            a_row_base,
            a_ptr,
            a_value,
            b_row_start,
            b_offsets,
            b_ptr,
            b_values,
            product,
        ]

    term0 = dot_term(0)
    accumulator0 = BinOp(op="+", lhs=Const(value=0.0), rhs=term0[-1])
    term1 = dot_term(1)
    accumulator1 = BinOp(op="+", lhs=accumulator0, rhs=term1[-1])
    term2 = dot_term(2)
    accumulator2 = BinOp(op="+", lhs=accumulator1, rhs=term2[-1])
    c_row_start = BinOp(op="*", lhs=row, rhs=n_cols_param)
    c_offsets = BinOp(
        op="+",
        lhs=c_row_start,
        rhs=cols_expr,
    )
    c_ptr = AddPtr(base=c_param, offset=c_offsets)
    store = Store(ptr=c_ptr, value=accumulator2, mask=mask)

    expected_ops = [
        row,
        col_block,
        col_start,
        lanes,
        cols_expr,
        mask,
        a_row_start,
        *term0,
        accumulator0,
        *term1,
        accumulator1,
        *term2,
        accumulator2,
        c_row_start,
        c_offsets,
        c_ptr,
        store,
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
        %1 = program_id {axis=1} : i32
        %2 = mul %1, 4 : i32
        %3 = arange {start=0, end=4} : vector<4 x i32>
        %4 = add %2, %3 : vector<4 x i32>
        %5 = cmp_lt %4, n_cols : vector<4 x bool>
        %6 = mul %0, 3 : i32
        %7 = addptr a, %6 : ptr<f32>
        %8 = addptr %7, 0 : ptr<f32>
        %9 = load %8, none, none : f32
        %10 = mul 0, n_cols : i32
        %11 = add %10, %4 : vector<4 x i32>
        %12 = addptr b, %11 : vector<4 x ptr<f32>>
        %13 = load %12, %5, 0.0 : vector<4 x f32>
        %14 = mul %9, %13 : vector<4 x f32>
        %15 = add 0.0, %14 : vector<4 x f32>
        %17 = addptr %7, 1 : ptr<f32>
        %18 = load %17, none, none : f32
        %19 = mul 1, n_cols : i32
        %20 = add %19, %4 : vector<4 x i32>
        %21 = addptr b, %20 : vector<4 x ptr<f32>>
        %22 = load %21, %5, 0.0 : vector<4 x f32>
        %23 = mul %18, %22 : vector<4 x f32>
        %24 = add %15, %23 : vector<4 x f32>
        %26 = addptr %7, 2 : ptr<f32>
        %27 = load %26, none, none : f32
        %28 = mul 2, n_cols : i32
        %29 = add %28, %4 : vector<4 x i32>
        %30 = addptr b, %29 : vector<4 x ptr<f32>>
        %31 = load %30, %5, 0.0 : vector<4 x f32>
        %32 = mul %27, %31 : vector<4 x f32>
        %33 = add %24, %32 : vector<4 x f32>
        %34 = mul %0, n_cols : i32
        %35 = add %34, %4 : vector<4 x i32>
        %36 = addptr c, %35 : vector<4 x ptr<f32>>
        store %36, %33, %5
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void naive_matmul_kernel(float* a, float* b, float* c, int n_cols) {
            int v0 = blockIdx.x;
            int v1 = blockIdx.y;
            int v2 = (v1 * 4);
            int v3 = threadIdx.x;
            int v4 = (v2 + v3);
            bool v5 = (v4 < n_cols);
            int v6 = (v0 * 3);
            float v9 = (true ? a[(v6 + 0)] : 0.0f);
            int v10 = (0 * n_cols);
            int v11 = (v10 + v4);
            float v13 = (v5 ? b[v11] : 0.0f);
            float v14 = (v9 * v13);
            float v15 = (0.0f + v14);
            float v18 = (true ? a[(v6 + 1)] : 0.0f);
            int v19 = (1 * n_cols);
            int v20 = (v19 + v4);
            float v22 = (v5 ? b[v20] : 0.0f);
            float v23 = (v18 * v22);
            float v24 = (v15 + v23);
            float v27 = (true ? a[(v6 + 2)] : 0.0f);
            int v28 = (2 * n_cols);
            int v29 = (v28 + v4);
            float v31 = (v5 ? b[v29] : 0.0f);
            float v32 = (v27 * v31);
            float v33 = (v24 + v32);
            int v34 = (v0 * n_cols);
            int v35 = (v34 + v4);
            if (v5) {
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
