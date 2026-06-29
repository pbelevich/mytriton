from __future__ import annotations

from textwrap import dedent

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
    Sum,
    trace,
)


@triton.jit
def row_sum_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(x + offsets, mask=mask, other=0.0)

    total = tl.sum(values)

    first_lane = cols < 1

    tl.store(out + row, total, mask=first_lane)


def row_sum(x, cp):
    if x.ndim != 2:
        raise ValueError(f"expected matrix, got shape {x.shape}")

    if x.dtype != cp.float32:
        raise TypeError("only float32 is supported")

    if not x.flags.c_contiguous:
        raise ValueError("matrix must be C-contiguous")

    rows, cols = x.shape
    if cols == 0:
        raise ValueError("matrix must have at least one column")

    block = triton.next_power_of_2(cols)

    if block > 1024:
        raise ValueError("current row_sum supports at most 1024 columns")

    out = cp.empty(rows, dtype=cp.float32)

    row_sum_kernel[(rows,)](
        x,
        out,
        cols,
        BLOCK=block,
    )

    return out


def test_reduce_sum_kernel_trace():
    rows = 3
    cols = 5
    BLOCK = 8

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)

    bound = row_sum_kernel.signature.bind(x, out, cols, BLOCK=BLOCK)

    ops, _ = trace(
        row_sum_kernel.fn,
        row_sum_kernel.signature,
        bound.arguments,
    )

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    i32 = ScalarType(name="i32")
    x_param = Param(name="x", ty=f32_ptr)
    out_param = Param(name="out", ty=f32_ptr)
    n_cols_param = Param(name="n_cols", ty=i32)
    row = ProgramId(axis=0)
    cols_expr = Arange(start=0, end=8)
    row_start = BinOp(op="*", lhs=row, rhs=n_cols_param)
    offsets = BinOp(
        op="+",
        lhs=row_start,
        rhs=cols_expr,
    )
    mask = BinOp(op="<", lhs=cols_expr, rhs=n_cols_param)
    load_ptr = AddPtr(base=x_param, offset=offsets)
    values = Load(
        ptr=load_ptr,
        mask=mask,
        other=Const(value=0.0),
    )
    total = Sum(value=values)
    store_mask = BinOp(op="<", lhs=cols_expr, rhs=Const(value=1))
    out_ptr = AddPtr(base=out_param, offset=row)
    store = Store(ptr=out_ptr, value=total, mask=store_mask)

    expected_ops = [
        row,
        cols_expr,
        row_start,
        offsets,
        mask,
        load_ptr,
        values,
        total,
        store_mask,
        out_ptr,
        store,
    ]

    assert ops == expected_ops


def test_reduce_sum_kernel_lowering():
    rows = 3
    cols = 5
    BLOCK = 8

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (rows,)

    _, ssa_ops, cuda_src = row_sum_kernel[grid](
        x,
        out,
        cols,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = arange {start=0, end=8} : vector<8 x i32>
        %2 = mul %0, n_cols : i32
        %3 = add %2, %1 : vector<8 x i32>
        %4 = cmp_lt %1, n_cols : vector<8 x bool>
        %5 = addptr x, %3 : vector<8 x ptr<f32>>
        %6 = load %5, %4, 0.0 : vector<8 x f32>
        %7 = sum %6 : f32
        %8 = cmp_lt %1, 1 : vector<8 x bool>
        %9 = addptr out, %0 : ptr<f32>
        store %9, %7, %8
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void row_sum_kernel(float* x, float* out, int n_cols) {
            __shared__ float reduce_smem_7[8];

            int v0 = blockIdx.x;
            int v1 = threadIdx.x;
            int v2 = (v0 * n_cols);
            int v3 = (v2 + v1);
            bool v4 = (v1 < n_cols);
            float v6 = (v4 ? x[v3] : 0.0f);
            reduce_smem_7[threadIdx.x] = v6;
            __syncthreads();
            for (int stride_7 = 4; stride_7 > 0; stride_7 >>= 1) {
                if (threadIdx.x < stride_7) {
                    reduce_smem_7[threadIdx.x] += reduce_smem_7[threadIdx.x + stride_7];
                }
                __syncthreads();
            }
            float v7 = reduce_smem_7[0];
            bool v8 = (v1 < 1);
            if (v8) {
                out[v0] = v7;
            }
        }
        """
    ).rstrip("\n")


def test_reduce_sum_kernel_cuda_execution(cp):
    rows = 127
    cols = 513

    x = cp.random.randn(rows, cols, dtype=cp.float32)

    actual = row_sum(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, cp.sum(x, axis=1), rtol=1e-5, atol=1e-5)


def test_reduce_sum_kernel_single_column_cuda_execution(cp):
    rows = 127
    x = cp.random.randn(rows, 1, dtype=cp.float32)

    actual = row_sum(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, x[:, 0], rtol=1e-5, atol=1e-6)


def test_row_sum_rejects_zero_columns():
    x = np.empty((3, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one column"):
        row_sum(x, np)
