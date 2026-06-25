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
    Sum,
    trace,
)


@triton.jit
def long_row_sum_kernel(x, out, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)

    row_start = row * N_COLS
    partial: Any = 0.0

    for start in tl.static_range(0, N_COLS, BLOCK):
        cols = start + lanes
        mask = cols < N_COLS

        values = tl.load(x + row_start + cols, mask=mask, other=0.0)

        partial = partial + values

    total = tl.sum(partial)
    first_lane = lanes < 1

    tl.store(out + row, total, mask=first_lane)


def long_row_sum(x, cp):
    if x.ndim != 2:
        raise ValueError(f"expected matrix, got shape {x.shape}")

    if x.dtype != cp.float32:
        raise TypeError("only float32 is supported")

    if not x.flags.c_contiguous:
        raise ValueError("matrix must be C-contiguous")

    rows, cols = x.shape
    if cols == 0:
        raise ValueError("matrix must have at least one column")

    block = min(256, triton.next_power_of_2(cols))

    out = cp.empty(rows, dtype=cp.float32)

    long_row_sum_kernel[(rows,)](
        x,
        out,
        N_COLS=cols,
        BLOCK=block,
    )

    return out


def test_long_row_sum_kernel_trace():
    rows = 3
    cols = 10
    BLOCK = 4

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)

    bound = long_row_sum_kernel.signature.bind(
        x,
        out,
        N_COLS=cols,
        BLOCK=BLOCK,
    )

    ops, _ = trace(
        long_row_sum_kernel.fn,
        long_row_sum_kernel.signature,
        bound.arguments,
    )

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    x_param = Param(name="x", ty=f32_ptr)
    out_param = Param(name="out", ty=f32_ptr)
    row = ProgramId(axis=0)
    lanes = Arange(start=0, end=4)
    row_start = BinOp(op="*", lhs=row, rhs=Const(value=10))
    row_base = AddPtr(base=x_param, offset=row_start)

    cols0 = BinOp(op="+", lhs=Const(value=0), rhs=lanes)
    values0 = Load(
        ptr=AddPtr(base=row_base, offset=cols0),
        mask=BinOp(op="<", lhs=cols0, rhs=Const(value=10)),
        other=Const(value=0.0),
    )
    partial0 = BinOp(op="+", lhs=Const(value=0.0), rhs=values0)

    cols4 = BinOp(op="+", lhs=Const(value=4), rhs=lanes)
    values4 = Load(
        ptr=AddPtr(base=row_base, offset=cols4),
        mask=BinOp(op="<", lhs=cols4, rhs=Const(value=10)),
        other=Const(value=0.0),
    )
    partial4 = BinOp(op="+", lhs=partial0, rhs=values4)

    cols8 = BinOp(op="+", lhs=Const(value=8), rhs=lanes)
    values8 = Load(
        ptr=AddPtr(base=row_base, offset=cols8),
        mask=BinOp(op="<", lhs=cols8, rhs=Const(value=10)),
        other=Const(value=0.0),
    )
    partial8 = BinOp(op="+", lhs=partial4, rhs=values8)

    expected_ops = [
        Store(
            ptr=AddPtr(base=out_param, offset=row),
            value=Sum(value=partial8),
            mask=BinOp(op="<", lhs=lanes, rhs=Const(value=1)),
        )
    ]

    assert ops == expected_ops


def test_long_row_sum_kernel_lowering():
    rows = 3
    cols = 10
    BLOCK = 4

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty(rows, dtype=np.float32)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (rows,)

    _, ssa_ops, cuda_src = long_row_sum_kernel[grid](
        x,
        out,
        N_COLS=cols,
        BLOCK=BLOCK,
    )

    assert received_meta == {"N_COLS": cols, "BLOCK": BLOCK}

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 10 : i32
        %2 = addptr x, %1 : ptr<f32>
        %3 = arange {start=0, end=4} : vector<4 x i32>
        %4 = add 0, %3 : vector<4 x i32>
        %5 = addptr %2, %4 : vector<4 x ptr<f32>>
        %6 = cmp_lt %4, 10 : vector<4 x bool>
        %7 = load %5, %6, 0.0 : vector<4 x f32>
        %8 = add 0.0, %7 : vector<4 x f32>
        %10 = add 4, %3 : vector<4 x i32>
        %11 = addptr %2, %10 : vector<4 x ptr<f32>>
        %12 = cmp_lt %10, 10 : vector<4 x bool>
        %13 = load %11, %12, 0.0 : vector<4 x f32>
        %14 = add %8, %13 : vector<4 x f32>
        %16 = add 8, %3 : vector<4 x i32>
        %17 = addptr %2, %16 : vector<4 x ptr<f32>>
        %18 = cmp_lt %16, 10 : vector<4 x bool>
        %19 = load %17, %18, 0.0 : vector<4 x f32>
        %20 = add %14, %19 : vector<4 x f32>
        %21 = sum %20 : f32
        %22 = addptr out, %0 : ptr<f32>
        %23 = cmp_lt %3, 1 : vector<4 x bool>
        store %22, %21, %23
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void long_row_sum_kernel(float* x, float* out) {
            __shared__ float reduce_smem_21[4];

            int v0 = blockIdx.x;
            int v1 = (v0 * 10);
            int v3 = threadIdx.x;
            int v4 = (0 + v3);
            bool v6 = (v4 < 10);
            float v7 = (v6 ? x[(v1 + v4)] : 0.0f);
            float v8 = (0.0f + v7);
            int v10 = (4 + v3);
            bool v12 = (v10 < 10);
            float v13 = (v12 ? x[(v1 + v10)] : 0.0f);
            float v14 = (v8 + v13);
            int v16 = (8 + v3);
            bool v18 = (v16 < 10);
            float v19 = (v18 ? x[(v1 + v16)] : 0.0f);
            float v20 = (v14 + v19);
            reduce_smem_21[threadIdx.x] = v20;
            __syncthreads();
            for (int stride_21 = 2; stride_21 > 0; stride_21 >>= 1) {
                if (threadIdx.x < stride_21) {
                    reduce_smem_21[threadIdx.x] += reduce_smem_21[threadIdx.x + stride_21];
                }
                __syncthreads();
            }
            float v21 = reduce_smem_21[0];
            bool v23 = (v3 < 1);
            if (v23) {
                out[v0] = v21;
            }
        }
        """
    ).rstrip("\n")


def test_long_row_sum_kernel_cuda_execution(cp):
    rows = 31
    cols = 4097

    x = cp.random.randn(rows, cols, dtype=cp.float32)

    actual = long_row_sum(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, cp.sum(x, axis=1), rtol=1e-4, atol=1e-3)


def test_long_row_sum_kernel_single_column_cuda_execution(cp):
    rows = 31
    x = cp.random.randn(rows, 1, dtype=cp.float32)

    actual = long_row_sum(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, x[:, 0], rtol=1e-5, atol=1e-6)


def test_long_row_sum_rejects_zero_columns():
    x = np.empty((3, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one column"):
        long_row_sum(x, np)
