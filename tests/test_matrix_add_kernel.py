from __future__ import annotations

from textwrap import dedent

import numpy as np

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
)


@triton.jit
def matrix_add_kernel(x, y, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK + tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    lhs = tl.load(x + offsets, mask=mask, other=0.0)
    rhs = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, lhs + rhs, mask=mask)


def test_matrix_add_kernel():
    rows = 127
    cols = 513
    BLOCK = 256

    x = np.random.randn(rows, cols).astype(np.float32)
    y = np.random.randn(rows, cols).astype(np.float32)
    out = np.empty_like(x)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (rows, triton.cdiv(cols, meta["BLOCK"]))

    ops, ssa_ops, cuda_src = matrix_add_kernel[grid](
        x,
        y,
        out,
        cols,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    i32 = ScalarType(name="i32")
    x_param = Param(name="x", ty=f32_ptr)
    y_param = Param(name="y", ty=f32_ptr)
    out_param = Param(name="out", ty=f32_ptr)
    n_cols_param = Param(name="n_cols", ty=i32)
    row = ProgramId(axis=0)
    col_block = ProgramId(axis=1)
    col_start = BinOp(op="*", lhs=col_block, rhs=Const(value=256))
    lanes = Arange(start=0, end=256)
    cols_expr = BinOp(
        op="+",
        lhs=col_start,
        rhs=lanes,
    )
    row_start = BinOp(op="*", lhs=row, rhs=n_cols_param)
    offsets = BinOp(
        op="+",
        lhs=row_start,
        rhs=cols_expr,
    )
    mask = BinOp(op="<", lhs=cols_expr, rhs=n_cols_param)
    x_ptr = AddPtr(base=x_param, offset=offsets)
    lhs = Load(ptr=x_ptr, mask=mask, other=Const(value=0.0))
    y_ptr = AddPtr(base=y_param, offset=offsets)
    rhs = Load(ptr=y_ptr, mask=mask, other=Const(value=0.0))
    out_ptr = AddPtr(base=out_param, offset=offsets)
    value = BinOp(op="+", lhs=lhs, rhs=rhs)
    store = Store(ptr=out_ptr, value=value, mask=mask)

    expected_ops = [
        row,
        col_block,
        col_start,
        lanes,
        cols_expr,
        row_start,
        offsets,
        mask,
        x_ptr,
        lhs,
        y_ptr,
        rhs,
        out_ptr,
        value,
        store,
    ]

    assert ops == expected_ops

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = program_id {axis=1} : i32
        %2 = mul %1, 256 : i32
        %3 = arange {start=0, end=256} : vector<256 x i32>
        %4 = add %2, %3 : vector<256 x i32>
        %5 = mul %0, n_cols : i32
        %6 = add %5, %4 : vector<256 x i32>
        %7 = cmp_lt %4, n_cols : vector<256 x bool>
        %8 = addptr x, %6 : vector<256 x ptr<f32>>
        %9 = load %8, %7, 0.0 : vector<256 x f32>
        %10 = addptr y, %6 : vector<256 x ptr<f32>>
        %11 = load %10, %7, 0.0 : vector<256 x f32>
        %12 = addptr out, %6 : vector<256 x ptr<f32>>
        %13 = add %9, %11 : vector<256 x f32>
        store %12, %13, %7
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matrix_add_kernel(float* x, float* y, float* out, int n_cols) {
            int v0 = blockIdx.x;
            int v1 = blockIdx.y;
            int v2 = (v1 * 256);
            int v3 = threadIdx.x;
            int v4 = (v2 + v3);
            int v5 = (v0 * n_cols);
            int v6 = (v5 + v4);
            bool v7 = (v4 < n_cols);
            float v9 = (v7 ? x[v6] : 0.0f);
            float v11 = (v7 ? y[v6] : 0.0f);
            float v13 = (v9 + v11);
            if (v7) {
                out[v6] = v13;
            }
        }
    """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


def test_matrix_add_kernel_cuda_execution(cp):
    rows = 127
    cols = 513
    BLOCK = 256

    x = cp.random.randn(rows, cols, dtype=cp.float32)
    y = cp.random.randn(rows, cols, dtype=cp.float32)
    out = cp.empty_like(x)

    matrix_add_kernel[lambda meta: (rows, triton.cdiv(cols, meta["BLOCK"]))](
        x,
        y,
        out,
        cols,
        BLOCK=BLOCK,
    )

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, x + y, rtol=1e-5, atol=1e-6)
