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
    Max,
    Param,
    PointerType,
    ProgramId,
    ScalarType,
    Store,
    Sum,
    UnaryOp,
    trace,
)


@triton.jit
def softmax_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(
        x + offsets,
        mask=mask,
        other=float("-inf"),
    )

    row_max = tl.max(values)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators)
    probabilities = numerators / denominator

    tl.store(out + offsets, probabilities, mask=mask)


def softmax(x, cp):
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
        raise ValueError("current softmax supports at most 1024 columns")

    out = cp.empty_like(x)

    softmax_kernel[(rows,)](
        x,
        out,
        cols,
        BLOCK=block,
    )

    return out


def test_softmax_kernel_trace():
    rows = 3
    cols = 5
    BLOCK = 8

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty_like(x)

    bound = softmax_kernel.signature.bind(x, out, cols, BLOCK=BLOCK)

    ops, _ = trace(
        softmax_kernel.fn,
        softmax_kernel.signature,
        bound.arguments,
    )

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    i32 = ScalarType(name="i32")
    x_param = Param(name="x", ty=f32_ptr)
    out_param = Param(name="out", ty=f32_ptr)
    n_cols_param = Param(name="n_cols", ty=i32)
    row = ProgramId(axis=0)
    cols_expr = Arange(start=0, end=8)
    offsets = BinOp(
        op="+",
        lhs=BinOp(op="*", lhs=row, rhs=n_cols_param),
        rhs=cols_expr,
    )
    mask = BinOp(op="<", lhs=cols_expr, rhs=n_cols_param)
    values = Load(
        ptr=AddPtr(base=x_param, offset=offsets),
        mask=mask,
        other=Const(value=float("-inf")),
    )
    row_max = Max(value=values)
    numerators = UnaryOp(op="exp", value=BinOp(op="-", lhs=values, rhs=row_max))
    denominator = Sum(value=numerators)
    probabilities = BinOp(op="/", lhs=numerators, rhs=denominator)

    expected_ops = [
        values,
        Store(
            ptr=AddPtr(base=out_param, offset=offsets),
            value=probabilities,
            mask=mask,
        ),
    ]

    assert ops == expected_ops


def test_softmax_kernel_lowering():
    rows = 3
    cols = 5
    BLOCK = 8

    x = np.empty((rows, cols), dtype=np.float32)
    out = np.empty_like(x)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (rows,)

    _, ssa_ops, cuda_src = softmax_kernel[grid](
        x,
        out,
        cols,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, n_cols : i32
        %2 = arange {start=0, end=8} : vector<8 x i32>
        %3 = add %1, %2 : vector<8 x i32>
        %4 = addptr x, %3 : vector<8 x ptr<f32>>
        %5 = cmp_lt %2, n_cols : vector<8 x bool>
        %6 = load %4, %5, -inf : vector<8 x f32>
        %7 = max %6 : f32
        %8 = sub %6, %7 : vector<8 x f32>
        %9 = exp %8 : vector<8 x f32>
        %10 = sum %9 : f32
        %11 = div %9, %10 : vector<8 x f32>
        %12 = addptr out, %3 : vector<8 x ptr<f32>>
        store %12, %11, %5
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void softmax_kernel(float* x, float* out, int n_cols) {
            __shared__ float reduce_smem_7[8];
            __shared__ float reduce_smem_10[8];

            int v0 = blockIdx.x;
            int v1 = (v0 * n_cols);
            int v2 = threadIdx.x;
            int v3 = (v1 + v2);
            bool v5 = (v2 < n_cols);
            float v6 = (v5 ? x[v3] : (-__int_as_float(0x7f800000)));
            reduce_smem_7[threadIdx.x] = v6;
            __syncthreads();
            for (int stride_7 = 4; stride_7 > 0; stride_7 >>= 1) {
                if (threadIdx.x < stride_7) {
                    reduce_smem_7[threadIdx.x] = fmaxf(reduce_smem_7[threadIdx.x], reduce_smem_7[threadIdx.x + stride_7]);
                }
                __syncthreads();
            }
            float v7 = reduce_smem_7[0];
            float v8 = (v6 - v7);
            float v9 = expf(v8);
            reduce_smem_10[threadIdx.x] = v9;
            __syncthreads();
            for (int stride_10 = 4; stride_10 > 0; stride_10 >>= 1) {
                if (threadIdx.x < stride_10) {
                    reduce_smem_10[threadIdx.x] += reduce_smem_10[threadIdx.x + stride_10];
                }
                __syncthreads();
            }
            float v10 = reduce_smem_10[0];
            float v11 = (v9 / v10);
            if (v5) {
                out[v3] = v11;
            }
        }
        """
    ).rstrip("\n")


def test_softmax_kernel_cuda_execution(cp):
    rows = 127
    cols = 513

    x = cp.random.randn(rows, cols, dtype=cp.float32)

    actual = softmax(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    shifted = x - cp.max(x, axis=1, keepdims=True)
    numerators = cp.exp(shifted)
    expected = numerators / cp.sum(numerators, axis=1, keepdims=True)
    cp.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_softmax_kernel_single_column_cuda_execution(cp):
    rows = 127
    x = cp.random.randn(rows, 1, dtype=cp.float32)

    actual = softmax(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, cp.ones_like(x), rtol=1e-5, atol=1e-6)


def test_softmax_kernel_large_values_cuda_execution(cp):
    x = cp.asarray(
        [
            [1000.0, 1001.0, 999.0, 998.0],
            [-1000.0, -999.0, -1001.0, -1002.0],
            [1234.0, 1234.0, 1233.0, 1232.0],
        ],
        dtype=cp.float32,
    )

    actual = softmax(x, cp)

    cp.cuda.runtime.deviceSynchronize()
    shifted = x - cp.max(x, axis=1, keepdims=True)
    numerators = cp.exp(shifted)
    expected = numerators / cp.sum(numerators, axis=1, keepdims=True)
    cp.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_softmax_rejects_zero_columns():
    x = np.empty((3, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one column"):
        softmax(x, np)
