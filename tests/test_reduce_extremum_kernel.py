from __future__ import annotations

from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def row_max_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(x + offsets, mask=mask, other=float("-inf"))

    total = tl.max(values)

    first_lane = cols < 1

    tl.store(out + row, total, mask=first_lane)


@triton.jit
def row_min_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(x + offsets, mask=mask, other=float("inf"))

    total = tl.min(values)

    first_lane = cols < 1

    tl.store(out + row, total, mask=first_lane)


def row_extremum(x, cp, kernel):
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
        raise ValueError("current row_extremum supports at most 1024 columns")

    out = cp.empty(rows, dtype=cp.float32)

    kernel[(rows,)](
        x,
        out,
        cols,
        BLOCK=block,
    )

    return out


@pytest.mark.parametrize(
    (
        "kernel",
        "opcode",
        "fallback_ssa",
        "fallback_cuda",
        "reduction_line",
    ),
    [
        (
            row_max_kernel,
            "max",
            "-inf",
            "(-__int_as_float(0x7f800000))",
            "reduce_smem_7[threadIdx.x] = fmaxf(reduce_smem_7[threadIdx.x], reduce_smem_7[threadIdx.x + stride_7]);",
        ),
        (
            row_min_kernel,
            "min",
            "inf",
            "__int_as_float(0x7f800000)",
            "reduce_smem_7[threadIdx.x] = fminf(reduce_smem_7[threadIdx.x], reduce_smem_7[threadIdx.x + stride_7]);",
        ),
    ],
)
def test_row_extremum_kernel_lowering(
    kernel,
    opcode,
    fallback_ssa,
    fallback_cuda,
    reduction_line,
):
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

    _, ssa_ops, cuda_src = kernel[grid](
        x,
        out,
        cols,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        f"""\
        %0 = program_id {{axis=0}} : i32
        %1 = mul %0, n_cols : i32
        %2 = arange {{start=0, end=8}} : vector<8 x i32>
        %3 = add %1, %2 : vector<8 x i32>
        %4 = addptr x, %3 : vector<8 x ptr<f32>>
        %5 = cmp_lt %2, n_cols : vector<8 x bool>
        %6 = load %4, %5, {fallback_ssa} : vector<8 x f32>
        %7 = {opcode} %6 : f32
        %8 = addptr out, %0 : ptr<f32>
        %9 = cmp_lt %2, 1 : vector<8 x bool>
        store %8, %7, %9
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        f"""\
        extern "C" __global__
        void {kernel.fn.__name__}(float* x, float* out, int n_cols) {{
            __shared__ float reduce_smem_7[8];

            int v0 = blockIdx.x;
            int v1 = (v0 * n_cols);
            int v2 = threadIdx.x;
            int v3 = (v1 + v2);
            bool v5 = (v2 < n_cols);
            float v6 = (v5 ? x[v3] : {fallback_cuda});
            reduce_smem_7[threadIdx.x] = v6;
            __syncthreads();
            for (int stride_7 = 4; stride_7 > 0; stride_7 >>= 1) {{
                if (threadIdx.x < stride_7) {{
                    {reduction_line}
                }}
                __syncthreads();
            }}
            float v7 = reduce_smem_7[0];
            bool v9 = (v2 < 1);
            if (v9) {{
                out[v0] = v7;
            }}
        }}
        """
    ).rstrip("\n")


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (row_max_kernel, lambda cp, x: cp.max(x, axis=1)),
        (row_min_kernel, lambda cp, x: cp.min(x, axis=1)),
    ],
)
def test_row_extremum_kernel_cuda_execution(cp, kernel, expected):
    rows = 127
    cols = 513

    x = cp.random.randn(rows, cols, dtype=cp.float32)

    actual = row_extremum(x, cp, kernel)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, expected(cp, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (row_max_kernel, lambda cp, x: x[:, 0]),
        (row_min_kernel, lambda cp, x: x[:, 0]),
    ],
)
def test_row_extremum_single_column_cuda_execution(cp, kernel, expected):
    rows = 127
    x = cp.random.randn(rows, 1, dtype=cp.float32)

    actual = row_extremum(x, cp, kernel)

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(actual, expected(cp, x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kernel", [row_max_kernel, row_min_kernel])
def test_row_extremum_rejects_zero_columns(kernel):
    x = np.empty((3, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one column"):
        row_extremum(x, np, kernel)
