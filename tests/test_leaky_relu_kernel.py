from __future__ import annotations

from textwrap import dedent

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def leaky_relu_kernel(x, out, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    b = tl.where(a < 0.0, alpha * a, a)

    tl.store(out + offs, b, mask=mask)


def test_leaky_relu_kernel_lowering():
    n = 1000
    block = 256
    x = np.empty(n, dtype=np.float32)
    out = np.empty_like(x)
    alpha = 0.01

    _, ssa_ops, cuda_src = leaky_relu_kernel[
        lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    ](x, out, n, alpha, BLOCK=block)

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 256 : i32
        %2 = arange {start=0, end=256} : vector<256 x i32>
        %3 = add %1, %2 : vector<256 x i32>
        %4 = cmp_lt %3, n : vector<256 x bool>
        %5 = addptr x, %3 : vector<256 x ptr<f32>>
        %6 = load %5, %4, 0.0 : vector<256 x f32>
        %7 = cmp_lt %6, 0.0 : vector<256 x bool>
        %8 = mul alpha, %6 : vector<256 x f32>
        %9 = select %7, %8, %6 : vector<256 x f32>
        %10 = addptr out, %3 : vector<256 x ptr<f32>>
        store %10, %9, %4
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void leaky_relu_kernel(float* x, float* out, int n, float alpha) {
            int v0 = blockIdx.x;
            int v1 = (v0 * 256);
            int v2 = threadIdx.x;
            int v3 = (v1 + v2);
            bool v4 = (v3 < n);
            float v6 = (v4 ? x[v3] : 0.0f);
            bool v7 = (v6 < 0.0f);
            float v8 = (alpha * v6);
            float v9 = (v7 ? v8 : v6);
            if (v4) {
                out[v3] = v9;
            }
        }
        """
    ).rstrip("\n")


def test_leaky_relu_kernel_cuda_execution(cp):
    n = 1000
    block = 256
    x = cp.random.randn(n, dtype=cp.float32)
    x[0] = cp.nan
    x[1] = -0.0
    x[2] = 0.0
    out = cp.empty_like(x)
    alpha = 0.01

    leaky_relu_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x,
        out,
        n,
        alpha,
        BLOCK=block,
    )

    cp.cuda.runtime.deviceSynchronize()
    expected = cp.where(x < 0.0, alpha * x, x)
    nan_mask = cp.isnan(expected)
    cp.testing.assert_array_equal(cp.isnan(out), nan_mask)
    cp.testing.assert_allclose(
        out[~nan_mask], expected[~nan_mask], rtol=1e-5, atol=1e-6
    )
    cp.testing.assert_array_equal(cp.signbit(out[1:3]), cp.signbit(expected[1:3]))
