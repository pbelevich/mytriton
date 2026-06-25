from __future__ import annotations

from textwrap import dedent

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def sigmoid_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    b = 1.0 / (1.0 + tl.exp(-a))

    tl.store(out + offs, b, mask=mask)


def test_sigmoid_kernel_lowering():
    n = 1000
    block = 256
    x = np.empty(n, dtype=np.float32)
    out = np.empty_like(x)

    _, ssa_ops, cuda_src = sigmoid_kernel[
        lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    ](x, out, n, BLOCK=block)

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 256 : i32
        %2 = arange {start=0, end=256} : vector<256 x i32>
        %3 = add %1, %2 : vector<256 x i32>
        %4 = addptr x, %3 : vector<256 x ptr<f32>>
        %5 = cmp_lt %3, n : vector<256 x bool>
        %6 = load %4, %5, 0.0 : vector<256 x f32>
        %7 = neg %6 : vector<256 x f32>
        %8 = exp %7 : vector<256 x f32>
        %9 = add 1.0, %8 : vector<256 x f32>
        %10 = div 1.0, %9 : vector<256 x f32>
        %11 = addptr out, %3 : vector<256 x ptr<f32>>
        store %11, %10, %5
        """
    ).rstrip("\n")

    assert cuda_src == dedent(
        """\
        extern "C" __global__
        void sigmoid_kernel(float* x, float* out, int n) {
            int v0 = blockIdx.x;
            int v1 = (v0 * 256);
            int v2 = threadIdx.x;
            int v3 = (v1 + v2);
            bool v5 = (v3 < n);
            float v6 = (v5 ? x[v3] : 0.0f);
            float v7 = -(v6);
            float v8 = expf(v7);
            float v9 = (1.0f + v8);
            float v10 = (1.0f / v9);
            if (v5) {
                out[v3] = v10;
            }
        }
        """
    ).rstrip("\n")


def test_sigmoid_kernel_cuda_execution(cp):
    n = 1000
    block = 256
    x = cp.random.randn(n, dtype=cp.float32)
    out = cp.empty_like(x)

    sigmoid_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x,
        out,
        n,
        BLOCK=block,
    )

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, 1.0 / (1.0 + cp.exp(-x)), rtol=1e-5, atol=1e-6)
