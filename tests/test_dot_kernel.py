from __future__ import annotations

from textwrap import dedent

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def dot_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n

    a = tl.load(x + offsets, mask=mask, other=0.0)
    b = tl.load(y + offsets, mask=mask, other=0.0)

    total = tl.dot(a, b)

    first_lane = offsets < 1
    tl.store(out + 0, total, mask=first_lane)


def test_dot_kernel_lowering():
    n = 1000
    BLOCK = 1024

    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    out = np.empty(1, dtype=np.float32)

    _, ssa_ops, cuda_src = dot_kernel[(1,)](
        x,
        y,
        out,
        n,
        BLOCK=BLOCK,
    )

    expected_ssa = dedent(
        """\
        %0 = arange {start=0, end=1024} : vector<1024 x i32>
        %1 = cmp_lt %0, n : vector<1024 x bool>
        %2 = addptr x, %0 : vector<1024 x ptr<f32>>
        %3 = load %2, %1, 0.0 : vector<1024 x f32>
        %4 = addptr y, %0 : vector<1024 x ptr<f32>>
        %5 = load %4, %1, 0.0 : vector<1024 x f32>
        %6 = dot %3, %5 : f32
        %7 = cmp_lt %0, 1 : vector<1024 x bool>
        %8 = addptr out, 0 : ptr<f32>
        store %8, %6, %7
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    assert "dot_product_6" in cuda_src
    assert "__shared__ float dot_smem_6[1024];" in cuda_src
    assert "dot_smem_6[threadIdx.x] += dot_smem_6[threadIdx.x + stride_6];" in cuda_src


def test_dot_kernel_cuda_execution(cp):
    n = 1000
    BLOCK = 1024

    x = cp.random.randn(n, dtype=cp.float32)
    y = cp.random.randn(n, dtype=cp.float32)
    out = cp.empty(1, dtype=cp.float32)

    dot_kernel[(1,)](
        x,
        y,
        out,
        n,
        BLOCK=BLOCK,
    )

    cp.cuda.runtime.deviceSynchronize()

    expected = cp.sum(x * y)

    cp.testing.assert_allclose(
        out[0],
        expected,
        rtol=1e-4,
        atol=1e-3,
    )
