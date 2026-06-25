from __future__ import annotations

import mytriton as triton
import mytriton.language as tl


@triton.jit
def copy_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    tl.store(out + offs, a, mask=mask)


def test_copy_kernel_cuda_execution(cp):
    n = 1000
    block = 256
    x = cp.random.randn(n, dtype=cp.float32)
    out = cp.empty_like(x)

    copy_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x,
        out,
        n,
        BLOCK=block,
    )

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, x, rtol=1e-5, atol=1e-6)
