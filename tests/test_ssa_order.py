from __future__ import annotations

import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def eager_order_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = xv + 1.0

    tl.store(out + offsets, yv, mask=mask)


def test_eager_order_kernel_load_is_lowered_before_use():
    x = np.empty(128, dtype=np.float32)
    out = np.empty_like(x)

    _, ssa_ops, _ = eager_order_kernel[(1,)](
        x,
        out,
        128,
        BLOCK=128,
    )

    ssa_lines = SSAPrinter().print_ops(ssa_ops).splitlines()

    assert ssa_lines.index(
        "%6 = load %5, %4, 0.0 : vector<128 x f32>"
    ) < ssa_lines.index("%7 = add %6, 1.0 : vector<128 x f32>")
    assert ssa_lines[-1] == "store %8, %7, %4"
