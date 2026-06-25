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
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)
    b = tl.load(y + offs, mask=mask, other=0.0)

    tl.store(out + offs, a + b, mask=mask)


def test_add_kernel():
    n = 1000
    BLOCK = 256

    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    out = np.empty_like(x)

    received_meta = None

    def grid(meta):
        nonlocal received_meta
        received_meta = meta
        return (triton.cdiv(n, meta["BLOCK"]),)

    ops, ssa_ops, cuda_src = add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    expected_ops = [
        Store(
            ptr=AddPtr(
                base=Param(
                    name="out",
                    ty=PointerType(
                        element=ScalarType(name="f32"), address_space="global"
                    ),
                ),
                offset=BinOp(
                    op="+",
                    lhs=BinOp(op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)),
                    rhs=Arange(start=0, end=256),
                ),
            ),
            value=BinOp(
                op="+",
                lhs=Load(
                    ptr=AddPtr(
                        base=Param(
                            name="x",
                            ty=PointerType(
                                element=ScalarType(name="f32"), address_space="global"
                            ),
                        ),
                        offset=BinOp(
                            op="+",
                            lhs=BinOp(
                                op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)
                            ),
                            rhs=Arange(start=0, end=256),
                        ),
                    ),
                    mask=BinOp(
                        op="<",
                        lhs=BinOp(
                            op="+",
                            lhs=BinOp(
                                op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)
                            ),
                            rhs=Arange(start=0, end=256),
                        ),
                        rhs=Param(name="n", ty=ScalarType(name="i32")),
                    ),
                    other=Const(value=0.0),
                ),
                rhs=Load(
                    ptr=AddPtr(
                        base=Param(
                            name="y",
                            ty=PointerType(
                                element=ScalarType(name="f32"), address_space="global"
                            ),
                        ),
                        offset=BinOp(
                            op="+",
                            lhs=BinOp(
                                op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)
                            ),
                            rhs=Arange(start=0, end=256),
                        ),
                    ),
                    mask=BinOp(
                        op="<",
                        lhs=BinOp(
                            op="+",
                            lhs=BinOp(
                                op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)
                            ),
                            rhs=Arange(start=0, end=256),
                        ),
                        rhs=Param(name="n", ty=ScalarType(name="i32")),
                    ),
                    other=Const(value=0.0),
                ),
            ),
            mask=BinOp(
                op="<",
                lhs=BinOp(
                    op="+",
                    lhs=BinOp(op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)),
                    rhs=Arange(start=0, end=256),
                ),
                rhs=Param(name="n", ty=ScalarType(name="i32")),
            ),
        )
    ]

    assert ops == expected_ops

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 256 : i32
        %2 = arange {start=0, end=256} : vector<256 x i32>
        %3 = add %1, %2 : vector<256 x i32>
        %4 = addptr x, %3 : vector<256 x ptr<f32>>
        %5 = cmp_lt %3, n : vector<256 x bool>
        %6 = load %4, %5, 0.0 : vector<256 x f32>
        %7 = addptr y, %3 : vector<256 x ptr<f32>>
        %8 = load %7, %5, 0.0 : vector<256 x f32>
        %9 = add %6, %8 : vector<256 x f32>
        %10 = addptr out, %3 : vector<256 x ptr<f32>>
        store %10, %9, %5
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void add_kernel(float* x, float* y, float* out, int n) {
            int v0 = blockIdx.x;
            int v1 = (v0 * 256);
            int v2 = threadIdx.x;
            int v3 = (v1 + v2);
            bool v5 = (v3 < n);
            float v6 = (v5 ? x[v3] : 0.0f);
            float v8 = (v5 ? y[v3] : 0.0f);
            float v9 = (v6 + v8);
            if (v5) {
                out[v3] = v9;
            }
        }
    """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


def test_add_kernel_cuda_execution(cp):
    n = 1000
    block = 256
    x = cp.random.randn(n, dtype=cp.float32)
    y = cp.random.randn(n, dtype=cp.float32)
    out = cp.empty_like(x)

    add_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x,
        y,
        out,
        n,
        BLOCK=block,
    )

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, x + y, rtol=1e-5, atol=1e-6)
