from __future__ import annotations

import numpy as np

import mytriton as triton
import mytriton.language as tl
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

    ops = add_kernel[grid](
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
