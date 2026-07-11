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


@pytest.mark.codegen
def test_add_kernel_codegen(backend):
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

    add_kernel.clear_cache()
    ops, ssa_ops, src = add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK=BLOCK,
    )

    assert received_meta == {"BLOCK": BLOCK}

    x_load = Load(
        ptr=AddPtr(
            base=Param(
                name="x",
                ty=PointerType(element=ScalarType(name="f32"), address_space="global"),
            ),
            offset=BinOp(
                op="+",
                lhs=BinOp(op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)),
                rhs=Arange(start=0, end=256),
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
        other=Const(value=0.0),
    )
    y_load = Load(
        ptr=AddPtr(
            base=Param(
                name="y",
                ty=PointerType(element=ScalarType(name="f32"), address_space="global"),
            ),
            offset=BinOp(
                op="+",
                lhs=BinOp(op="*", lhs=ProgramId(axis=0), rhs=Const(value=256)),
                rhs=Arange(start=0, end=256),
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
        other=Const(value=0.0),
    )

    expected_ops = [
        x_load,
        y_load,
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
                lhs=x_load,
                rhs=y_load,
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
        ),
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

    if backend == "cuda":
        expected_src = dedent(
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
    else:
        expected_src = dedent(
            """\
            module attributes {gpu.container_module} {
              gpu.module @kernels {
                gpu.func @add_kernel(%x: memref<?xf32>, %y: memref<?xf32>, %out: memref<?xf32>, %n: i32) kernel {
                  %bid_x = gpu.block_id x
                  %tid_x = gpu.thread_id x
                  %block_id_x = arith.index_cast %bid_x : index to i32
                  %thread_id_x = arith.index_cast %tid_x : index to i32
                  %c_i32_256 = arith.constant 256 : i32
                  %v1 = arith.muli %block_id_x, %c_i32_256 : i32
                  %v3 = arith.addi %v1, %thread_id_x : i32
                  %idx4 = arith.index_cast %v3 : i32 to index
                  %v5 = arith.cmpi slt, %v3, %n : i32
                  %c_f32_0_0 = arith.constant 0.000000e+00 : f32
                  %v6 = scf.if %v5 -> (f32) {
                    %loaded6 = memref.load %x[%idx4] : memref<?xf32>
                    scf.yield %loaded6 : f32
                  } else {
                    scf.yield %c_f32_0_0 : f32
                  }
                  %idx7 = arith.index_cast %v3 : i32 to index
                  %v8 = scf.if %v5 -> (f32) {
                    %loaded8 = memref.load %y[%idx7] : memref<?xf32>
                    scf.yield %loaded8 : f32
                  } else {
                    scf.yield %c_f32_0_0 : f32
                  }
                  %v9 = arith.addf %v6, %v8 : f32
                  %idx10 = arith.index_cast %v3 : i32 to index
                  scf.if %v5 {
                    memref.store %v9, %out[%idx10] : memref<?xf32>
                  }
                  gpu.return
                }
              }
            }"""
        )

    assert src == expected_src


@pytest.mark.execution
def test_add_kernel_execution(cp, backend):
    n = 1000
    block = 256
    x = cp.random.randn(n, dtype=cp.float32)
    y = cp.random.randn(n, dtype=cp.float32)
    out = cp.empty_like(x)

    add_kernel.clear_cache()
    add_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        x,
        y,
        out,
        n,
        BLOCK=block,
    )

    cp.cuda.runtime.deviceSynchronize()
    cp.testing.assert_allclose(out, x + y, rtol=1e-5, atol=1e-6)
