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

    f32_ptr = PointerType(element=ScalarType(name="f32"), address_space="global")
    i32 = ScalarType(name="i32")
    x_param = Param(name="x", ty=f32_ptr)
    y_param = Param(name="y", ty=f32_ptr)
    out_param = Param(name="out", ty=f32_ptr)
    n_param = Param(name="n", ty=i32)
    pid = ProgramId(axis=0)
    block_start = BinOp(op="*", lhs=pid, rhs=Const(value=256))
    lanes = Arange(start=0, end=256)
    offs = BinOp(op="+", lhs=block_start, rhs=lanes)
    mask = BinOp(op="<", lhs=offs, rhs=n_param)
    x_ptr = AddPtr(base=x_param, offset=offs)
    a = Load(ptr=x_ptr, mask=mask, other=Const(value=0.0))
    y_ptr = AddPtr(base=y_param, offset=offs)
    b = Load(ptr=y_ptr, mask=mask, other=Const(value=0.0))
    out_ptr = AddPtr(base=out_param, offset=offs)
    value = BinOp(op="+", lhs=a, rhs=b)
    store = Store(ptr=out_ptr, value=value, mask=mask)

    expected_ops = [
        pid,
        block_start,
        lanes,
        offs,
        mask,
        x_ptr,
        a,
        y_ptr,
        b,
        out_ptr,
        value,
        store,
    ]

    assert ops == expected_ops

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 256 : i32
        %2 = arange {start=0, end=256} : vector<256 x i32>
        %3 = add %1, %2 : vector<256 x i32>
        %4 = cmp_lt %3, n : vector<256 x bool>
        %5 = addptr x, %3 : vector<256 x ptr<f32>>
        %6 = load %5, %4, 0.0 : vector<256 x f32>
        %7 = addptr y, %3 : vector<256 x ptr<f32>>
        %8 = load %7, %4, 0.0 : vector<256 x f32>
        %9 = addptr out, %3 : vector<256 x ptr<f32>>
        %10 = add %6, %8 : vector<256 x f32>
        store %9, %10, %4
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
            bool v4 = (v3 < n);
            float v6 = (v4 ? x[v3] : 0.0f);
            float v8 = (v4 ? y[v3] : 0.0f);
            float v10 = (v6 + v8);
            if (v4) {
                out[v3] = v10;
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
