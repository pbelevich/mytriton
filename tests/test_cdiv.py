import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAOp


@triton.jit
def cdiv_kernel(out, size, block_size: tl.constexpr):
    blocks = tl.cdiv(size, block_size)
    tl.store(out, blocks)


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (0, 4, 0),
        (1, 4, 1),
        (4, 4, 1),
        (5, 4, 2),
    ],
)
def test_cdiv_computes_compile_time_ceil_division(lhs, rhs, expected):
    assert tl.cdiv(lhs, rhs) == expected
    assert triton.cdiv(lhs, rhs) == expected


def test_language_cdiv_accepts_triton_keyword_names():
    assert tl.cdiv(x=5, div=4) == 2


def test_cdiv_rejects_compile_time_zero_divisor():
    with pytest.raises(ZeroDivisionError):
        tl.cdiv(1, 0)


@pytest.mark.codegen
def test_ast_frontend_lowers_runtime_cdiv_to_integer_arithmetic():
    out = np.empty(1, dtype=np.float32)

    _, ssa_ops, cuda_src = cdiv_kernel[(1,)](out, 17, block_size=8)

    assert all(isinstance(op, SSAOp) for op in ssa_ops)
    assert [op.opcode for op in ssa_ops if isinstance(op, SSAOp)] == [
        "add",
        "sub",
        "div",
        "store",
    ]
    assert "int v0 = (size + 8);" in cuda_src
    assert "int v1 = (v0 - 1);" in cuda_src
    assert "int v2 = (v1 / 8);" in cuda_src
    assert "out[0] = static_cast<float>(v2);" in cuda_src


@pytest.mark.execution
@pytest.mark.parametrize(("size", "expected"), [(0, 0), (1, 1), (16, 2), (17, 3)])
def test_runtime_cdiv_executes_on_cuda(cp, size, expected):
    out = cp.empty(1, dtype=cp.float32)

    cdiv_kernel[(1,)](out, size, block_size=8)

    cp.testing.assert_array_equal(out, cp.asarray([expected], dtype=cp.float32))
