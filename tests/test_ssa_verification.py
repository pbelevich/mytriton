import pytest

from mytriton.ssa import SSAOp, SSAValue
from mytriton.ssa_verification import CompileError, SSAVerifier
from mytriton.trace import (
    F32,
    I32,
    PTR_F32,
    BlockType,
    Const,
    Param,
    PointerType,
    VectorType,
)


def verify(ops, block_size=1):
    return SSAVerifier(block_size).verify(ops)


def test_verifier_accepts_valid_scalar_program():
    out = Param("out", PointerType(I32))
    value = SSAValue(0, I32)
    ops = [
        SSAOp("add", (Const(1), Const(2)), value),
        SSAOp("store", (out, value, None)),
    ]

    assert verify(ops) == ops


def test_verifier_rejects_use_before_definition():
    out = Param("out", PTR_F32)

    with pytest.raises(CompileError, match="%0 used before definition"):
        verify([SSAOp("store", (out, SSAValue(0, F32), None))])


@pytest.mark.parametrize(
    "op",
    [
        SSAOp("store", (Param("out", PTR_F32), Const(1.0), None), SSAValue(0, F32)),
        SSAOp("add", (Const(1), Const(2)), None),
    ],
)
def test_verifier_rejects_invalid_result_declaration(op):
    with pytest.raises(CompileError, match="invalid result declaration"):
        verify([op])


def test_verifier_rejects_unsupported_opcode():
    with pytest.raises(CompileError, match="unsupported operation"):
        verify([SSAOp("reduce_sum", (Const(1),), SSAValue(0, I32))])


def test_verifier_rejects_wrong_arithmetic_result_type():
    op = SSAOp("add", (Const(1), Const(2.0)), SSAValue(0, I32))

    with pytest.raises(CompileError, match="expected f32, got i32"):
        verify([op])


def test_verifier_rejects_integer_exp():
    op = SSAOp("exp", (Const(1),), SSAValue(0, I32))

    with pytest.raises(CompileError, match="exp requires f32"):
        verify([op])


def test_verifier_rejects_non_bool_select_condition():
    op = SSAOp("select", (Const(1), Const(1.0), Const(2.0)), SSAValue(0, F32))

    with pytest.raises(CompileError, match="condition must be bool"):
        verify([op])


def test_verifier_rejects_wrong_select_result_shape():
    op = SSAOp(
        "select",
        (Const(True), Const(1.0), Const(2.0)),
        SSAValue(0, VectorType(4, F32)),
    )

    with pytest.raises(CompileError, match="expected f32"):
        verify([op])


def test_verifier_checks_load_fallback_conversion():
    value = SSAValue(0, F32)
    op = SSAOp("load", (Param("x", PTR_F32), None, Const(True)), value)

    with pytest.raises(CompileError, match="fallback must be convertible to f32"):
        verify([op])


def test_verifier_accepts_store_numeric_conversion():
    verify([SSAOp("store", (Param("out", PTR_F32), Const(1), None))])


def vector_load_ops(width: int):
    lanes = SSAValue(0, VectorType(width, I32))
    ptrs = SSAValue(1, VectorType(width, PTR_F32))
    values = SSAValue(2, VectorType(width, F32))
    return (
        [
            SSAOp("arange", result=lanes, attrs={"start": 0, "end": width}),
            SSAOp("addptr", (Param("x", PTR_F32), lanes), ptrs),
            SSAOp("load", (ptrs, None, None), values),
        ],
        values,
    )


@pytest.mark.parametrize("opcode", ["sum", "max", "min"])
def test_verifier_accepts_valid_reduction(opcode):
    ops, value = vector_load_ops(8)
    reduced = SSAValue(3, F32)
    ops.extend(
        [
            SSAOp(opcode, (value,), reduced),
            SSAOp("store", (Param("out", PTR_F32), reduced, None)),
        ]
    )

    assert verify(ops, block_size=8) == ops


def test_verifier_rejects_scalar_reduction_input():
    op = SSAOp("sum", (Const(1.0),), SSAValue(0, F32))

    with pytest.raises(CompileError, match="reduction expects rank-1 block"):
        verify([op])


def test_verifier_rejects_wrong_reduction_result_type():
    ops, value = vector_load_ops(8)
    op = SSAOp("sum", (value,), SSAValue(3, VectorType(8, F32)))

    with pytest.raises(CompileError, match="expected f32"):
        verify([*ops, op], block_size=8)


def test_verifier_rejects_reduction_width_mismatch():
    value = SSAValue(0, VectorType(8, F32))
    op = SSAOp("sum", (value,), SSAValue(1, F32))

    with pytest.raises(CompileError, match="does not match CUDA block size 4"):
        SSAVerifier(block_size=4).check_reduction(0, op)


def test_verifier_rejects_non_power_of_two_reduction_width():
    ops, value = vector_load_ops(6)
    op = SSAOp("sum", (value,), SSAValue(3, F32))

    with pytest.raises(CompileError, match="power of two"):
        verify([*ops, op], block_size=6)


def test_verifier_accepts_dot_k1():
    lhs_lanes = SSAValue(0, BlockType((4,), I32))
    lhs_ptrs = SSAValue(1, BlockType((4,), PTR_F32))
    lhs_vec = SSAValue(2, BlockType((4,), F32))
    lhs = SSAValue(3, BlockType((4, 1), F32))

    rhs_lanes = SSAValue(4, BlockType((8,), I32))
    rhs_ptrs = SSAValue(5, BlockType((8,), PTR_F32))
    rhs_vec = SSAValue(6, BlockType((8,), F32))
    rhs = SSAValue(7, BlockType((1, 8), F32))

    out = SSAValue(8, BlockType((4, 8), F32))

    ops = [
        SSAOp("arange", result=lhs_lanes, attrs={"start": 0, "end": 4}),
        SSAOp("addptr", (Param("a", PTR_F32), lhs_lanes), lhs_ptrs),
        SSAOp("load", (lhs_ptrs, None, None), lhs_vec),
        SSAOp("expand_dims", (lhs_vec,), lhs, attrs={"axis": 1}),
        SSAOp("arange", result=rhs_lanes, attrs={"start": 0, "end": 8}),
        SSAOp("addptr", (Param("b", PTR_F32), rhs_lanes), rhs_ptrs),
        SSAOp("load", (rhs_ptrs, None, None), rhs_vec),
        SSAOp("expand_dims", (rhs_vec,), rhs, attrs={"axis": 0}),
        SSAOp("dot", (lhs, rhs), out),
    ]

    verify(ops, block_size=8)


def test_verifier_rejects_dot_k_greater_than_one_for_now():
    lhs = SSAValue(0, BlockType((4, 16), F32))
    rhs = SSAValue(1, BlockType((16, 8), F32))
    out = SSAValue(2, BlockType((4, 8), F32))
    op = SSAOp("dot", (lhs, rhs), out)

    with pytest.raises(CompileError, match="dot MVP supports only K=1"):
        SSAVerifier(block_size=32).check_dot(0, op)
