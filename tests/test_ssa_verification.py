import pytest

from mytriton.ssa import SSAOp, SSAValue
from mytriton.ssa_verification import CompileError, SSAVerifier
from mytriton.trace import F32, I32, PTR_F32, Const, Param, PointerType, VectorType


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
