import math

import pytest

from mytriton.optim import ConstantFoldPass, CSEPass, DCEPass, PassManager
from mytriton.ssa import SSAOp, SSAValue
from mytriton.trace import BOOL, F32, I32, PTR_F32, Const, Param, PointerType


def folded_store_operand(opcode, lhs, rhs):
    value = SSAValue(0, F32)
    out = Param("out", PTR_F32)
    ops = [
        SSAOp(opcode, (Const(lhs), Const(rhs)), value),
        SSAOp("store", (out, value, None)),
    ]

    optimized = ConstantFoldPass().run(ops)

    assert len(optimized) == 1
    assert optimized[0].opcode == "store"
    return optimized[0].operands[1]


def test_constant_fold_select_true():
    condition = Const(True)
    true_value = Param("x", F32)
    false_value = Const(0.0)
    selected = SSAValue(0, F32)
    out = Param("out", PTR_F32)
    ops = [
        SSAOp("select", (condition, true_value, false_value), selected),
        SSAOp("store", (out, selected, None)),
    ]

    assert ConstantFoldPass().run(ops) == [
        SSAOp("store", (out, true_value, None)),
    ]


def test_constant_fold_select_same_arms():
    condition = Param("condition", BOOL)
    value = Param("x", F32)
    selected = SSAValue(0, F32)
    out = Param("out", PTR_F32)
    ops = [
        SSAOp("select", (condition, value, value), selected),
        SSAOp("store", (out, selected, None)),
    ]

    assert ConstantFoldPass().run(ops) == [
        SSAOp("store", (out, value, None)),
    ]


def test_constant_fold_does_not_simplify_float_add_zero():
    value = Param("x", F32)
    result = SSAValue(0, F32)
    out = Param("out", PTR_F32)
    ops = [
        SSAOp("add", (value, Const(0.0)), result),
        SSAOp("store", (out, result, None)),
    ]

    assert ConstantFoldPass().run(ops) == ops


@pytest.mark.parametrize("opcode", ["maximum", "minimum"])
def test_constant_fold_extrema_propagates_right_hand_nan(opcode):
    folded = folded_store_operand(opcode, 2.0, float("nan"))

    assert isinstance(folded, Const)
    assert math.isnan(folded.value)


@pytest.mark.parametrize("opcode", ["maximum", "minimum"])
@pytest.mark.parametrize(
    ("lhs", "rhs", "expected_sign"),
    [
        (0.0, -0.0, -1.0),
        (-0.0, 0.0, 1.0),
    ],
)
def test_constant_fold_extrema_choose_rhs_for_equal_values(
    opcode,
    lhs,
    rhs,
    expected_sign,
):
    folded = folded_store_operand(opcode, lhs, rhs)

    assert isinstance(folded, Const)
    assert folded.value == 0.0
    assert math.copysign(1.0, folded.value) == expected_sign


def test_cse_reuses_duplicate_pure_ops():
    out = Param("out", PointerType(I32))
    first = SSAValue(0, I32)
    second = SSAValue(1, I32)
    ops = [
        SSAOp("add", (Const(1), Const(2)), first),
        SSAOp("add", (Const(1), Const(2)), second),
        SSAOp("store", (out, second, None)),
    ]

    assert CSEPass().run(ops) == [
        SSAOp("add", (Const(1), Const(2)), first),
        SSAOp("store", (out, first, None)),
    ]


def test_cse_distinguishes_positive_and_negative_zero_constants():
    out = Param("out", PTR_F32)
    first = SSAValue(0, F32)
    second = SSAValue(1, F32)
    ops = [
        SSAOp("add", (Const(0.0), Const(1.0)), first),
        SSAOp("add", (Const(-0.0), Const(1.0)), second),
        SSAOp("store", (out, second, None)),
    ]

    assert CSEPass().run(ops) == ops


def test_dce_removes_dead_ops_but_keeps_store_dependencies():
    out = Param("out", PointerType(I32))
    live = SSAValue(0, I32)
    used = SSAValue(1, I32)
    dead = SSAValue(2, I32)
    ops = [
        SSAOp("add", (Const(1), Const(2)), live),
        SSAOp("mul", (live, Const(3)), used),
        SSAOp("add", (Const(4), Const(5)), dead),
        SSAOp("store", (out, used, None)),
    ]

    assert DCEPass().run(ops) == [
        SSAOp("add", (Const(1), Const(2)), live),
        SSAOp("mul", (live, Const(3)), used),
        SSAOp("store", (out, used, None)),
    ]


class NoopPass:
    def run(self, ops):
        return ops


class CountingVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, ops):
        self.calls.append(list(ops))
        return ops


def test_pass_manager_verifies_after_every_pass():
    ops = [SSAOp("store", (Param("out", PTR_F32), Const(1.0), None))]
    verifier = CountingVerifier()

    assert PassManager([NoopPass(), NoopPass()], verifier).run(ops) == ops
    assert verifier.calls == [ops, ops]
