import pytest

from mytriton.ssa import SSAForRange, SSAOp, SSAValue
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


def test_verifier_rejects_unexpected_result():
    op = SSAOp(
        "store",
        (Param("out", PTR_F32), Const(1.0), None),
        SSAValue(0, F32),
    )

    with pytest.raises(
        CompileError, match="ssa-verifier: op #0 'store': unexpected result"
    ):
        verify([op])


def test_verifier_rejects_missing_result():
    op = SSAOp("add", (Const(1), Const(2)), None)

    with pytest.raises(CompileError, match="ssa-verifier: op #0 'add': missing result"):
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


def scalar_for_range(
    *,
    start=None,
    stop=None,
    step=None,
    carried_input=None,
    index_id=1,
    carried_arg_id=2,
    yielded=None,
    result_id=4,
    result_ty=I32,
):
    start = Const(0) if start is None else start
    stop = Param("stop", I32) if stop is None else stop
    step = Const(1) if step is None else step
    initial = SSAValue(0, I32)
    carried_input = initial if carried_input is None else carried_input
    index = SSAValue(index_id, I32)
    carried_arg = SSAValue(carried_arg_id, I32)
    body_result = SSAValue(3, I32)
    yielded = body_result if yielded is None else yielded
    result = SSAValue(result_id, result_ty)
    loop = SSAForRange(
        index=index,
        start=start,
        stop=stop,
        step=step,
        carried_inputs=(carried_input,),
        carried_args=(carried_arg,),
        body=[SSAOp("add", (carried_arg, index), body_result)],
        yields=(yielded,),
        results=(result,),
    )
    return [SSAOp("add", (Const(0), Const(0)), initial), loop]


def test_verifier_accepts_valid_for_range():
    ops = scalar_for_range()

    assert verify(ops) == ops


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stop": SSAValue(99, I32)}, "loop stop %99 used before definition"),
        (
            {"carried_input": SSAValue(99, I32)},
            "carried input %99 used before definition",
        ),
        ({"yielded": SSAValue(99, I32)}, "yield %99 used before definition"),
    ],
)
def test_verifier_rejects_for_range_use_before_definition(kwargs, message):
    with pytest.raises(CompileError, match=message):
        verify(scalar_for_range(**kwargs))


def test_verifier_rejects_for_range_carried_type_change():
    with pytest.raises(CompileError, match="yield expected i32, got f32"):
        verify(scalar_for_range(yielded=Const(1.0), result_ty=F32))


def test_verifier_rejects_non_i32_for_range_index():
    ops = scalar_for_range()
    loop = ops[-1]
    assert isinstance(loop, SSAForRange)
    loop.index = SSAValue(loop.index.id, F32)

    with pytest.raises(CompileError, match="loop index must be i32, got f32"):
        verify(ops)


@pytest.mark.parametrize("step", [Const(0), Const(-1)])
def test_verifier_rejects_non_positive_for_range_step(step):
    with pytest.raises(CompileError, match="loop step must be positive"):
        verify(scalar_for_range(step=step))


def test_verifier_rejects_dynamic_for_range_step():
    with pytest.raises(CompileError, match="loop step must be a constant integer"):
        verify(scalar_for_range(step=Param("step", I32)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index_id": 0}, "duplicate definition of loop index %0"),
        ({"carried_arg_id": 1}, "duplicate definition of carried argument %1"),
        ({"result_id": 3}, "duplicate definition of loop result %3"),
    ],
)
def test_verifier_rejects_duplicate_for_range_value_ids(kwargs, message):
    with pytest.raises(CompileError, match=message):
        verify(scalar_for_range(**kwargs))
