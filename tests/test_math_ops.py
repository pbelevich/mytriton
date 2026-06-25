from textwrap import dedent

import numpy as np
import pytest

import mytriton.language as tl
from mytriton.cuda_codegen import SSACUDACodegen
from mytriton.ssa import SSALowering, SSAPrinter
from mytriton.trace import (
    BOOL,
    F32,
    I32,
    PTR_F32,
    AddPtr,
    Arange,
    Load,
    Param,
    Store,
    Value,
    VectorType,
)
from mytriton.type_inference import TypeInference


@pytest.mark.parametrize(
    ("operation", "opcode", "symbol"),
    [
        (tl.minimum, "minimum", "<"),
        (tl.maximum, "maximum", ">"),
    ],
)
def test_extremum_lowering_and_cuda_codegen(operation, opcode, symbol):
    value = Param("value", F32)
    out = Param("out", PTR_F32)
    expression = operation(Value(value), 2.0).expr

    ssa_ops = SSALowering().lower([Store(out, expression, None)])

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        f"""\
        %0 = {opcode} value, 2.0 : f32
        store out, %0, none
        """
    ).rstrip("\n")

    assert SSACUDACodegen().generate("extremum", ssa_ops, [value, out]) == dedent(
        f"""\
        extern "C" __global__
        void extremum(float value, float* out) {{
            float v0 = (isnan(value) ? (value) : (isnan(2.0f) ? (2.0f) : ((value) {symbol} (2.0f) ? (value) : (2.0f))));
            out[0] = v0;
        }}
        """
    ).rstrip("\n")


@pytest.mark.parametrize(
    ("operation", "opcode", "symbol"),
    [
        (tl.minimum, "minimum", "<"),
        (tl.maximum, "maximum", ">"),
    ],
)
def test_integer_extremum_does_not_emit_float_nan_checks(operation, opcode, symbol):
    value = Param("value", I32)
    out = Param("out", PTR_F32)
    expression = operation(Value(value), 2).expr

    ssa_ops = SSALowering().lower([Store(out, expression, None)])
    cuda_src = SSACUDACodegen().generate("extremum", ssa_ops, [value, out])

    assert SSAPrinter().print_ops(ssa_ops).splitlines()[0] == (
        f"%0 = {opcode} value, 2 : i32"
    )
    assert f"int v0 = ((value) {symbol} (2) ? (value) : (2));" in cuda_src
    assert "isnan" not in cuda_src


@pytest.mark.parametrize("operation", [tl.minimum, tl.maximum])
def test_extremum_broadcasts_scalar_over_vector(operation):
    offsets = Arange(0, 4)
    values = Load(AddPtr(Param("x", PTR_F32), offsets), None, None)
    expression = operation(Value(values), 0.0).expr

    assert TypeInference().infer(expression) == VectorType(4, F32)


@pytest.mark.parametrize("operation", [tl.minimum, tl.maximum])
def test_extremum_promotes_mixed_numeric_types(operation):
    expression = operation(Value(Param("value", I32)), 2.0).expr

    assert TypeInference().infer(expression) == F32


@pytest.mark.parametrize("operation", [tl.minimum, tl.maximum])
def test_extremum_rejects_boolean_operands(operation):
    expression = operation(Value(Param("condition", BOOL)), True).expr

    with pytest.raises(TypeError, match="Cannot combine bool and bool"):
        TypeInference().infer(expression)


@pytest.mark.parametrize(
    ("operation", "opcode", "cuda_expression"),
    [
        (lambda value: -value, "neg", "-(value)"),
        (tl.exp, "exp", "expf(value)"),
    ],
)
def test_unary_lowering_and_cuda_codegen(operation, opcode, cuda_expression):
    value = Param("value", F32)
    out = Param("out", PTR_F32)
    expression = operation(Value(value)).expr

    ssa_ops = SSALowering().lower([Store(out, expression, None)])

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        f"""\
        %0 = {opcode} value : f32
        store out, %0, none
        """
    ).rstrip("\n")

    assert SSACUDACodegen().generate("unary", ssa_ops, [value, out]) == dedent(
        f"""\
        extern "C" __global__
        void unary(float value, float* out) {{
            float v0 = {cuda_expression};
            out[0] = v0;
        }}
        """
    ).rstrip("\n")


def test_integer_negation_preserves_integer_type():
    expression = (-Value(Param("value", I32))).expr

    assert TypeInference().infer(expression) == I32


def test_negation_rejects_boolean_operand():
    expression = (-Value(Param("value", BOOL))).expr

    with pytest.raises(TypeError, match="Cannot negate bool"):
        TypeInference().infer(expression)


@pytest.mark.parametrize("operation", [lambda value: -value, tl.exp])
def test_unary_operation_preserves_vector_shape(operation):
    offsets = Arange(0, 4)
    values = Load(AddPtr(Param("x", PTR_F32), offsets), None, None)
    expression = operation(Value(values)).expr

    assert TypeInference().infer(expression) == VectorType(4, F32)


def test_exp_rejects_integer_operand():
    value = Param("value", I32)
    out = Param("out", PTR_F32)
    expression = tl.exp(Value(value)).expr

    with pytest.raises(TypeError, match="exp requires f32, got i32"):
        SSALowering().lower([Store(out, expression, None)])


def test_where_lowering_and_cuda_codegen():
    condition = Param("condition", BOOL)
    true_value = Param("true_value", F32)
    false_value = Param("false_value", F32)
    out = Param("out", PTR_F32)
    expression = tl.where(
        Value(condition),
        Value(true_value),
        Value(false_value),
    ).expr

    ssa_ops = SSALowering().lower([Store(out, expression, None)])

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = select condition, true_value, false_value : f32
        store out, %0, none
        """
    ).rstrip("\n")

    assert SSACUDACodegen().generate(
        "select_kernel",
        ssa_ops,
        [condition, true_value, false_value, out],
    ) == dedent(
        """\
        extern "C" __global__
        void select_kernel(bool condition, float true_value, float false_value, float* out) {
            float v0 = (condition ? true_value : false_value);
            out[0] = v0;
        }
        """
    ).rstrip("\n")


@pytest.mark.parametrize(
    ("operation", "numpy_operation", "kernel_name"),
    [
        (tl.minimum, np.minimum, "minimum"),
        (tl.maximum, np.maximum, "maximum"),
    ],
)
def test_extremum_cuda_execution(cp, operation, numpy_operation, kernel_name):
    lhs = Param("lhs", F32)
    rhs = Param("rhs", F32)
    out = Param("out", PTR_F32)
    expression = operation(Value(lhs), Value(rhs)).expr
    ssa_ops = SSALowering().lower([Store(out, expression, None)])
    cuda_src = SSACUDACodegen().generate(kernel_name, ssa_ops, [lhs, rhs, out])
    kernel = cp.RawKernel(cuda_src, kernel_name, options=("--std=c++14",))
    device_out = cp.empty(1, dtype=cp.float32)

    cases = [
        (np.nan, 2.0),
        (2.0, np.nan),
        (0.0, -0.0),
        (-0.0, 0.0),
        (3.0, 2.0),
        (2.0, 3.0),
    ]
    for lhs_value, rhs_value in cases:
        kernel(
            (1,),
            (1,),
            (np.float32(lhs_value), np.float32(rhs_value), device_out),
        )
        actual = cp.asnumpy(device_out)[0]
        expected = numpy_operation(np.float32(lhs_value), np.float32(rhs_value))

        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == expected
            assert np.signbit(actual) == np.signbit(expected)
