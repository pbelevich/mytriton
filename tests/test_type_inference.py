import pytest

from mytriton.trace import (
    BOOL,
    F32,
    I32,
    PTR_F32,
    AddPtr,
    Arange,
    BinOp,
    Const,
    Load,
    Param,
    PointerType,
    Store,
    VectorType,
)
from mytriton.type_inference import TypeInference


def test_promotes_and_broadcasts_scalar_over_vector():
    expr = BinOp("+", Arange(0, 4), Const(1.0))

    assert TypeInference().infer(expr) == VectorType(4, F32)


def test_rejects_incompatible_vector_sizes():
    expr = BinOp("+", Arange(0, 4), Arange(0, 8))

    with pytest.raises(TypeError, match="Cannot broadcast"):
        TypeInference().infer(expr)


def test_comparison_produces_boolean_vector():
    expr = BinOp("<", Arange(0, 4), Const(4))

    assert TypeInference().infer(expr) == VectorType(4, BOOL)


def test_load_shape_includes_vector_mask():
    mask = BinOp("<", Arange(0, 4), Const(4))
    expr = Load(Param("x", PTR_F32), mask, Const(0.0))

    assert TypeInference().infer(expr) == VectorType(4, F32)


def test_rejects_non_boolean_load_mask():
    expr = Load(Param("x", PTR_F32), Const(1), Const(0.0))

    with pytest.raises(TypeError, match="Mask must be bool"):
        TypeInference().infer(expr)


def test_rejects_incompatible_load_fallback_shape():
    ptr = AddPtr(Param("x", PTR_F32), Arange(0, 4))
    expr = Load(ptr, None, Arange(0, 8))

    with pytest.raises(TypeError, match="Cannot broadcast"):
        TypeInference().infer(expr)


def test_rejects_load_from_non_pointer():
    expr = Load(Const(1.0), None, None)

    with pytest.raises(TypeError, match="Cannot load from f32"):
        TypeInference().infer(expr)


def test_pointer_addition_produces_vector_of_pointers():
    expr = AddPtr(Param("x", PTR_F32), Arange(0, 4))

    assert TypeInference().infer(expr) == VectorType(4, PTR_F32)


def test_rejects_non_integer_pointer_offset():
    expr = AddPtr(Param("x", PTR_F32), Const(1.0))

    with pytest.raises(TypeError, match="Pointer offset must be i32"):
        TypeInference().infer(expr)


def test_rejects_pointer_addition_to_non_pointer():
    expr = AddPtr(Const(1), Const(1))

    with pytest.raises(TypeError, match="Expected pointer"):
        TypeInference().infer(expr)


def test_store_accepts_numeric_conversion_and_scalar_broadcast():
    ptr = AddPtr(Param("out", PTR_F32), Arange(0, 4))
    store = Store(ptr, Const(1), None)

    TypeInference().check_store(store)


def test_store_accepts_boolean_value_for_boolean_pointer():
    store = Store(Param("out", PointerType(BOOL)), Const(True), None)

    TypeInference().check_store(store)


def test_rejects_numeric_value_for_boolean_pointer():
    store = Store(Param("out", PointerType(BOOL)), Const(1), None)

    with pytest.raises(TypeError, match="Stored value must be convertible to bool"):
        TypeInference().check_store(store)


def test_rejects_non_boolean_store_mask():
    store = Store(Param("out", PTR_F32), Const(1.0), Const(1))

    with pytest.raises(TypeError, match="Mask must be bool"):
        TypeInference().check_store(store)


def test_rejects_store_to_non_pointer():
    store = Store(Const(0), Const(1.0), None)

    with pytest.raises(TypeError, match="Cannot store to i32"):
        TypeInference().check_store(store)


def test_rejects_incompatible_store_shapes():
    ptr = AddPtr(Param("out", PTR_F32), Arange(0, 4))
    store = Store(ptr, Arange(0, 8), None)

    with pytest.raises(TypeError, match="Cannot broadcast"):
        TypeInference().check_store(store)


def test_rejects_invalid_arange():
    with pytest.raises(TypeError, match="arange requires end > start"):
        TypeInference().infer(Arange(4, 4))


def test_rejects_unknown_binary_operator():
    expr = BinOp("%", Const(3), Const(2))

    with pytest.raises(TypeError, match="Unsupported binary operator"):
        TypeInference().infer(expr)


def test_integer_arithmetic_stays_integer():
    expr = BinOp("*", Const(2), Const(3))

    assert TypeInference().infer(expr) == I32
