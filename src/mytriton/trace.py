from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, get_args, get_origin

import numpy as np

# ----------------------------
# Language API
# ----------------------------


if TYPE_CHECKING:
    constexpr: TypeAlias = Any
else:

    class constexpr:
        pass


def is_constexpr_annotation(annotation: object) -> bool:
    if annotation is constexpr:
        return True

    return get_origin(annotation) is Annotated and constexpr in get_args(annotation)[1:]


def program_id(axis: int) -> Value:
    return Value(Builder.record(ProgramId(axis)))


def arange(start: int, end: int) -> Value:
    return Value(Builder.record(Arange(start, end)))


def load(
    ptr: Ptr,
    mask: Value | bool | None = None,
    other: Value | int | float | None = None,
) -> Value:
    return Value(
        Builder.record(
            Load(
                ptr=unwrap(ptr),
                mask=unwrap(mask) if mask is not None else None,
                other=unwrap(other) if other is not None else None,
            )
        )
    )


def store(
    ptr: Ptr,
    value: Value | int | float,
    mask: Value | bool | None = None,
) -> None:
    Builder.record(
        Store(
            ptr=unwrap(ptr),
            value=unwrap(value),
            mask=unwrap(mask) if mask is not None else None,
        )
    )


def maximum(lhs: Value | int | float, rhs: Value | int | float) -> Value:
    return Value(Builder.record(Maximum(unwrap(lhs), unwrap(rhs))))


def minimum(lhs: Value | int | float, rhs: Value | int | float) -> Value:
    return Value(Builder.record(Minimum(unwrap(lhs), unwrap(rhs))))


def exp(value: Value | float) -> Value:
    return Value(Builder.record(UnaryOp("exp", unwrap(value))))


def where(
    condition: Value | bool,
    true_value: Value | int | float,
    false_value: Value | int | float,
) -> Value:
    return Value(
        Builder.record(
            Where(unwrap(condition), unwrap(true_value), unwrap(false_value))
        )
    )


def sum(value: Value) -> Value:
    return Value(Builder.record(Sum(unwrap(value))))


def max(value: Value) -> Value:
    return Value(Builder.record(Max(unwrap(value))))


def min(value: Value) -> Value:
    return Value(Builder.record(Min(unwrap(value))))


def static_range(start: int, stop: int | None = None, step: int = 1) -> range:
    if stop is None:
        start, stop = 0, start

    for name, value in (
        ("start", start),
        ("stop", stop),
        ("step", step),
    ):
        if not isinstance(value, int):
            raise ValueError(
                f"static_range {name} must be compile-time int, "
                f"got {type(value).__name__}"
            )

    return range(start, stop, step)


def dot(lhs: Value, rhs: Value) -> Value:
    return Value(Builder.record(Dot(unwrap(lhs), unwrap(rhs))))


# ----------------------------
# Types
# ----------------------------


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class PointerType:
    element: ScalarType
    address_space: str = "global"

    def __str__(self):
        if self.address_space == "global":
            return f"ptr<{self.element}>"

        return f"ptr<{self.address_space}, {self.element}>"


@dataclass(frozen=True)
class VectorType:
    size: int
    element: ScalarType | PointerType

    def __str__(self):
        return f"vector<{self.size} x {self.element}>"


def _prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


@dataclass(frozen=True)
class BlockType:
    shape: tuple[int, ...]
    element: ScalarType | PointerType

    def __post_init__(self):
        if not self.shape:
            raise TypeError("0D BlockType is forbidden")

        for dim in self.shape:
            if type(dim) is not int or dim <= 0:
                raise TypeError(f"invalid block dimension: {dim!r}")

    @property
    def numel(self) -> int:
        return _prod(self.shape)

    def __str__(self):
        dims = " x ".join(str(dim) for dim in self.shape)
        return f"block<{dims} x {self.element}>"


def is_block_type(ty: Type) -> bool:
    return isinstance(ty, (VectorType, BlockType))


def block_shape(ty: Type) -> tuple[int, ...] | None:
    if isinstance(ty, VectorType):
        return (ty.size,)

    if isinstance(ty, BlockType):
        return ty.shape

    return None


def block_numel(ty: Type) -> int | None:
    shape = block_shape(ty)
    return None if shape is None else _prod(shape)


def type_element(ty: Type) -> ScalarType | PointerType:
    if isinstance(ty, VectorType):
        return ty.element

    if isinstance(ty, BlockType):
        return ty.element

    return ty


def make_block_type(
    shape: tuple[int, ...],
    element: ScalarType | PointerType,
) -> Type:
    if len(shape) == 1:
        return VectorType(shape[0], element)

    return BlockType(shape, element)


Type = ScalarType | PointerType | VectorType | BlockType


I32 = ScalarType("i32")
F32 = ScalarType("f32")
BOOL = ScalarType("bool")
PTR_F32 = PointerType(F32)


@dataclass
class Const:
    value: Any


@dataclass
class Param:
    name: str
    ty: ScalarType | PointerType


@dataclass
class ProgramId:
    axis: int


@dataclass
class Arange:
    start: int
    end: int


@dataclass
class BinOp:
    op: str
    lhs: Any
    rhs: Any


@dataclass
class AddPtr:
    base: Any
    offset: Any


@dataclass
class Load:
    ptr: Any
    mask: Any | None
    other: Any | None


@dataclass
class Store:
    ptr: Any
    value: Any
    mask: Any | None


@dataclass
class Maximum:
    lhs: Any
    rhs: Any


@dataclass
class Minimum:
    lhs: Any
    rhs: Any


@dataclass
class UnaryOp:
    op: str
    value: Any


@dataclass
class Where:
    condition: Any
    true_value: Any
    false_value: Any


@dataclass
class Sum:
    value: Any


@dataclass
class Max:
    value: Any


@dataclass
class Min:
    value: Any


@dataclass
class Dot:
    lhs: Any
    rhs: Any


# ----------------------------
# Symbolic values
# ----------------------------


def unwrap(x: Any) -> Any:
    if isinstance(x, Value):
        return x.expr
    if isinstance(x, Ptr):
        return x.expr
    if isinstance(x, (int, float)):
        return Const(x)
    return x


class Value:
    def __init__(self, expr: Any) -> None:
        self.expr = expr

    def __add__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("+", self.expr, unwrap(other))))

    def __radd__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("+", unwrap(other), self.expr)))

    def __sub__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("-", self.expr, unwrap(other))))

    def __rsub__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("-", unwrap(other), self.expr)))

    def __mul__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("*", self.expr, unwrap(other))))

    def __rmul__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("*", unwrap(other), self.expr)))

    def __lt__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("<", self.expr, unwrap(other))))

    def __truediv__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("/", self.expr, unwrap(other))))

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return Value(Builder.record(BinOp("/", unwrap(other), self.expr)))

    def __neg__(self) -> Value:
        return Value(Builder.record(UnaryOp("neg", self.expr)))

    def __bool__(self) -> bool:
        raise TypeError("Python control flow over symbolic values is not supported")

    def __repr__(self) -> str:
        return f"Value({self.expr})"


class Ptr:
    def __init__(self, expr: Any) -> None:
        self.expr = expr

    def __add__(self, offset: Value | int) -> Ptr:
        return Ptr(Builder.record(AddPtr(self.expr, unwrap(offset))))

    def __repr__(self) -> str:
        return f"Ptr({self.expr})"


class Builder:
    _current = None

    def __init__(self):
        self.ops = []

    def __enter__(self):
        Builder._current = self
        return self

    def __exit__(self, exc_type, exc, tb):
        Builder._current = None

    @staticmethod
    def current():
        assert Builder._current is not None
        return Builder._current

    @staticmethod
    def record(node):
        if Builder._current is not None:
            Builder._current.ops.append(node)
        return node


def _make_param(name, value) -> Param:
    if hasattr(value, "dtype") and hasattr(value, "flags"):
        if str(value.dtype) != "float32":
            raise TypeError(f"{name}: only float32 arrays are supported")
        if not value.flags.c_contiguous:
            raise TypeError(f"{name}: only C-contiguous arrays are supported")
        return Param(name, PTR_F32)

    if isinstance(value, (int, np.integer)):
        return Param(name, I32)

    if isinstance(value, (float, np.floating)):
        return Param(name, F32)

    raise TypeError(f"{name}: unsupported runtime value {type(value)}")


def make_runtime_params(signature, bound_args):
    return [
        _make_param(name, bound_args[name])
        for name, parameter in signature.parameters.items()
        if not is_constexpr_annotation(parameter.annotation)
    ]


def trace(fn, signature, bound_args, runtime_params=None):
    symbolic_arguments = {}
    if runtime_params is None:
        runtime_params = make_runtime_params(signature, bound_args)
    params_by_name = {param.name: param for param in runtime_params}

    for name, parameter in signature.parameters.items():
        value = bound_args[name]

        if is_constexpr_annotation(parameter.annotation):
            symbolic_arguments[name] = value
            continue

        param = params_by_name[name]

        if isinstance(param.ty, PointerType):
            symbolic_arguments[name] = Ptr(param)
        else:
            symbolic_arguments[name] = Value(param)

    symbolic_bound = signature.bind_partial()
    symbolic_bound.arguments.update(symbolic_arguments)

    with Builder() as builder:
        fn(*symbolic_bound.args, **symbolic_bound.kwargs)

    return builder.ops, runtime_params
