from dataclasses import dataclass
from typing import Any

import numpy as np

# ----------------------------
# Language API
# ----------------------------


class constexpr:
    pass


def program_id(axis: int):
    return Value(ProgramId(axis))


def arange(start: int, end: int):
    return Value(Arange(start, end))


def load(ptr, mask=None, other=None):
    node = Load(
        ptr=unwrap(ptr),
        mask=unwrap(mask) if mask is not None else None,
        other=unwrap(other) if other is not None else None,
    )
    return Value(node)


def store(ptr, value, mask=None):
    node = Store(
        ptr=unwrap(ptr),
        value=unwrap(value),
        mask=unwrap(mask) if mask is not None else None,
    )
    Builder.current().ops.append(node)


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
    element: object
    address_space: str = "global"

    def __str__(self):
        if self.address_space == "global":
            return f"ptr<{self.element}>"

        return f"ptr<{self.address_space}, {self.element}>"


@dataclass(frozen=True)
class VectorType:
    size: int
    element: object

    def __str__(self):
        return f"vector<{self.size} x {self.element}>"


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


# ----------------------------
# Symbolic values
# ----------------------------


def unwrap(x):
    if isinstance(x, Value):
        return x.expr
    if isinstance(x, Ptr):
        return x.expr
    if isinstance(x, (int, float)):
        return Const(x)
    return x


class Value:
    def __init__(self, expr):
        self.expr = expr

    def __add__(self, other):
        return Value(BinOp("+", self.expr, unwrap(other)))

    def __radd__(self, other):
        return Value(BinOp("+", unwrap(other), self.expr))

    def __sub__(self, other):
        return Value(BinOp("-", self.expr, unwrap(other)))

    def __rsub__(self, other):
        return Value(BinOp("-", unwrap(other), self.expr))

    def __mul__(self, other):
        return Value(BinOp("*", self.expr, unwrap(other)))

    def __rmul__(self, other):
        return Value(BinOp("*", unwrap(other), self.expr))

    def __lt__(self, other):
        return Value(BinOp("<", self.expr, unwrap(other)))

    def __truediv__(self, other):
        return Value(BinOp("/", self.expr, unwrap(other)))

    def __rtruediv__(self, other):
        return Value(BinOp("/", unwrap(other), self.expr))

    def __bool__(self) -> bool:
        raise TypeError("Python control flow over symbolic values is not supported")

    def __repr__(self):
        return f"Value({self.expr})"


class Ptr:
    def __init__(self, expr):
        self.expr = expr

    def __add__(self, offset):
        return Ptr(AddPtr(self.expr, unwrap(offset)))

    def __repr__(self):
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


def trace(fn, signature, bound_args):
    symbolic_arguments = {}

    for name, parameter in signature.parameters.items():
        value = bound_args[name]

        if parameter.annotation is constexpr:
            symbolic_arguments[name] = value
            continue

        param = _make_param(name, value)

        if isinstance(param.ty, PointerType):
            symbolic_arguments[name] = Ptr(param)
        else:
            symbolic_arguments[name] = Value(param)

    symbolic_bound = signature.bind_partial()
    symbolic_bound.arguments.update(symbolic_arguments)

    with Builder() as builder:
        fn(*symbolic_bound.args, **symbolic_bound.kwargs)

    return builder.ops
