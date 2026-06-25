from typing import ClassVar

from .trace import (
    BOOL,
    F32,
    I32,
    AddPtr,
    Arange,
    BinOp,
    Const,
    Load,
    Max,
    Maximum,
    Min,
    Minimum,
    Param,
    PointerType,
    ProgramId,
    ScalarType,
    Store,
    Sum,
    Type,
    UnaryOp,
    VectorType,
    Where,
)


class TypeInference:
    ARITHMETIC_OPS: ClassVar[set[str]] = {"+", "-", "*", "/"}

    def __init__(self):
        self.types: dict[int, Type] = {}

    def element_type(self, ty: Type) -> ScalarType | PointerType:
        return ty.element if isinstance(ty, VectorType) else ty

    def common_vector_size(self, *types: Type) -> int | None:
        sizes = {ty.size for ty in types if isinstance(ty, VectorType)}

        if len(sizes) > 1:
            rendered = ", ".join(str(ty) for ty in types)
            raise TypeError(f"Cannot broadcast: {rendered}")

        return next(iter(sizes), None)

    def with_shape(
        self,
        element: ScalarType | PointerType,
        *types: Type,
    ) -> Type:
        size = self.common_vector_size(*types)
        return VectorType(size, element) if size is not None else element

    def promote(self, lhs: Type, rhs: Type) -> ScalarType:
        lhs_element = self.element_type(lhs)
        rhs_element = self.element_type(rhs)

        if lhs_element not in (I32, F32) or rhs_element not in (I32, F32):
            raise TypeError(f"Cannot combine {lhs} and {rhs}")

        return F32 if F32 in (lhs_element, rhs_element) else I32

    def require_mask(self, mask: Type) -> None:
        if self.element_type(mask) != BOOL:
            raise TypeError(f"Mask must be bool, got {mask}")

    def require_convertible(
        self,
        source: Type,
        destination: ScalarType,
        *,
        context: str,
    ) -> None:
        source_element = self.element_type(source)

        if source_element == destination:
            return

        numeric = (I32, F32)
        if source_element in numeric and destination in numeric:
            return

        raise TypeError(f"{context} must be convertible to {destination}, got {source}")

    def infer(self, expr) -> Type:
        key = id(expr)
        ty: Type

        if key in self.types:
            return self.types[key]

        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                ty = BOOL
            elif isinstance(expr.value, int):
                ty = I32
            elif isinstance(expr.value, float):
                ty = F32
            else:
                raise TypeError(f"Unsupported constant: {expr.value!r}")

        elif isinstance(expr, Param):
            ty = expr.ty

        elif isinstance(expr, ProgramId):
            ty = I32

        elif isinstance(expr, Arange):
            size = expr.end - expr.start

            if size <= 0:
                raise TypeError(
                    f"arange requires end > start, got [{expr.start}, {expr.end})"
                )

            ty = VectorType(size, I32)

        elif isinstance(expr, BinOp):
            lhs = self.infer(expr.lhs)
            rhs = self.infer(expr.rhs)

            if expr.op == "<":
                self.promote(lhs, rhs)
                ty = self.with_shape(BOOL, lhs, rhs)

            elif expr.op in self.ARITHMETIC_OPS:
                element = self.promote(lhs, rhs)
                ty = self.with_shape(element, lhs, rhs)

            else:
                raise TypeError(f"Unsupported binary operator: {expr.op!r}")

        elif isinstance(expr, AddPtr):
            base = self.infer(expr.base)
            offset = self.infer(expr.offset)
            base_element = self.element_type(base)

            if not isinstance(base_element, PointerType):
                raise TypeError(f"Expected pointer, got {base}")

            if self.element_type(offset) != I32:
                raise TypeError(f"Pointer offset must be i32, got {offset}")

            ty = self.with_shape(base_element, base, offset)

        elif isinstance(expr, Load):
            ptr = self.infer(expr.ptr)
            ptr_element = self.element_type(ptr)

            if not isinstance(ptr_element, PointerType):
                raise TypeError(f"Cannot load from {ptr}")

            operands = [ptr]

            if expr.mask is not None:
                mask = self.infer(expr.mask)
                self.require_mask(mask)
                operands.append(mask)

            if expr.other is not None:
                other = self.infer(expr.other)
                self.require_convertible(
                    other,
                    ptr_element.element,
                    context="Load fallback",
                )
                operands.append(other)

            ty = self.with_shape(ptr_element.element, *operands)

        elif isinstance(expr, (Maximum, Minimum)):
            lhs = self.infer(expr.lhs)
            rhs = self.infer(expr.rhs)
            element = self.promote(lhs, rhs)
            ty = self.with_shape(element, lhs, rhs)

        elif isinstance(expr, UnaryOp):
            value = self.infer(expr.value)
            unary_element = self.element_type(value)
            if expr.op == "neg":
                if unary_element not in (I32, F32):
                    raise TypeError(f"Cannot negate {value}")
            elif expr.op == "exp":
                if unary_element != F32:
                    raise TypeError(f"exp requires f32, got {value}")
            else:
                raise TypeError(f"Unknown unary operation: {expr.op}")
            ty = value
        elif isinstance(expr, Where):
            condition = self.infer(expr.condition)
            if self.element_type(condition) != BOOL:
                raise TypeError(f"where condition must be bool, got {condition}")
            true_ty = self.infer(expr.true_value)
            false_ty = self.infer(expr.false_value)
            element = self.promote(true_ty, false_ty)
            ty = self.with_shape(element, condition, true_ty, false_ty)
        elif isinstance(expr, (Sum, Max, Min)):
            value_ty = self.infer(expr.value)
            if not isinstance(value_ty, VectorType):
                raise TypeError(f"{type(expr).__name__} expects vector, got {value_ty}")
            if value_ty.element not in (I32, F32):
                raise TypeError(f"cannot reduce elements of type {value_ty.element}")
            ty = value_ty.element
        else:
            raise TypeError(f"Cannot infer type of {expr}")

        self.types[key] = ty
        return ty

    def check_store(self, store: Store) -> None:
        ptr = self.infer(store.ptr)
        value = self.infer(store.value)
        ptr_element = self.element_type(ptr)

        if not isinstance(ptr_element, PointerType):
            raise TypeError(f"Cannot store to {ptr}")

        self.require_convertible(
            value,
            ptr_element.element,
            context="Stored value",
        )

        operands = [ptr, value]

        if store.mask is not None:
            mask = self.infer(store.mask)
            self.require_mask(mask)
            operands.append(mask)

        self.common_vector_size(*operands)
