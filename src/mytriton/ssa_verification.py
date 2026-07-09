from typing import ClassVar, NoReturn

from .ssa import SSAOp, SSAOperand, SSAValue
from .trace import (
    BOOL,
    F32,
    I32,
    BlockType,
    Const,
    Param,
    PointerType,
    ScalarType,
    Type,
)


class CompileError(Exception):
    pass


class SSAVerifier:
    ARITY: ClassVar[dict[str, int]] = {
        "program_id": 0,
        "arange": 0,
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "cmp_lt": 2,
        "neg": 1,
        "exp": 1,
        "maximum": 2,
        "minimum": 2,
        "addptr": 2,
        "load": 3,
        "store": 3,
        "select": 3,
        "sum": 1,
        "max": 1,
        "min": 1,
    }

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def fail(self, index: int, op: SSAOp, message: str) -> NoReturn:
        raise CompileError(f"ssa-verifier: op #{index} '{op.opcode}': {message}")

    def element_type(self, ty: Type):
        return ty.element if isinstance(ty, BlockType) else ty

    def width(self, ty: Type) -> int | None:
        return ty.size if isinstance(ty, BlockType) else None

    def operand_type(self, operand: SSAOperand) -> Type | None:
        if operand is None:
            return None
        if isinstance(operand, SSAValue):
            return operand.ty
        if isinstance(operand, Param):
            return operand.ty
        if isinstance(operand, Const):
            if isinstance(operand.value, bool):
                return BOOL
            if isinstance(operand.value, int):
                return I32
            if isinstance(operand.value, float):
                return F32

        raise TypeError(f"Unknown typed operand: {operand}")

    def require_operand_type(
        self,
        index: int,
        op: SSAOp,
        operand: SSAOperand,
        role: str,
    ) -> Type:
        ty = self.operand_type(operand)
        if ty is None:
            self.fail(index, op, f"{role} must not be none")

        return ty

    def result_type(self, index: int, op: SSAOp) -> Type:
        if op.result is None:
            self.fail(index, op, "missing result")

        return op.result.ty

    def block_shape(self, index, op, *types):
        shapes = {ty.shape for ty in types if isinstance(ty, BlockType)}

        if len(shapes) > 1:
            rendered = ", ".join(str(ty) for ty in types)
            self.fail(index, op, f"incompatible shapes: {rendered}")

        return next(iter(shapes), None)

    def with_shape(
        self,
        index: int,
        op: SSAOp,
        element: ScalarType | PointerType,
        *types: Type,
    ) -> Type:
        size = self.block_shape(index, op, *types)
        return BlockType(size, element) if size is not None else element

    def require_type(self, index: int, op: SSAOp, actual: Type, expected: Type) -> None:
        if actual != expected:
            self.fail(index, op, f"expected {expected}, got {actual}")

    def promote_numeric(self, index: int, op: SSAOp, *types: Type) -> ScalarType:
        elements = [self.element_type(ty) for ty in types]
        if any(element not in (I32, F32) for element in elements):
            rendered = " and ".join(str(ty) for ty in types)
            self.fail(index, op, f"cannot combine {rendered}")

        return F32 if F32 in elements else I32

    def is_convertible(self, source: Type, destination: ScalarType) -> bool:
        source_element = self.element_type(source)

        if source_element == destination:
            return True

        return source_element in (I32, F32) and destination in (I32, F32)

    def check_binary_numeric(self, index: int, op: SSAOp) -> None:
        lhs_ty = self.require_operand_type(index, op, op.operands[0], "lhs")
        rhs_ty = self.require_operand_type(index, op, op.operands[1], "rhs")
        result_ty = self.result_type(index, op)

        element = self.promote_numeric(index, op, lhs_ty, rhs_ty)
        if op.opcode == "cmp_lt":
            element = BOOL

        expected_ty = self.with_shape(index, op, element, lhs_ty, rhs_ty)
        self.require_type(index, op, result_ty, expected_ty)

    def check_unary(self, index: int, op: SSAOp) -> None:
        value_ty = self.require_operand_type(index, op, op.operands[0], "value")
        result_ty = self.result_type(index, op)
        value_element = self.element_type(value_ty)

        if op.opcode == "neg":
            if value_element not in (I32, F32):
                self.fail(index, op, f"cannot negate {value_ty}")
        elif op.opcode == "exp":
            if value_element != F32:
                self.fail(index, op, f"exp requires f32, got {value_ty}")
        else:
            self.fail(index, op, "unknown unary operation")

        self.require_type(index, op, result_ty, value_ty)

    def check_addptr(self, index: int, op: SSAOp) -> None:
        base_ty = self.require_operand_type(index, op, op.operands[0], "base")
        offset_ty = self.require_operand_type(index, op, op.operands[1], "offset")
        result_ty = self.result_type(index, op)
        base_element = self.element_type(base_ty)

        if not isinstance(base_element, PointerType):
            self.fail(index, op, f"invalid base {base_ty}")

        if self.element_type(offset_ty) != I32:
            self.fail(index, op, f"invalid offset {offset_ty}")

        expected_ty = self.with_shape(index, op, base_element, base_ty, offset_ty)
        self.require_type(index, op, result_ty, expected_ty)

    def check_load(self, index: int, op: SSAOp) -> None:
        ptr, mask, other = op.operands
        ptr_ty = self.require_operand_type(index, op, ptr, "pointer")
        ptr_element = self.element_type(ptr_ty)
        result_ty = self.result_type(index, op)
        shape_operands = [ptr_ty]

        if not isinstance(ptr_element, PointerType):
            self.fail(index, op, f"expected pointer, got {ptr_ty}")

        if mask is not None:
            mask_ty = self.require_operand_type(index, op, mask, "mask")
            if self.element_type(mask_ty) != BOOL:
                self.fail(index, op, f"mask must be bool, got {mask_ty}")
            shape_operands.append(mask_ty)

        if other is not None:
            other_ty = self.require_operand_type(index, op, other, "fallback")
            if not self.is_convertible(other_ty, ptr_element.element):
                self.fail(
                    index,
                    op,
                    f"fallback must be convertible to {ptr_element.element}, "
                    f"got {other_ty}",
                )
            shape_operands.append(other_ty)

        expected_ty = self.with_shape(
            index,
            op,
            ptr_element.element,
            *shape_operands,
        )
        self.require_type(index, op, result_ty, expected_ty)

    def check_store(self, index: int, op: SSAOp) -> None:
        ptr, value, mask = op.operands
        ptr_ty = self.require_operand_type(index, op, ptr, "pointer")
        value_ty = self.require_operand_type(index, op, value, "value")
        ptr_element = self.element_type(ptr_ty)
        shape_operands = [ptr_ty, value_ty]

        if not isinstance(ptr_element, PointerType):
            self.fail(index, op, f"expected pointer, got {ptr_ty}")

        if not self.is_convertible(value_ty, ptr_element.element):
            self.fail(
                index,
                op,
                f"stored value must be convertible to {ptr_element.element}, "
                f"got {value_ty}",
            )

        if mask is not None:
            mask_ty = self.require_operand_type(index, op, mask, "mask")

            if self.element_type(mask_ty) != BOOL:
                self.fail(index, op, f"mask must be bool, got {mask_ty}")

            shape_operands.append(mask_ty)

        self.block_shape(index, op, *shape_operands)

    def check_select(self, index: int, op: SSAOp) -> None:
        condition, true_value, false_value = op.operands
        condition_ty = self.require_operand_type(index, op, condition, "condition")
        true_ty = self.require_operand_type(index, op, true_value, "true value")
        false_ty = self.require_operand_type(index, op, false_value, "false value")
        result_ty = self.result_type(index, op)

        if self.element_type(condition_ty) != BOOL:
            self.fail(index, op, f"condition must be bool, got {condition_ty}")

        element = self.promote_numeric(index, op, true_ty, false_ty)
        expected_ty = self.with_shape(
            index,
            op,
            element,
            condition_ty,
            true_ty,
            false_ty,
        )
        self.require_type(index, op, result_ty, expected_ty)

    def check_reduction(self, index: int, op: SSAOp) -> None:
        value_ty = self.require_operand_type(index, op, op.operands[0], "value")
        result_ty = self.result_type(index, op)

        if not isinstance(value_ty, BlockType) or value_ty.rank != 1:
            self.fail(index, op, f"reduction expects rank-1 block, got {value_ty}")

        if value_ty.size != self.block_size:
            self.fail(
                index,
                op,
                f"reduction width {value_ty.size} does not match "
                f"CUDA block size {self.block_size}",
            )

        if value_ty.size & (value_ty.size - 1):
            self.fail(
                index,
                op,
                f"reduction width must be a power of two, got {value_ty.size}",
            )

        if value_ty.element not in (I32, F32):
            self.fail(index, op, f"cannot reduce elements of type {value_ty.element}")

        self.require_type(index, op, result_ty, value_ty.element)

    def verify(self, ops: list[SSAOp]) -> list[SSAOp]:
        defined: set[int] = set()

        for index, op in enumerate(ops):
            if op.opcode not in self.ARITY:
                self.fail(index, op, "unsupported operation")

            expected_arity = self.ARITY[op.opcode]

            if len(op.operands) != expected_arity:
                self.fail(
                    index,
                    op,
                    f"expected {expected_arity} operands, got {len(op.operands)}",
                )

            should_have_result = op.opcode != "store"

            if should_have_result != (op.result is not None):
                self.fail(index, op, "invalid result declaration")

            for operand in op.operands:
                if isinstance(operand, SSAValue) and operand.id not in defined:
                    self.fail(index, op, f"{operand} used before definition")

            if op.opcode == "program_id":
                axis = op.attrs.get("axis")

                if axis not in (0, 1, 2):
                    self.fail(index, op, f"invalid axis {axis}")

                self.require_type(index, op, self.result_type(index, op), I32)

            elif op.opcode == "arange":
                start = op.attrs.get("start")
                end = op.attrs.get("end")

                if type(start) is not int or type(end) is not int:
                    self.fail(index, op, "arange start and end must be integers")

                assert isinstance(start, int)
                assert isinstance(end, int)
                width = end - start

                if width <= 0:
                    self.fail(index, op, f"invalid range [{start}, {end})")

                expected_ty = BlockType((width,), I32)
                self.require_type(index, op, self.result_type(index, op), expected_ty)

                if width != self.block_size:
                    self.fail(
                        index,
                        op,
                        f"range width {width} does not match "
                        f"CUDA block size {self.block_size}",
                    )

            elif op.opcode in ("add", "sub", "mul", "div", "cmp_lt") or op.opcode in (
                "maximum",
                "minimum",
            ):
                self.check_binary_numeric(index, op)

            elif op.opcode in ("neg", "exp"):
                self.check_unary(index, op)

            elif op.opcode == "addptr":
                self.check_addptr(index, op)

            elif op.opcode == "load":
                self.check_load(index, op)

            elif op.opcode == "store":
                self.check_store(index, op)

            elif op.opcode == "select":
                self.check_select(index, op)

            elif op.opcode in ("sum", "max", "min"):
                self.check_reduction(index, op)

            if op.result is not None:
                if op.result.id in defined:
                    self.fail(index, op, f"duplicate definition of {op.result}")

                defined.add(op.result.id)

        return ops
