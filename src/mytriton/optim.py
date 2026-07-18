import math
from collections.abc import Mapping
from typing import ClassVar

from .ssa import SSAForRange, SSAOp, SSAOperand, SSAValue
from .trace import BOOL, F32, I32, BlockType, Const, Param

# Optimization passes assume their input SSA has already passed SSAVerifier.
# PassManager verifies after every pass so broken rewrites fail before codegen.

OperandKey = tuple[object, ...]
CSEKey = tuple[
    str,
    tuple[OperandKey, ...],
    tuple[tuple[str, object], ...],
    object,
]


def const_key(value: object) -> tuple[object, ...]:
    if isinstance(value, float):
        return (float, value.hex())

    return (type(value), value)


def operand_key(operand: SSAOperand) -> OperandKey:
    if isinstance(operand, SSAValue):
        return ("ssa", operand.id)
    if isinstance(operand, Param):
        return ("param", operand.name, operand.ty)
    if isinstance(operand, Const):
        return ("const", *const_key(operand.value))
    if operand is None:
        return ("none",)

    raise TypeError(f"Unknown operand: {operand}")


def resolve_operand(
    operand: SSAOperand,
    replacements: Mapping[int, SSAOperand],
) -> SSAOperand:
    while isinstance(operand, SSAValue) and operand.id in replacements:
        operand = replacements[operand.id]

    return operand


class ConstantFoldPass:
    def scalar_type(self, ty):
        return ty.element if isinstance(ty, BlockType) else ty

    def is_integer_result(self, op):
        return self.scalar_type(op.result.ty) == I32

    def normalize(self, value, ty):
        ty = self.scalar_type(ty)

        if ty == F32:
            return float(value)
        if ty == I32:
            return int(value)
        if ty == BOOL:
            return bool(value)

        raise TypeError(f"Cannot create constant of type {ty}")

    def extremum(self, opcode, lhs, rhs, ty):
        if self.scalar_type(ty) == F32:
            lhs = float(lhs)
            rhs = float(rhs)

            if math.isnan(lhs):
                return lhs
            if math.isnan(rhs):
                return rhs

        if opcode == "maximum":
            return lhs if lhs > rhs else rhs

        return lhs if lhs < rhs else rhs

    def fold(self, op):
        args = op.operands

        if op.opcode == "select":
            condition, true_value, false_value = args

            if isinstance(condition, Const):
                return true_value if condition.value else false_value

            if operand_key(true_value) == operand_key(false_value):
                return true_value

            return None

        # Algebraic identities.
        if len(args) == 2:
            lhs, rhs = args

            if (
                op.opcode in ("add", "sub")
                and self.is_integer_result(op)
                and isinstance(rhs, Const)
                and rhs.value == 0
            ):
                return lhs

            if (
                op.opcode in ("mul", "div")
                and isinstance(rhs, Const)
                and rhs.value == 1
            ):
                return lhs

        if not all(isinstance(x, Const) for x in args):
            return None

        values = [x.value for x in args]

        try:
            if op.opcode == "add":
                result = values[0] + values[1]
            elif op.opcode == "sub":
                result = values[0] - values[1]
            elif op.opcode == "mul":
                result = values[0] * values[1]
            elif op.opcode == "div":
                if self.scalar_type(op.result.ty) == I32:
                    result = int(values[0] / values[1])
                else:
                    result = values[0] / values[1]
            elif op.opcode == "cmp_lt":
                result = values[0] < values[1]
            elif op.opcode == "neg":
                result = -values[0]
            elif op.opcode == "exp":
                result = math.exp(values[0])
            elif op.opcode == "maximum" or op.opcode == "minimum":
                result = self.extremum(op.opcode, values[0], values[1], op.result.ty)
            else:
                return None
        except (ArithmeticError, OverflowError):
            return None

        return Const(self.normalize(result, op.result.ty))

    def run(self, ops):
        replacements: dict[int, SSAOperand] = {}
        output = []

        for op in ops:
            operands = tuple(resolve_operand(x, replacements) for x in op.operands)

            rewritten = SSAOp(
                opcode=op.opcode,
                operands=operands,
                result=op.result,
                attrs=dict(op.attrs),
            )

            folded = self.fold(rewritten) if rewritten.result is not None else None

            if folded is not None:
                result = rewritten.result
                assert result is not None
                replacements[result.id] = folded
            else:
                output.append(rewritten)

        return output


class CSEPass:
    PURE_OPS: ClassVar[set[str]] = {
        "program_id",
        "arange",
        "add",
        "sub",
        "mul",
        "div",
        "cmp_lt",
        "neg",
        "exp",
        "maximum",
        "minimum",
        "addptr",
        "select",
        "and",
        "expand_dims",
    }

    def operand_key(self, operand):
        return operand_key(operand)

    def run(self, ops):
        replacements: dict[int, SSAOperand] = {}
        available: dict[CSEKey, SSAValue] = {}
        output = []

        for op in ops:
            operands = tuple(resolve_operand(x, replacements) for x in op.operands)

            rewritten = SSAOp(
                opcode=op.opcode,
                operands=operands,
                result=op.result,
                attrs=dict(op.attrs),
            )

            result = rewritten.result

            if op.opcode not in self.PURE_OPS or result is None:
                output.append(rewritten)
                continue

            key = (
                op.opcode,
                tuple(self.operand_key(x) for x in operands),
                tuple(sorted(op.attrs.items())),
                result.ty,
            )

            if key in available:
                replacements[result.id] = available[key]
            else:
                available[key] = result
                output.append(rewritten)

        return output


class DCEPass:
    SIDE_EFFECTS: ClassVar[set[str]] = {"store"}

    def run(self, ops):
        live = set()
        output = []

        for op in reversed(ops):
            needed = op.opcode in self.SIDE_EFFECTS or (
                op.result is not None and op.result.id in live
            )

            if not needed:
                continue

            if op.result is not None:
                live.discard(op.result.id)

            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    live.add(operand.id)

            output.append(op)

        return list(reversed(output))


class PassManager:
    def __init__(self, passes, verifier):
        self.passes = passes
        self.verifier = verifier

    def run(self, ops):
        if any(isinstance(op, SSAForRange) for op in ops):
            self.verifier.verify(ops)
            return ops

        for compiler_pass in self.passes:
            ops = compiler_pass.run(ops)
            self.verifier.verify(ops)

        return ops
