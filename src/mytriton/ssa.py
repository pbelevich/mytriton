from dataclasses import dataclass, field
from typing import ClassVar

from .trace import (
    AddPtr,
    Arange,
    Barrier,
    BinOp,
    Const,
    Dot,
    ExpandDims,
    Load,
    Max,
    Maximum,
    Min,
    Minimum,
    Param,
    ProgramId,
    SharedAlloc,
    Store,
    Sum,
    Type,
    UnaryOp,
    Where,
)
from .type_inference import TypeInference


@dataclass(frozen=True)
class SSAValue:
    id: int
    ty: Type

    def __str__(self):
        return f"%{self.id}"


SSAOperand = SSAValue | Param | Const | None


@dataclass
class SSAOp:
    opcode: str
    operands: tuple[SSAOperand, ...] = ()
    result: SSAValue | None = None
    attrs: dict[str, object] = field(default_factory=dict)


class SSALowering:
    BINOPS: ClassVar[dict[str, str]] = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "<": "cmp_lt",
        "&": "and",
    }

    def __init__(self):
        self.ops = []
        self.memo = {}
        self.next_id = 0
        self.type_inference = TypeInference()

    def new_result(self, expr):
        result = SSAValue(
            id=self.next_id,
            ty=self.type_inference.infer(expr),
        )
        self.next_id += 1
        return result

    def emit(self, opcode, expr, operands=(), attrs=None):
        result = self.new_result(expr)

        self.ops.append(
            SSAOp(
                opcode=opcode,
                operands=tuple(operands),
                result=result,
                attrs=attrs or {},
            )
        )

        self.memo[id(expr)] = result
        return result

    def lower_expr(self, expr):
        if isinstance(expr, (Const, Param)):
            return expr

        if id(expr) in self.memo:
            return self.memo[id(expr)]

        if isinstance(expr, ProgramId):
            return self.emit(
                "program_id",
                expr,
                attrs={"axis": expr.axis},
            )

        if isinstance(expr, Arange):
            return self.emit(
                "arange",
                expr,
                attrs={
                    "start": expr.start,
                    "end": expr.end,
                },
            )

        if isinstance(expr, BinOp):
            if expr.op not in self.BINOPS:
                raise TypeError(f"Unsupported binary operator: {expr.op}")
            lhs = self.lower_expr(expr.lhs)
            rhs = self.lower_expr(expr.rhs)

            return self.emit(
                self.BINOPS[expr.op],
                expr,
                operands=(lhs, rhs),
            )

        if isinstance(expr, Load):
            ptr = self.lower_expr(expr.ptr)
            mask = self.lower_expr(expr.mask) if expr.mask is not None else None
            other = self.lower_expr(expr.other) if expr.other is not None else None

            return self.emit(
                "load",
                expr,
                operands=(ptr, mask, other),
            )

        if isinstance(expr, SharedAlloc):
            return self.emit(
                "shared_alloc",
                expr,
                attrs={
                    "shape": expr.shape,
                    "dtype": expr.dtype,
                },
            )

        if isinstance(expr, AddPtr):
            base = self.lower_expr(expr.base)
            offset = self.lower_expr(expr.offset)

            return self.emit(
                "addptr",
                expr,
                operands=(base, offset),
            )

        if isinstance(expr, (Maximum, Minimum)):
            lhs = self.lower_expr(expr.lhs)
            rhs = self.lower_expr(expr.rhs)

            return self.emit(
                "maximum" if isinstance(expr, Maximum) else "minimum",
                expr,
                operands=(lhs, rhs),
            )

        if isinstance(expr, UnaryOp):
            value = self.lower_expr(expr.value)

            return self.emit(
                expr.op,
                expr,
                operands=(value,),
            )

        if isinstance(expr, Where):
            condition = self.lower_expr(expr.condition)
            true_value = self.lower_expr(expr.true_value)
            false_value = self.lower_expr(expr.false_value)
            return self.emit(
                "select",
                expr,
                operands=(
                    condition,
                    true_value,
                    false_value,
                ),
            )

        if isinstance(expr, Sum):
            value = self.lower_expr(expr.value)
            return self.emit(
                "sum",
                expr,
                operands=(value,),
            )

        if isinstance(expr, Max):
            value = self.lower_expr(expr.value)
            return self.emit(
                "max",
                expr,
                operands=(value,),
            )

        if isinstance(expr, Min):
            value = self.lower_expr(expr.value)
            return self.emit(
                "min",
                expr,
                operands=(value,),
            )

        if isinstance(expr, ExpandDims):
            value = self.lower_expr(expr.value)
            return self.emit(
                "expand_dims",
                expr,
                operands=(value,),
                attrs={"axis": expr.axis},
            )

        if isinstance(expr, Dot):
            lhs = self.lower_expr(expr.lhs)
            rhs = self.lower_expr(expr.rhs)

            return self.emit(
                "dot",
                expr,
                operands=(lhs, rhs),
            )

        raise TypeError(f"Cannot lower expression: {expr}")

    def lower(self, top_level_ops):
        for op in top_level_ops:
            if isinstance(op, Store):
                self.type_inference.check_store(op)

                value = self.lower_expr(op.value)
                ptr = self.lower_expr(op.ptr)
                mask = self.lower_expr(op.mask) if op.mask is not None else None

                self.ops.append(
                    SSAOp(
                        opcode="store",
                        operands=(ptr, value, mask),
                    )
                )
                continue

            if isinstance(op, Load):
                self.lower_expr(op)
                continue

            if isinstance(op, Barrier):
                self.ops.append(SSAOp(opcode="barrier"))
                continue

            raise TypeError(f"Cannot lower operation: {op}")

        return self.ops


class SSAPrinter:
    def operand(self, value):
        if value is None:
            return "none"

        if isinstance(value, SSAValue):
            return str(value)

        if isinstance(value, Param):
            return value.name

        if isinstance(value, Const):
            return repr(value.value)

        raise TypeError(f"Unknown SSA operand: {value}")

    def print_ops(self, ops):
        lines = []

        for op in ops:
            operands = ", ".join(self.operand(x) for x in op.operands)

            attrs = ", ".join(f"{key}={value}" for key, value in op.attrs.items())

            suffix = f" {{{attrs}}}" if attrs else ""
            operation = op.opcode
            if operands:
                operation += f" {operands}"
            operation += suffix

            if op.result is None:
                lines.append(operation)
            else:
                lines.append(f"{op.result} = {operation} : {op.result.ty}")

        return "\n".join(lines)
