from dataclasses import dataclass, field
from typing import ClassVar

from .trace import (
    I32,
    AddPtr,
    Arange,
    BinOp,
    Const,
    ExpandDims,
    ForRange,
    Load,
    LoopCarry,
    LoopIndex,
    LoopResult,
    Max,
    Maximum,
    Min,
    Minimum,
    Param,
    ProgramId,
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


@dataclass
class SSAForRange:
    index: SSAValue
    start: SSAOperand
    stop: SSAOperand
    step: SSAOperand
    carried_inputs: tuple[SSAOperand, ...]
    carried_args: tuple[SSAValue, ...]
    body: list["SSAItem"]
    yields: tuple[SSAOperand, ...]
    results: tuple[SSAValue, ...]


SSAItem = SSAOp | SSAForRange


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

    def new_value(self, ty: Type) -> SSAValue:
        result = SSAValue(
            id=self.next_id,
            ty=ty,
        )
        self.next_id += 1
        return result

    def new_result(self, expr):
        return self.new_value(self.type_inference.infer(expr))

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

        if isinstance(expr, (LoopIndex, LoopCarry)):
            if id(expr) not in self.memo:
                raise TypeError(f"loop value used outside of loop body: {expr}")
            return self.memo[id(expr)]

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

        if isinstance(expr, LoopResult):
            if id(expr) not in self.memo:
                raise TypeError("loop result used before loop was lowered")
            return self.memo[id(expr)]

        raise TypeError(f"Cannot lower expression: {expr}")

    def lower_for_range(self, loop: ForRange) -> None:
        old_ops = self.ops
        old_memo = self.memo.copy()
        old_next_id = self.next_id
        old_inferred_types = self.type_inference.types.copy()
        old_memo_values: dict[int, SSAOperand] = {}

        def set_loop_memo(expr, value: SSAOperand) -> None:
            key = id(expr)
            if key in self.memo:
                old_memo_values[key] = self.memo[key]
            self.memo[key] = value

        try:
            start = self.lower_expr(loop.start)
            stop = self.lower_expr(loop.stop)
            step = self.lower_expr(loop.step)
            for capture in loop.captures:
                self.lower_expr(capture)
            carried_inputs = tuple(
                self.lower_expr(value) for value in loop.carried_inputs
            )

            index_value = self.new_value(I32)
            carried_args = tuple(
                self.new_value(self.type_inference.infer(arg.initial))
                for arg in loop.carried_args
            )

            set_loop_memo(loop.index, index_value)
            for carried_arg, ssa_arg in zip(
                loop.carried_args,
                carried_args,
                strict=True,
            ):
                set_loop_memo(carried_arg, ssa_arg)

            self.ops = []
            for body_op in loop.body:
                self._lower_top_level_op(body_op)

            yields = tuple(self.lower_expr(value) for value in loop.carried_outputs)
            body_ops = self.ops

            results = tuple(
                self.new_value(self.type_inference.infer(result))
                for result in loop.results
            )

            for loop_result, ssa_result in zip(loop.results, results, strict=True):
                self.memo[id(loop_result)] = ssa_result

            self.ops = old_ops

            # Restore memo entries for loop-local placeholders.
            for expr in (loop.index, *loop.carried_args):
                key = id(expr)
                if key in old_memo_values:
                    self.memo[key] = old_memo_values[key]
                else:
                    del self.memo[key]

            self.ops.append(
                SSAForRange(
                    index=index_value,
                    start=start,
                    stop=stop,
                    step=step,
                    carried_inputs=carried_inputs,
                    carried_args=carried_args,
                    body=body_ops,
                    yields=yields,
                    results=results,
                )
            )
        except BaseException:
            self.ops = old_ops
            self.memo.clear()
            self.memo.update(old_memo)
            self.next_id = old_next_id
            self.type_inference.types.clear()
            self.type_inference.types.update(old_inferred_types)
            raise

    def _lower_top_level_op(self, op) -> None:
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
            return

        if isinstance(op, ForRange):
            self.lower_for_range(op)
            return

        raise TypeError(f"Cannot lower operation: {op}")

    def lower(self, top_level_ops):
        for op in top_level_ops:
            self._lower_top_level_op(op)

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

    def print_op(self, op: SSAOp) -> str:
        operands = ", ".join(self.operand(x) for x in op.operands)
        attrs = ", ".join(f"{key}={value}" for key, value in op.attrs.items())

        suffix = f" {{{attrs}}}" if attrs else ""
        operation = op.opcode
        if operands:
            operation += f" {operands}"
        operation += suffix

        if op.result is None:
            return operation
        return f"{op.result} = {operation} : {op.result.ty}"

    def print_for_range(self, loop: SSAForRange) -> list[str]:
        results = ", ".join(str(result) for result in loop.results)
        result_prefix = f"{results} = " if results else ""
        carried = ", ".join(
            f"{arg} = {self.operand(initial)}"
            for arg, initial in zip(
                loop.carried_args,
                loop.carried_inputs,
                strict=True,
            )
        )
        iter_args = f" iter_args({carried})" if carried else ""
        result_types = ", ".join(str(result.ty) for result in loop.results)
        type_suffix = f" : {result_types}" if result_types else ""
        lines = [
            f"{result_prefix}for {loop.index} in range("
            f"{self.operand(loop.start)}, {self.operand(loop.stop)}, "
            f"{self.operand(loop.step)}){iter_args}{type_suffix} {{"
        ]

        for body_op in loop.body:
            body_lines = (
                self.print_for_range(body_op)
                if isinstance(body_op, SSAForRange)
                else [self.print_op(body_op)]
            )
            lines.extend(f"  {line}" for line in body_lines)

        yields = ", ".join(self.operand(value) for value in loop.yields)
        lines.append(f"  yield {yields}" if yields else "  yield")
        lines.append("}")
        return lines

    def print_ops(self, ops: list[SSAItem]) -> str:
        lines = []

        for op in ops:
            if isinstance(op, SSAForRange):
                lines.extend(self.print_for_range(op))
            else:
                lines.append(self.print_op(op))

        return "\n".join(lines)
