import math
from dataclasses import dataclass
from typing import ClassVar

from .ssa import SSAOp, SSAOperand, SSAValue
from .trace import (
    BOOL,
    F32,
    I32,
    Const,
    Param,
    PointerType,
    ScalarType,
    Type,
    VectorType,
)


@dataclass(frozen=True)
class CudaPtrRef:
    base: str
    index: str


class SSACUDACodegen:
    BINARY_OPS: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "cmp_lt": "<",
    }

    def __init__(self):
        self.lines: list[str] = []
        self.values: dict[int, str | CudaPtrRef] = {}
        self.shared_lines: list[str] = []

    def cuda_type(self, ty: Type) -> str:
        if isinstance(ty, VectorType):
            ty = ty.element

        if ty == I32:
            return "int"
        if ty == F32:
            return "float"
        if ty == BOOL:
            return "bool"
        if isinstance(ty, PointerType):
            return f"{self.cuda_type(ty.element)}*"

        raise TypeError(f"Cannot lower CUDA type: {ty}")

    def literal(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if math.isnan(value):
                return "__int_as_float(0x7fc00000)"

            if math.isinf(value):
                infinity = "__int_as_float(0x7f800000)"

                return infinity if value > 0 else f"(-{infinity})"

            return f"{value}f"
        if isinstance(value, int):
            return str(value)

        raise TypeError(f"Unsupported CUDA literal: {value!r}")

    def operand(self, operand: SSAOperand) -> str | CudaPtrRef | None:
        if operand is None:
            return None
        if isinstance(operand, SSAValue):
            if operand.id not in self.values:
                raise RuntimeError(f"SSA value {operand} is not defined")
            return self.values[operand.id]
        if isinstance(operand, Param):
            return operand.name
        if isinstance(operand, Const):
            return self.literal(operand.value)

        raise TypeError(f"Unknown operand: {operand}")

    def expression_operand(self, operand: SSAOperand) -> str:
        value = self.operand(operand)
        if not isinstance(value, str):
            raise TypeError(f"Expected CUDA expression, got {value}")
        return value

    def pointer_operand(self, operand: SSAOperand) -> CudaPtrRef:
        value = self.operand(operand)
        if isinstance(value, str):
            return CudaPtrRef(value, "0")
        if not isinstance(value, CudaPtrRef):
            raise TypeError(f"Expected CUDA pointer, got {value}")
        return value

    def assign(self, result: SSAValue, expression: str) -> None:
        name = f"v{result.id}"
        cuda_ty = self.cuda_type(result.ty)
        self.lines.append(f"    {cuda_ty} {name} = {expression};")
        self.values[result.id] = name

    def scalar_type(self, ty: Type) -> ScalarType | PointerType:
        return ty.element if isinstance(ty, VectorType) else ty

    def reduction_update(
        self,
        opcode: str,
        element_ty: ScalarType | PointerType,
        lhs: str,
        rhs: str,
    ) -> str:
        if opcode == "sum":
            if element_ty in (F32, I32):
                return f"{lhs} += {rhs};"
            raise TypeError(f"Unsupported type for sum: {element_ty}")

        if opcode == "max":
            if element_ty == F32:
                return f"{lhs} = fmaxf({lhs}, {rhs});"
            if element_ty == I32:
                return f"{lhs} = ({lhs} > {rhs} ? {lhs} : {rhs});"
            raise TypeError(f"Unsupported type for max: {element_ty}")

        if opcode == "min":
            if element_ty == F32:
                return f"{lhs} = fminf({lhs}, {rhs});"
            if element_ty == I32:
                return f"{lhs} = ({lhs} < {rhs} ? {lhs} : {rhs});"
            raise TypeError(f"Unsupported type for min: {element_ty}")

        raise TypeError(f"Unsupported reduction opcode: {opcode}")

    def emit_reduction(self, op: SSAOp) -> None:
        operand = op.operands[0]
        if not isinstance(operand, SSAValue):
            raise TypeError(f"{op.opcode} expects an SSA value, got {operand}")

        result = op.result
        if result is None:
            raise TypeError(f"{op.opcode} requires a result")

        input_ty = operand.ty
        if not isinstance(input_ty, VectorType):
            raise TypeError(f"{op.opcode} expects a vector input, got {input_ty}")

        value = self.expression_operand(operand)

        element_ty = input_ty.element
        cuda_ty = self.cuda_type(element_ty)
        width = input_ty.size
        if width & (width - 1):
            raise TypeError(f"reduction width must be a power of two, got {width}")

        shared = f"reduce_smem_{result.id}"
        stride = f"stride_{result.id}"

        self.shared_lines.append(f"    __shared__ {cuda_ty} {shared}[{width}];")

        self.lines.extend(
            [
                f"    {shared}[threadIdx.x] = {value};",
                "    __syncthreads();",
                f"    for (int {stride} = {width // 2}; {stride} > 0; {stride} >>= 1) {{",
                f"        if (threadIdx.x < {stride}) {{",
            ]
        )

        lhs = f"{shared}[threadIdx.x]"
        rhs = f"{shared}[threadIdx.x + {stride}]"

        self.lines.append(
            f"            {self.reduction_update(op.opcode, element_ty, lhs, rhs)}"
        )

        self.lines.extend(
            [
                "        }",
                "        __syncthreads();",
                "    }",
            ]
        )
        self.assign(result, f"{shared}[0]")

    def emit_dot(self, op: SSAOp) -> None:
        lhs_operand = op.operands[0]
        rhs_operand = op.operands[1]

        if not isinstance(lhs_operand, SSAValue):
            raise TypeError(f"dot lhs expects SSA value, got {lhs_operand}")

        if not isinstance(rhs_operand, SSAValue):
            raise TypeError(f"dot rhs expects SSA value, got {rhs_operand}")

        result = op.result
        if result is None:
            raise TypeError("dot requires a result")

        lhs_ty = lhs_operand.ty
        rhs_ty = rhs_operand.ty

        if not isinstance(lhs_ty, VectorType):
            raise TypeError(f"dot lhs expects vector, got {lhs_ty}")

        if not isinstance(rhs_ty, VectorType):
            raise TypeError(f"dot rhs expects vector, got {rhs_ty}")

        if lhs_ty.size != rhs_ty.size:
            raise TypeError(f"dot size mismatch: {lhs_ty} and {rhs_ty}")

        element_ty = result.ty
        cuda_ty = self.cuda_type(element_ty)
        width = lhs_ty.size

        if width & (width - 1):
            raise TypeError(f"dot width must be a power of two, got {width}")

        lhs = self.expression_operand(lhs_operand)
        rhs = self.expression_operand(rhs_operand)

        product = f"dot_product_{result.id}"
        shared = f"dot_smem_{result.id}"
        stride = f"stride_{result.id}"

        self.shared_lines.append(f"    __shared__ {cuda_ty} {shared}[{width}];")

        self.lines.extend(
            [
                f"    {cuda_ty} {product} = ({lhs} * {rhs});",
                f"    {shared}[threadIdx.x] = {product};",
                "    __syncthreads();",
                f"    for (int {stride} = {width // 2}; "
                f"{stride} > 0; {stride} >>= 1) {{",
                f"        if (threadIdx.x < {stride}) {{",
                f"            {shared}[threadIdx.x] += "
                f"{shared}[threadIdx.x + {stride}];",
                "        }",
                "        __syncthreads();",
                "    }",
            ]
        )

        self.assign(result, f"{shared}[0]")

    def emit(self, op: SSAOp) -> None:
        if op.opcode == "store":
            ptr = self.pointer_operand(op.operands[0])
            value = self.expression_operand(op.operands[1])
            mask_operand = op.operands[2]
            mask = (
                None if mask_operand is None else self.expression_operand(mask_operand)
            )

            if mask is None:
                self.lines.append(f"    {ptr.base}[{ptr.index}] = {value};")
            else:
                self.lines.extend(
                    [
                        f"    if ({mask}) {{",
                        f"        {ptr.base}[{ptr.index}] = {value};",
                        "    }",
                    ]
                )
            return

        result = op.result
        if result is None:
            raise TypeError(f"SSA opcode {op.opcode!r} requires a result")

        if op.opcode == "program_id":
            axis = op.attrs["axis"]

            if axis not in (0, 1, 2):
                raise ValueError(f"Invalid program axis: {axis}")

            component = ("x", "y", "z")[axis]
            self.assign(result, f"blockIdx.{component}")
        elif op.opcode == "arange":
            start = op.attrs["start"]
            expression = "threadIdx.x" if start == 0 else f"({start} + threadIdx.x)"
            self.assign(result, expression)
        elif op.opcode in self.BINARY_OPS:
            lhs = self.expression_operand(op.operands[0])
            rhs = self.expression_operand(op.operands[1])
            symbol = self.BINARY_OPS[op.opcode]
            self.assign(result, f"({lhs} {symbol} {rhs})")
        elif op.opcode == "addptr":
            base = self.operand(op.operands[0])
            offset = self.expression_operand(op.operands[1])
            if isinstance(base, CudaPtrRef):
                if base.index != "0":
                    offset = f"({base.index} + {offset})"
                base = base.base
            if not isinstance(base, str):
                raise TypeError(f"addptr expects pointer base, got {base}")
            self.values[result.id] = CudaPtrRef(base, offset)
        elif op.opcode == "load":
            ptr = self.pointer_operand(op.operands[0])
            mask_operand = op.operands[1]
            other_operand = op.operands[2]
            mask = (
                "true"
                if mask_operand is None
                else self.expression_operand(mask_operand)
            )
            if other_operand is None:
                other = "0.0f" if self.scalar_type(result.ty) == F32 else "0"
            else:
                other = self.expression_operand(other_operand)
            self.assign(
                result,
                f"({mask} ? {ptr.base}[{ptr.index}] : {other})",
            )
        elif op.opcode in ("maximum", "minimum"):
            lhs = self.expression_operand(op.operands[0])
            rhs = self.expression_operand(op.operands[1])
            symbol = ">" if op.opcode == "maximum" else "<"
            comparison = f"(({lhs}) {symbol} ({rhs}) ? ({lhs}) : ({rhs}))"
            if self.scalar_type(result.ty) == F32:
                comparison = (
                    f"(isnan({lhs}) ? ({lhs}) : "
                    f"(isnan({rhs}) ? ({rhs}) : {comparison}))"
                )
            self.assign(result, comparison)
        elif op.opcode == "neg":
            value = self.expression_operand(op.operands[0])
            self.assign(result, f"-({value})")
        elif op.opcode == "exp":
            value = self.expression_operand(op.operands[0])
            if self.scalar_type(result.ty) != F32:
                raise TypeError(f"exp requires f32, got {result.ty}")
            self.assign(result, f"expf({value})")
        elif op.opcode == "select":
            condition = self.expression_operand(op.operands[0])
            true_value = self.expression_operand(op.operands[1])
            false_value = self.expression_operand(op.operands[2])
            self.assign(
                result,
                f"({condition} ? {true_value} : {false_value})",
            )
        elif op.opcode in ("sum", "max", "min"):
            self.emit_reduction(op)
        elif op.opcode == "dot":
            self.emit_dot(op)
        else:
            raise TypeError(f"Unsupported SSA opcode: {op.opcode}")

    def generate(
        self,
        kernel_name: str,
        ssa_ops: list[SSAOp],
        params: list[Param],
    ) -> str:
        self.lines = []
        self.shared_lines = []
        self.values = {}

        signature = ", ".join(
            f"{self.cuda_type(param.ty)} {param.name}" for param in params
        )

        for op in ssa_ops:
            self.emit(op)

        body = [
            'extern "C" __global__',
            f"void {kernel_name}({signature}) {{",
        ]

        body.extend(self.shared_lines)

        if self.shared_lines:
            body.append("")

        body.extend(self.lines)
        body.append("}")

        return "\n".join(body)
