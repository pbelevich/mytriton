from __future__ import annotations

from dataclasses import dataclass

from .ssa import SSAOp, SSAOperand, SSAValue
from .trace import BOOL, F32, I32, Const, Param, PointerType, Type, VectorType


@dataclass(frozen=True)
class MLIRPtrRef:
    base: str
    index: str | None
    memref_ty: str


class SSAMLIRCodegen:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.values: dict[int, str | MLIRPtrRef] = {}
        self.constants: dict[tuple[object, str], str] = {}

    def scalar_type(self, ty: Type) -> Type:
        return ty.element if isinstance(ty, VectorType) else ty

    def mlir_scalar_type(self, ty: Type) -> str:
        ty = self.scalar_type(ty)
        if ty == I32:
            return "i32"
        if ty == F32:
            return "f32"
        if ty == BOOL:
            return "i1"
        raise TypeError(f"unsupported scalar MLIR type: {ty}")

    def mlir_param_type(self, ty: Type) -> str:
        if isinstance(ty, PointerType):
            if ty.element == F32:
                return "memref<?xf32>"
            raise TypeError(f"unsupported pointer element type: {ty.element}")
        return self.mlir_scalar_type(ty)

    def const_literal(self, value: object, mlir_ty: str) -> str:
        if mlir_ty == "i1":
            if not isinstance(value, bool):
                raise TypeError(f"expected bool constant for i1, got {value!r}")
            return "true" if value else "false"
        if mlir_ty == "i32":
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"expected int constant for i32, got {value!r}")
            return str(int(value))
        if mlir_ty == "f32":
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"expected numeric constant for f32, got {value!r}")
            return f"{float(value):.6e}"
        raise TypeError(f"unsupported constant type: {mlir_ty}")

    def constant(self, value: object, mlir_ty: str) -> str:
        key = (value, mlir_ty)
        if key in self.constants:
            return self.constants[key]

        safe_value = str(value).replace("-", "neg_").replace(".", "_")
        name = f"%c_{mlir_ty}_{safe_value}"
        literal = self.const_literal(value, mlir_ty)

        self.lines.append(f"    {name} = arith.constant {literal} : {mlir_ty}")
        self.constants[key] = name
        return name

    def operand_type(self, operand: SSAOperand) -> str:
        if isinstance(operand, SSAValue):
            return self.mlir_scalar_type(operand.ty)
        if isinstance(operand, Param):
            return self.mlir_param_type(operand.ty)
        if isinstance(operand, Const):
            if isinstance(operand.value, bool):
                return "i1"
            if isinstance(operand.value, int):
                return "i32"
            if isinstance(operand.value, float):
                return "f32"
        raise TypeError(f"cannot infer MLIR operand type: {operand}")

    def operand(self, operand: SSAOperand, expected_ty: str | None = None):
        if isinstance(operand, SSAValue):
            return self.values[operand.id]

        if isinstance(operand, Param):
            if isinstance(operand.ty, PointerType):
                return MLIRPtrRef(
                    base=f"%{operand.name}",
                    index=None,
                    memref_ty=self.mlir_param_type(operand.ty),
                )
            return f"%{operand.name}"

        if isinstance(operand, Const):
            return self.constant(
                operand.value, expected_ty or self.operand_type(operand)
            )

        raise TypeError(f"unsupported MLIR operand: {operand}")

    def emit_binary(self, op: SSAOp) -> None:
        result = op.result
        assert result is not None

        lhs_ty = self.operand_type(op.operands[0])
        result_ty = self.mlir_scalar_type(result.ty)

        lhs = self.operand(op.operands[0], lhs_ty)
        rhs = self.operand(op.operands[1], lhs_ty)

        if op.opcode == "cmp_lt":
            opcode = "arith.cmpi slt" if lhs_ty == "i32" else "arith.cmpf olt"
            self.lines.append(f"    %v{result.id} = {opcode}, {lhs}, {rhs} : {lhs_ty}")
        else:
            opcode = {
                ("add", "i32"): "arith.addi",
                ("sub", "i32"): "arith.subi",
                ("mul", "i32"): "arith.muli",
                ("div", "i32"): "arith.divsi",
                ("add", "f32"): "arith.addf",
                ("sub", "f32"): "arith.subf",
                ("mul", "f32"): "arith.mulf",
                ("div", "f32"): "arith.divf",
            }[(op.opcode, result_ty)]
            self.lines.append(
                f"    %v{result.id} = {opcode} {lhs}, {rhs} : {result_ty}"
            )

        self.values[result.id] = f"%v{result.id}"

    def emit_addptr(self, op: SSAOp) -> None:
        result = op.result
        assert result is not None

        base = self.operand(op.operands[0])
        offset = self.operand(op.operands[1], "i32")

        if not isinstance(base, MLIRPtrRef):
            raise TypeError(f"addptr expected pointer base, got {base}")

        index = f"%idx{result.id}"
        self.lines.append(f"    {index} = arith.index_cast {offset} : i32 to index")
        self.values[result.id] = MLIRPtrRef(
            base=base.base, index=index, memref_ty=base.memref_ty
        )

    def emit_load(self, op: SSAOp) -> None:
        result = op.result
        assert result is not None

        ptr = self.operand(op.operands[0])
        if not isinstance(ptr, MLIRPtrRef) or ptr.index is None:
            raise TypeError(f"load expected indexed pointer, got {ptr}")

        result_ty = self.mlir_scalar_type(result.ty)
        mask = None if op.operands[1] is None else self.operand(op.operands[1], "i1")
        other = (
            self.constant(0.0, result_ty)
            if op.operands[2] is None
            else self.operand(op.operands[2], result_ty)
        )

        if mask is None:
            self.lines.append(
                f"    %v{result.id} = memref.load {ptr.base}[{ptr.index}] : {ptr.memref_ty}"
            )
        else:
            self.lines.extend(
                [
                    f"    %v{result.id} = scf.if {mask} -> ({result_ty}) {{",
                    f"      %loaded{result.id} = memref.load {ptr.base}[{ptr.index}] : {ptr.memref_ty}",
                    f"      scf.yield %loaded{result.id} : {result_ty}",
                    "    } else {",
                    f"      scf.yield {other} : {result_ty}",
                    "    }",
                ]
            )

        self.values[result.id] = f"%v{result.id}"

    def emit_store(self, op: SSAOp) -> None:
        ptr = self.operand(op.operands[0])
        value = self.operand(op.operands[1])
        mask = None if op.operands[2] is None else self.operand(op.operands[2], "i1")

        if not isinstance(ptr, MLIRPtrRef) or ptr.index is None:
            raise TypeError(f"store expected indexed pointer, got {ptr}")

        if mask is None:
            self.lines.append(
                f"    memref.store {value}, {ptr.base}[{ptr.index}] : {ptr.memref_ty}"
            )
        else:
            self.lines.extend(
                [
                    f"    scf.if {mask} {{",
                    f"      memref.store {value}, {ptr.base}[{ptr.index}] : {ptr.memref_ty}",
                    "    }",
                ]
            )

    def emit(self, op: SSAOp) -> None:
        result = op.result

        if op.opcode == "program_id":
            assert result is not None
            self.values[result.id] = "%block_id_x"
        elif op.opcode == "arange":
            assert result is not None
            self.values[result.id] = "%thread_id_x"
        elif op.opcode in {"add", "sub", "mul", "div", "cmp_lt"}:
            self.emit_binary(op)
        elif op.opcode == "addptr":
            self.emit_addptr(op)
        elif op.opcode == "load":
            self.emit_load(op)
        elif op.opcode == "store":
            self.emit_store(op)
        else:
            raise TypeError(f"MLIR lowering does not support {op.opcode!r} yet")

    def generate(
        self, kernel_name: str, ssa_ops: list[SSAOp], params: list[Param]
    ) -> str:
        self.lines = []
        self.values = {}
        self.constants = {}

        args = [f"%{param.name}: {self.mlir_param_type(param.ty)}" for param in params]
        args.extend(["%block_id_x: i32", "%thread_id_x: i32"])

        for op in ssa_ops:
            self.emit(op)

        body = ["module {", f"  func.func @{kernel_name}({', '.join(args)}) {{"]
        body.extend(self.lines)
        body.append("    return")
        body.append("  }")
        body.append("}")
        return "\n".join(body)
