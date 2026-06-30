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
    def __init__(self, indent: str = "    ") -> None:
        self.indent = indent
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

        self.lines.append(f"{self.indent}{name} = arith.constant {literal} : {mlir_ty}")
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
            self.lines.append(
                f"{self.indent}%v{result.id} = {opcode}, {lhs}, {rhs} : {lhs_ty}"
            )
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
                f"{self.indent}%v{result.id} = {opcode} {lhs}, {rhs} : {result_ty}"
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
        self.lines.append(
            f"{self.indent}{index} = arith.index_cast {offset} : i32 to index"
        )
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
                f"{self.indent}%v{result.id} = memref.load {ptr.base}[{ptr.index}] : {ptr.memref_ty}"
            )
        else:
            self.lines.extend(
                [
                    f"{self.indent}%v{result.id} = scf.if {mask} -> ({result_ty}) {{",
                    f"{self.indent}  %loaded{result.id} = memref.load {ptr.base}[{ptr.index}] : {ptr.memref_ty}",
                    f"{self.indent}  scf.yield %loaded{result.id} : {result_ty}",
                    f"{self.indent}}} else {{",
                    f"{self.indent}  scf.yield {other} : {result_ty}",
                    f"{self.indent}}}",
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
                f"{self.indent}memref.store {value}, {ptr.base}[{ptr.index}] : {ptr.memref_ty}"
            )
        else:
            self.lines.extend(
                [
                    f"{self.indent}scf.if {mask} {{",
                    f"{self.indent}  memref.store {value}, {ptr.base}[{ptr.index}] : {ptr.memref_ty}",
                    f"{self.indent}}}",
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
        self,
        kernel_name: str,
        ssa_ops: list[SSAOp],
        params: list[Param],
        *,
        host_name: str | None = None,
        grid_x: int | None = None,
        block_x: int | None = None,
    ) -> str:
        del host_name, grid_x, block_x
        self.lines = []
        self.values = {}
        self.constants = {}

        args = [f"%{param.name}: {self.mlir_param_type(param.ty)}" for param in params]
        args.extend(["%block_id_x: i32", "%thread_id_x: i32"])

        for op in ssa_ops:
            self.emit(op)

        body = ["module {", f"  func.func @{kernel_name}({', '.join(args)}) {{"]
        body.extend(self.lines)
        body.append(f"{self.indent}return")
        body.append("  }")
        body.append("}")
        return "\n".join(body)


class SSAGPUMLIRCodegen(SSAMLIRCodegen):
    def __init__(self) -> None:
        super().__init__(indent="      ")

    def generate(
        self,
        kernel_name: str,
        ssa_ops: list[SSAOp],
        params: list[Param],
        *,
        host_name: str | None = None,
        grid_x: int | None = None,
        block_x: int | None = None,
    ) -> str:
        del host_name, grid_x, block_x
        self.lines = []
        self.values = {}
        self.constants = {}

        args = [f"%{param.name}: {self.mlir_param_type(param.ty)}" for param in params]

        self.lines.extend(
            [
                f"{self.indent}%bid_x = gpu.block_id x",
                f"{self.indent}%tid_x = gpu.thread_id x",
                f"{self.indent}%block_id_x = arith.index_cast %bid_x : index to i32",
                f"{self.indent}%thread_id_x = arith.index_cast %tid_x : index to i32",
            ]
        )

        for op in ssa_ops:
            self.emit(op)

        body = [
            "module attributes {gpu.container_module} {",
            "  gpu.module @kernels {",
            f"    gpu.func @{kernel_name}({', '.join(args)}) kernel {{",
        ]
        body.extend(self.lines)
        body.append("      gpu.return")
        body.append("    }")
        body.append("  }")
        body.append("}")

        return "\n".join(body)


class SSAGPUExecutableMLIRCodegen(SSAGPUMLIRCodegen):
    def __init__(self) -> None:
        super().__init__()

    def index_const(self, name: str, value: int, indent: str) -> str:
        return f"{indent}%{name} = arith.constant {value} : index"

    def i32_const(self, name: str, value: int, indent: str) -> str:
        return f"{indent}%{name} = arith.constant {value} : i32"

    def generate(
        self,
        kernel_name: str,
        ssa_ops: list[SSAOp],
        params: list[Param],
        *,
        host_name: str | None = None,
        grid_x: int | None = None,
        block_x: int | None = None,
    ) -> str:
        if grid_x is None or block_x is None:
            raise ValueError(
                "executable GPU MLIR generation requires grid_x and block_x"
            )

        host_name = host_name or f"launch_{kernel_name}"

        self.lines = []
        self.values = {}
        self.constants = {}

        device_args = [
            f"%{param.name}: {self.mlir_param_type(param.ty)}" for param in params
        ]

        host_args = [
            f"%{param.name}: {self.mlir_param_type(param.ty)}" for param in params
        ]

        # Device prelude.
        self.lines.extend(
            [
                f"{self.indent}%bid_x = gpu.block_id x",
                f"{self.indent}%tid_x = gpu.thread_id x",
                f"{self.indent}%block_id_x = arith.index_cast %bid_x : index to i32",
                f"{self.indent}%thread_id_x = arith.index_cast %tid_x : index to i32",
            ]
        )

        for op in ssa_ops:
            self.emit(op)

        body: list[str] = [
            "module attributes {gpu.container_module} {",
            f"  func.func @{host_name}({', '.join(host_args)}) {{",
            "    %c1 = arith.constant 1 : index",
            f"    %grid_x = arith.constant {grid_x} : index",
            f"    %block_x = arith.constant {block_x} : index",
            "    %dynamic_smem = arith.constant 0 : i32",
            f"    gpu.launch_func @kernels::@{kernel_name}",
            "        blocks in (%grid_x, %c1, %c1)",
            "        threads in (%block_x, %c1, %c1)",
            "        dynamic_shared_memory_size %dynamic_smem",
            f"        args({self.host_launch_args(params)})",
            "    return",
            "  }",
            "",
            "  gpu.module @kernels {",
            f"    gpu.func @{kernel_name}({', '.join(device_args)}) kernel {{",
        ]

        body.extend(self.lines)

        body.extend(
            [
                "      gpu.return",
                "    }",
                "  }",
                "}",
            ]
        )

        return "\n".join(body)

    def host_launch_args(self, params: list[Param]) -> str:
        parts = []
        for param in params:
            parts.append(f"%{param.name} : {self.mlir_param_type(param.ty)}")
        return ", ".join(parts)
