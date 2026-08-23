import math
from dataclasses import dataclass
from typing import ClassVar

from .block_shapes import CudaKernelLayout, cuda_kernel_layout
from .cuda_dot_staging import (
    CudaDotSharedBuffers,
    CudaDotStagingAnalysis,
    CudaDotStagingAnalyzer,
    CudaDotStagingPlan,
    CudaGlobalTile,
    CudaGlobalTilePlan,
    CudaSharedBuffer,
    SSADefinitions,
    cuda_scalar_nbytes,
)
from .ssa import SSAForRange, SSAItem, SSAOp, SSAOperand, SSAValue
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


@dataclass(frozen=True)
class CudaPtrRef:
    base: str
    index: str


@dataclass(frozen=True)
class CudaArangeRef:
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


class SSACUDACodegen:
    MAX_SHARED_MEMORY_BYTES: ClassVar[int] = 48 * 1024

    BINARY_OPS: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "cmp_lt": "<",
        "and": "&&",
    }

    def __init__(self):
        self.lines: list[str] = []
        self.values: dict[int, str | CudaPtrRef | CudaArangeRef] = {}
        self.layout = CudaKernelLayout(
            output_tile_shape=(1,),
            thread_shape=(1,),
        )
        self.shared_lines: list[str] = []
        self.shared_memory_bytes = 0
        self.definitions = SSADefinitions([])
        self.staging_analysis = CudaDotStagingAnalysis(
            dot_plans={},
            staging_only_ids=frozenset(),
        )

    def cuda_type(self, ty: Type) -> str:
        if isinstance(ty, BlockType):
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

    def operand(self, operand: SSAOperand) -> str | CudaPtrRef | CudaArangeRef | None:
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
        if (
            isinstance(value, CudaArangeRef)
            and self.is_rank2_kernel()
            and value.width == self.threads_in_kernel_block()
        ):
            return (
                "threadIdx.x" if value.start == 0 else f"({value.start} + threadIdx.x)"
            )

        if not isinstance(value, str):
            raise TypeError(f"Expected CUDA scalar expression, got {value}")
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

    def declare(self, result: SSAValue) -> None:
        name = f"v{result.id}"
        self.lines.append(f"    {self.cuda_type(result.ty)} {name};")
        self.values[result.id] = name

    def reserve_shared_memory(self, additional_bytes: int) -> None:
        required_bytes = self.shared_memory_bytes + additional_bytes
        if required_bytes > self.MAX_SHARED_MEMORY_BYTES:
            raise ValueError(
                f"CUDA shared memory requires {required_bytes} bytes, "
                f"exceeding the conservative {self.MAX_SHARED_MEMORY_BYTES}-byte limit"
            )

        self.shared_memory_bytes = required_bytes

    def append_shared_buffer_declaration(self, buffer: CudaSharedBuffer) -> None:
        cuda_ty = self.cuda_type(buffer.element_ty)
        self.shared_lines.append(
            f"    __shared__ {cuda_ty} {buffer.name}[{buffer.size}];"
        )

    def declare_shared_buffer(
        self,
        name: str,
        logical_shape: tuple[int, ...],
        element_ty: ScalarType,
    ) -> CudaSharedBuffer:
        buffer = CudaSharedBuffer(
            name=name,
            logical_shape=logical_shape,
            element_ty=element_ty,
        )

        self.reserve_shared_memory(buffer.nbytes)
        self.append_shared_buffer_declaration(buffer)

        return buffer

    def emit_cooperative_load(
        self,
        target: CudaSharedBuffer,
        source: CudaGlobalTile,
        *,
        order: tuple[int, ...] = (1, 0),
    ) -> None:
        cooperative_layout = self.layout.cooperative_tile_layout(
            target.logical_shape,
            order=order,
        )
        if cooperative_layout.order != (1, 0):
            raise ValueError(
                "CUDA cooperative dot loads require row-major order (1, 0), "
                f"got {cooperative_layout.order}"
            )

        index = f"{target.name}_index"
        row = f"{target.name}_row"
        column = f"{target.name}_column"
        global_row = f"{target.name}_global_row"
        global_column = f"{target.name}_global_column"
        source_index = f"{target.name}_source_index"
        in_bounds = f"{target.name}_in_bounds"

        self.lines.extend(
            [
                (
                    f"    for (int {index} = threadIdx.x; "
                    f"{index} < {cooperative_layout.size}; "
                    f"{index} += {cooperative_layout.threads_per_block}) {{"
                ),
                f"        int {row} = {index} / {target.columns};",
                f"        int {column} = {index} % {target.columns};",
                (f"        int {global_row} = ({source.row_offset}) + {row};"),
                (f"        int {global_column} = ({source.column_offset}) + {column};"),
                (
                    f"        int {source_index} = "
                    f"{global_row} * ({source.row_stride}) + {global_column};"
                ),
                (
                    f"        bool {in_bounds} = "
                    f"{global_row} < ({source.row_bound}) && "
                    f"{global_column} < ({source.column_bound});"
                ),
                (
                    f"        {target.element(row, column)} = "
                    f"{in_bounds} ? "
                    f"{source.base}[{source_index}] : {source.other};"
                ),
                "    }",
            ]
        )

    def emit_block_barrier(self) -> None:
        self.lines.append("    __syncthreads();")

    def emit_dot_operand_staging(
        self,
        dot_result_id: int,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        element_ty: ScalarType,
        lhs_source: CudaGlobalTile,
        rhs_source: CudaGlobalTile,
    ) -> CudaDotSharedBuffers:
        if len(lhs_shape) != 2 or len(rhs_shape) != 2 or lhs_shape[1] != rhs_shape[0]:
            raise ValueError(
                "dot staging expects compatible rank-2 operands, "
                f"got {lhs_shape} and {rhs_shape}"
            )

        lhs = CudaSharedBuffer(
            name=f"dot_lhs_{dot_result_id}",
            logical_shape=lhs_shape,
            element_ty=element_ty,
        )
        rhs = CudaSharedBuffer(
            name=f"dot_rhs_{dot_result_id}",
            logical_shape=rhs_shape,
            element_ty=element_ty,
        )

        # Reserve both operands before mutating the generated CUDA fragment.
        self.reserve_shared_memory(lhs.nbytes + rhs.nbytes)
        self.append_shared_buffer_declaration(lhs)
        self.append_shared_buffer_declaration(rhs)

        self.emit_cooperative_load(lhs, lhs_source)
        self.emit_cooperative_load(rhs, rhs_source)
        self.emit_block_barrier()

        return CudaDotSharedBuffers(
            lhs=lhs,
            rhs=rhs,
        )

    def resolve_global_tile(
        self,
        plan: CudaGlobalTilePlan,
    ) -> CudaGlobalTile:
        base = self.pointer_operand(plan.base)

        if base.index != "0":
            raise TypeError(
                f"dot staging expects an unmodified global base pointer, got {base}"
            )

        return CudaGlobalTile(
            base=base.base,
            row_offset=self.expression_operand(plan.row_offset),
            column_offset=self.expression_operand(plan.column_offset),
            row_stride=self.expression_operand(plan.row_stride),
            row_bound=self.expression_operand(plan.row_bound),
            column_bound=self.expression_operand(plan.column_bound),
            other=self.expression_operand(plan.other),
        )

    def emit_dot_operand_staging_from_ssa(
        self,
        op: SSAOp,
        plan: CudaDotStagingPlan,
    ) -> CudaDotSharedBuffers:
        if op.opcode != "dot":
            raise TypeError(
                f"expected dot operation for shared staging, got {op.opcode}"
            )

        if op.result is None:
            raise TypeError("dot operation requires a result")

        lhs, rhs = op.operands

        if not isinstance(lhs, SSAValue) or not isinstance(
            lhs.ty,
            BlockType,
        ):
            raise TypeError(f"dot lhs must be a block SSA value, got {lhs}")

        if not isinstance(rhs, SSAValue) or not isinstance(
            rhs.ty,
            BlockType,
        ):
            raise TypeError(f"dot rhs must be a block SSA value, got {rhs}")

        element_ty = lhs.ty.element
        if not isinstance(element_ty, ScalarType):
            raise TypeError(
                f"dot shared-memory element must be scalar, got {element_ty}"
            )

        if rhs.ty.element != element_ty:
            raise TypeError(
                "dot shared-memory operands must have matching elements, "
                f"got {lhs.ty.element} and {rhs.ty.element}"
            )

        # Resolve every operand before mutating shared_lines/lines.
        lhs_source = self.resolve_global_tile(plan.lhs)
        rhs_source = self.resolve_global_tile(plan.rhs)

        return self.emit_dot_operand_staging(
            dot_result_id=op.result.id,
            lhs_shape=lhs.ty.shape,
            rhs_shape=rhs.ty.shape,
            element_ty=element_ty,
            lhs_source=lhs_source,
            rhs_source=rhs_source,
        )

    def is_staging_only(self, op: SSAOp) -> bool:
        return (
            op.result is not None
            and op.result.id in self.staging_analysis.staging_only_ids
        )

    def scalar_type(self, ty: Type) -> ScalarType | PointerType:
        return ty.element if isinstance(ty, BlockType) else ty

    def is_rank2_kernel(self) -> bool:
        return self.layout.is_rank2

    def threads_in_kernel_block(self) -> int:
        return self.layout.threads_per_block

    def thread_coordinate(self, thread_axis: int) -> str:
        if self.layout.rank == 1:
            if thread_axis != 0:
                raise ValueError(
                    f"invalid thread axis {thread_axis} for rank-1 CUDA layout"
                )
            return "threadIdx.x"

        coordinates = ("tile_i", "tile_j")
        if thread_axis < 0 or thread_axis >= len(coordinates):
            raise ValueError(
                f"invalid thread axis {thread_axis} for {self.layout.thread_shape}"
            )

        return coordinates[thread_axis]

    def emit_rank2_prologue(self) -> None:
        if not self.is_rank2_kernel():
            return

        _, cols = self.layout.thread_shape

        self.lines.extend(
            [
                f"    int tile_i = threadIdx.x / {cols};",
                f"    int tile_j = threadIdx.x % {cols};",
            ]
        )

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
        if not isinstance(input_ty, BlockType) or input_ty.rank != 1:
            raise TypeError(f"{op.opcode} expects a vector input, got {input_ty}")

        value = self.expression_operand(operand)

        element_ty = input_ty.element
        if not isinstance(element_ty, ScalarType):
            raise TypeError(f"{op.opcode} expects scalar elements, got {element_ty}")

        cuda_ty = self.cuda_type(element_ty)
        width = input_ty.size
        if width & (width - 1):
            raise TypeError(f"reduction width must be a power of two, got {width}")

        shared = f"reduce_smem_{result.id}"
        stride = f"stride_{result.id}"

        self.reserve_shared_memory(width * cuda_scalar_nbytes(element_ty))
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

    def emit_for_range(self, loop: SSAForRange) -> None:
        start = self.expression_operand(loop.start)
        stop = self.expression_operand(loop.stop)
        step = self.expression_operand(loop.step)

        index_name = f"v{loop.index.id}"

        carried_names = []

        for carried_input, carried_arg, result in zip(
            loop.carried_inputs,
            loop.carried_args,
            loop.results,
            strict=True,
        ):
            init = self.expression_operand(carried_input)
            name = f"v{result.id}"
            cuda_ty = self.cuda_type(result.ty)

            self.lines.append(f"    {cuda_ty} {name} = {init};")
            self.values[carried_arg.id] = name
            self.values[result.id] = name
            carried_names.append(name)

        self.lines.append(
            f"    for (int {index_name} = {start}; {index_name} < {stop}; {index_name} += {step}) {{"
        )

        self.values[loop.index.id] = index_name

        body_start = len(self.lines)

        for body_op in loop.body:
            if isinstance(body_op, SSAForRange):
                self.emit_for_range(body_op)
            elif not self.is_staging_only(body_op):
                self.emit(body_op)

        for yielded, carried_name in zip(loop.yields, carried_names, strict=True):
            value = self.expression_operand(yielded)
            self.lines.append(f"    {carried_name} = {value};")

        body_lines = self.lines[body_start:]
        self.lines[body_start:] = [f"    {line}" for line in body_lines]

        self.lines.append("    }")

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
            end = op.attrs["end"]

            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError(f"arange expects integer start/end, got {start}, {end}")

            if self.is_rank2_kernel():
                self.values[result.id] = CudaArangeRef(start=start, end=end)
            else:
                expression = "threadIdx.x" if start == 0 else f"({start} + threadIdx.x)"
                self.assign(result, expression)
        elif op.opcode == "empty":
            self.declare(result)
        elif op.opcode == "full":
            self.assign(result, self.expression_operand(op.operands[0]))
        elif op.opcode == "zeros":
            element_ty = self.scalar_type(result.ty)
            zero = False if element_ty == BOOL else 0.0 if element_ty == F32 else 0
            self.assign(result, self.literal(zero))
        elif op.opcode == "dot":
            if result.id not in self.staging_analysis.stageable_dot_ids:
                raise TypeError("CUDA lowering for tl.dot is not implemented")

            plan = self.staging_analysis.plan_for(result.id)
            self.emit_dot_operand_staging_from_ssa(op, plan)

            raise TypeError(
                "CUDA shared-memory staging for tl.dot is implemented, "
                "but CUDA computation for tl.dot is not implemented"
            )
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
        elif op.opcode == "expand_dims":
            if not self.is_rank2_kernel():
                raise TypeError(
                    "CUDA expand_dims lowering currently requires rank-2 kernel"
                )

            operand = op.operands[0]
            if not isinstance(operand, SSAValue):
                raise TypeError(f"expand_dims expects SSA operand, got {operand}")

            arange_ref = self.operand(operand)
            if not isinstance(arange_ref, CudaArangeRef):
                raise TypeError(
                    "CUDA expand_dims MVP supports only direct arange expansion, "
                    f"got {arange_ref}"
                )

            axis = op.attrs.get("axis")
            if type(axis) is not int:
                raise TypeError(f"expand_dims axis must be an integer, got {axis}")

            assert isinstance(axis, int)

            if not isinstance(result.ty, BlockType):
                raise TypeError(f"expand_dims expects block result, got {result.ty}")

            result_shape = result.ty.shape

            try:
                tile_layout = self.layout.tile_layout(
                    result_shape,
                    broadcast_axes=(axis,),
                )
            except ValueError as error:
                raise TypeError(
                    f"cannot map expand_dims result {result.ty} into CUDA tile "
                    f"shape {self.layout.thread_shape}"
                ) from error

            mapped_axes = [
                thread_axis
                for thread_axis in tile_layout.thread_axes
                if thread_axis is not None
            ]

            if len(mapped_axes) != 1:
                raise TypeError(
                    "expanded arange must map to exactly one CUDA thread axis, "
                    f"got {tile_layout}"
                )

            coord = self.thread_coordinate(mapped_axes[0])

            expression = (
                coord if arange_ref.start == 0 else f"({arange_ref.start} + {coord})"
            )
            self.assign(result, expression)
        else:
            raise TypeError(f"Unsupported SSA opcode: {op.opcode}")

    def generate(
        self,
        kernel_name: str,
        ssa_ops: list[SSAItem],
        params: list[Param],
    ) -> str:
        self.lines = []
        self.shared_lines = []
        self.shared_memory_bytes = 0
        self.values = {}
        self.definitions = SSADefinitions(ssa_ops)
        self.staging_analysis = CudaDotStagingAnalyzer(self.definitions).analyze()
        self.layout = cuda_kernel_layout(ssa_ops)

        self.emit_rank2_prologue()

        signature = ", ".join(
            f"{self.cuda_type(param.ty)} {param.name}" for param in params
        )

        for op in ssa_ops:
            if isinstance(op, SSAForRange):
                self.emit_for_range(op)
            elif not self.is_staging_only(op):
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
