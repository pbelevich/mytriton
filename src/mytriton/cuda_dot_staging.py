from dataclasses import dataclass
from math import gcd

from .ssa import SSAForRange, SSAItem, SSAOp, SSAOperand, SSAValue
from .trace import F32, I32, BlockType, Const, Param, PointerType, ScalarType


def cuda_scalar_nbytes(ty: ScalarType) -> int:
    if ty in (F32, I32):
        return 4
    if ty.name == "bool":
        return 1

    raise TypeError(f"cannot determine CUDA storage size for {ty}")


CUDA_SHARED_MEMORY_BANKS = 32


def cuda_f32_shared_row_padding(
    *,
    columns: int,
    simultaneous_rows: int,
) -> int:
    """Return one padding element when row reads would conflict."""

    if type(columns) is not int or columns <= 0:
        raise ValueError(
            f"shared-memory columns must be a positive integer, got {columns}"
        )

    if (
        type(simultaneous_rows) is not int
        or not 1 <= simultaneous_rows <= CUDA_SHARED_MEMORY_BANKS
    ):
        raise ValueError(
            "simultaneous shared-memory rows must be between 1 and "
            f"{CUDA_SHARED_MEMORY_BANKS}, got {simultaneous_rows}"
        )

    distinct_banks = CUDA_SHARED_MEMORY_BANKS // gcd(
        columns,
        CUDA_SHARED_MEMORY_BANKS,
    )

    return int(simultaneous_rows > distinct_banks)


@dataclass(frozen=True)
class CudaSharedBuffer:
    name: str
    logical_shape: tuple[int, ...]
    element_ty: ScalarType
    row_padding: int = 0
    stage_count: int = 1

    def __post_init__(self) -> None:
        if len(self.logical_shape) != 2 or any(
            type(dim) is not int or dim <= 0 for dim in self.logical_shape
        ):
            raise ValueError(
                "shared buffer must be a positive rank-2 tile, "
                f"got {self.logical_shape}"
            )

        if type(self.row_padding) is not int or self.row_padding < 0:
            raise ValueError(
                "shared buffer row padding must be a non-negative integer, "
                f"got {self.row_padding}"
            )

        if type(self.stage_count) is not int or self.stage_count <= 0:
            raise ValueError(
                "shared buffer stage count must be a positive integer, "
                f"got {self.stage_count}"
            )

    @property
    def rows(self) -> int:
        return self.logical_shape[0]

    @property
    def columns(self) -> int:
        return self.logical_shape[1]

    @property
    def row_stride(self) -> int:
        return self.columns + self.row_padding

    @property
    def stage_size(self) -> int:
        return self.rows * self.row_stride

    @property
    def size(self) -> int:
        return self.stage_count * self.stage_size

    @property
    def nbytes(self) -> int:
        return self.size * cuda_scalar_nbytes(self.element_ty)

    def offset(
        self,
        row: int,
        column: int,
        *,
        stage: int = 0,
    ) -> int:
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise ValueError(
                f"shared buffer coordinate {(row, column)} is outside "
                f"{self.logical_shape}"
            )

        if not 0 <= stage < self.stage_count:
            raise ValueError(
                f"shared buffer stage {stage} is outside "
                f"0 <= stage < {self.stage_count}"
            )

        return stage * self.stage_size + row * self.row_stride + column

    def element(
        self,
        row: str,
        column: str,
        *,
        stage: str | None = None,
    ) -> str:
        element_offset = f"({row}) * {self.row_stride} + ({column})"

        if self.stage_count == 1:
            if stage is not None:
                raise ValueError(
                    f"single-stage shared buffer {self.name} "
                    "does not accept a stage expression"
                )

            return f"{self.name}[{element_offset}]"

        if stage is None:
            raise ValueError(
                f"multi-stage shared buffer {self.name} requires a stage expression"
            )

        return f"{self.name}[(({stage}) * {self.stage_size}) + {element_offset}]"


@dataclass(frozen=True)
class CudaGlobalTile:
    base: str
    row_offset: str
    column_offset: str
    row_stride: str
    row_bound: str
    column_bound: str
    other: str = "0.0f"


@dataclass(frozen=True)
class CudaGlobalTilePlan:
    base: SSAOperand
    row_offset: SSAOperand
    column_offset: SSAOperand
    row_stride: SSAOperand
    row_bound: SSAOperand
    column_bound: SSAOperand
    other: SSAOperand


@dataclass(frozen=True)
class CudaDotStagingPlan:
    lhs: CudaGlobalTilePlan
    rhs: CudaGlobalTilePlan


@dataclass(frozen=True)
class CudaDotSharedBuffers:
    lhs: CudaSharedBuffer
    rhs: CudaSharedBuffer

    def __post_init__(self) -> None:
        if self.lhs.stage_count != self.rhs.stage_count:
            raise ValueError(
                "dot shared buffers require matching stage counts, "
                f"got {self.lhs.stage_count} and {self.rhs.stage_count}"
            )

    @property
    def reduction_size(self) -> int:
        return self.lhs.columns

    @property
    def stage_count(self) -> int:
        return self.lhs.stage_count


class SSADefinitions:
    def __init__(self, ssa_ops: list[SSAItem]) -> None:
        self.ops: dict[int, SSAOp] = {}
        self.ordered_ops: list[SSAOp] = []
        self._collect(ssa_ops)

    def _collect(self, ssa_ops: list[SSAItem]) -> None:
        for item in ssa_ops:
            if isinstance(item, SSAForRange):
                self._collect(item.body)
                continue

            self.ordered_ops.append(item)

            if item.result is None:
                continue

            result_id = item.result.id
            if result_id in self.ops:
                raise ValueError(f"duplicate SSA definition for %{result_id}")

            self.ops[result_id] = item

    def get(self, value: SSAValue) -> SSAOp | None:
        return self.ops.get(value.id)

    def require(
        self,
        operand: SSAOperand,
        opcode: str,
    ) -> SSAOp:
        if not isinstance(operand, SSAValue):
            raise TypeError(f"expected SSA value defined by {opcode}, got {operand}")

        op = self.get(operand)
        if op is None:
            raise TypeError(f"SSA value {operand} has no operation definition")

        if op.opcode != opcode:
            raise TypeError(
                f"expected {operand} to be defined by {opcode}, got {op.opcode}"
            )

        return op

    def dependency_ids(
        self,
        *operands: SSAOperand,
    ) -> set[int]:
        result: set[int] = set()

        def visit(operand: SSAOperand) -> None:
            if not isinstance(operand, SSAValue):
                return

            if operand.id in result:
                return

            op = self.get(operand)
            if op is None:
                return

            result.add(operand.id)

            for dependency in op.operands:
                visit(dependency)

        for operand in operands:
            visit(operand)

        return result


class CudaDotOperandMatcher:
    def __init__(self, definitions: SSADefinitions) -> None:
        self.definitions = definitions

    @staticmethod
    def _is_i32_scalar(operand: SSAOperand) -> bool:
        if isinstance(operand, SSAValue):
            return operand.ty == I32

        if isinstance(operand, Param):
            return operand.ty == I32

        return isinstance(operand, Const) and type(operand.value) is int

    def _split_block_and_scalar(
        self,
        lhs: SSAOperand,
        rhs: SSAOperand,
        opcode: str,
    ) -> tuple[SSAValue, SSAOperand]:
        for block, scalar in ((lhs, rhs), (rhs, lhs)):
            if (
                isinstance(block, SSAValue)
                and isinstance(block.ty, BlockType)
                and block.ty.element == I32
                and self._is_i32_scalar(scalar)
            ):
                return block, scalar

        raise TypeError(
            f"dot staging expects {opcode} of block and scalar, got {lhs} and {rhs}"
        )

    def _match_axis_offset(
        self,
        coordinates: SSAValue,
        axis: int,
    ) -> SSAOperand:
        add = self.definitions.require(coordinates, "add")
        lhs, rhs = add.operands

        for offset, expanded in ((lhs, rhs), (rhs, lhs)):
            if not self._is_i32_scalar(offset):
                continue

            if not isinstance(expanded, SSAValue):
                continue

            expand = self.definitions.get(expanded)
            if (
                expand is None
                or expand.opcode != "expand_dims"
                or expand.attrs.get("axis") != axis
            ):
                continue

            arange = self.definitions.require(
                expand.operands[0],
                "arange",
            )
            if arange.attrs.get("start") != 0:
                continue

            return offset

        raise TypeError(
            "dot staging expects coordinates in the form "
            f"scalar_offset + expand_dims(arange(0, size), axis={axis})"
        )

    def _match_mask(
        self,
        mask: SSAOperand,
        rows: SSAValue,
        columns: SSAValue,
    ) -> tuple[SSAOperand, SSAOperand]:
        conjunction = self.definitions.require(mask, "and")

        row_bound: SSAOperand = None
        column_bound: SSAOperand = None

        for comparison_operand in conjunction.operands:
            comparison = self.definitions.require(
                comparison_operand,
                "cmp_lt",
            )
            coordinate, bound = comparison.operands

            if not self._is_i32_scalar(bound):
                raise TypeError(
                    f"dot staging bounds must be scalar i32 values, got {bound}"
                )

            if coordinate == rows:
                row_bound = bound
            elif coordinate == columns:
                column_bound = bound

        if row_bound is None or column_bound is None:
            raise TypeError(
                "dot staging expects mask (rows < row_bound) & (columns < column_bound)"
            )

        return row_bound, column_bound

    def match(self, load_value: SSAOperand) -> CudaGlobalTilePlan:
        load = self.definitions.require(load_value, "load")
        pointer, mask, other = load.operands

        outer_addptr = self.definitions.require(pointer, "addptr")
        row_pointer, columns = outer_addptr.operands

        inner_addptr = self.definitions.require(row_pointer, "addptr")
        base, row_offset_expression = inner_addptr.operands

        row_stride_mul = self.definitions.require(
            row_offset_expression,
            "mul",
        )
        row_stride_lhs, row_stride_rhs = row_stride_mul.operands
        rows, row_stride = self._split_block_and_scalar(
            row_stride_lhs,
            row_stride_rhs,
            opcode="row-stride multiplication",
        )

        if not isinstance(columns, SSAValue):
            raise TypeError(
                f"dot staging expects block-shaped column coordinates, got {columns}"
            )

        if not isinstance(base, Param) or not isinstance(
            base.ty,
            PointerType,
        ):
            raise TypeError(
                f"dot staging expects a global pointer parameter, got {base}"
            )

        if (
            not isinstance(other, Const)
            or type(other.value) is not float
            or other.value != 0.0
        ):
            raise TypeError("dot staging requires masked loads with other=0.0")

        row_offset = self._match_axis_offset(rows, axis=1)
        column_offset = self._match_axis_offset(columns, axis=0)
        row_bound, column_bound = self._match_mask(
            mask,
            rows,
            columns,
        )

        return CudaGlobalTilePlan(
            base=base,
            row_offset=row_offset,
            column_offset=column_offset,
            row_stride=row_stride,
            row_bound=row_bound,
            column_bound=column_bound,
            other=other,
        )


@dataclass(frozen=True)
class CudaDotStagingAnalysis:
    dot_plans: dict[int, CudaDotStagingPlan]
    staging_only_ids: frozenset[int]

    @property
    def stageable_dot_ids(self) -> frozenset[int]:
        return frozenset(self.dot_plans)

    def plan_for(self, dot_result_id: int) -> CudaDotStagingPlan:
        try:
            return self.dot_plans[dot_result_id]
        except KeyError as error:
            raise KeyError(
                f"dot result %{dot_result_id} has no staging plan"
            ) from error


class CudaDotStagingAnalyzer:
    def __init__(self, definitions: SSADefinitions) -> None:
        self.definitions = definitions

    def analyze(self) -> CudaDotStagingAnalysis:
        matcher = CudaDotOperandMatcher(self.definitions)

        dot_plans: dict[int, CudaDotStagingPlan] = {}
        staging_dependency_ids: set[int] = set()
        required_scalar_ids: set[int] = set()

        for op in self.definitions.ordered_ops:
            if op.opcode != "dot" or op.result is None:
                continue

            lhs, rhs = op.operands
            if not isinstance(lhs, SSAValue) or not isinstance(
                rhs,
                SSAValue,
            ):
                continue

            lhs_definition = self.definitions.get(lhs)
            rhs_definition = self.definitions.get(rhs)

            # Version 13 examples using zeros remain non-stageable.
            if (
                lhs_definition is None
                or lhs_definition.opcode != "load"
                or rhs_definition is None
                or rhs_definition.opcode != "load"
            ):
                continue

            plan = CudaDotStagingPlan(
                lhs=matcher.match(lhs),
                rhs=matcher.match(rhs),
            )
            dot_plans[op.result.id] = plan
            staging_dependency_ids.update(self.definitions.dependency_ids(lhs, rhs))

            for operand_plan in (plan.lhs, plan.rhs):
                required_scalar_ids.update(
                    self.definitions.dependency_ids(
                        operand_plan.base,
                        operand_plan.row_offset,
                        operand_plan.column_offset,
                        operand_plan.row_stride,
                        operand_plan.row_bound,
                        operand_plan.column_bound,
                        operand_plan.other,
                    )
                )

        external_dependency_ids: set[int] = set()

        for op in self.definitions.ordered_ops:
            if op.opcode == "dot":
                continue

            result_id = op.result.id if op.result is not None else None

            if result_id in staging_dependency_ids:
                continue

            for operand in op.operands:
                if (
                    isinstance(operand, SSAValue)
                    and operand.id in staging_dependency_ids
                ):
                    external_dependency_ids.update(
                        self.definitions.dependency_ids(operand)
                    )

        staging_only_ids = (
            staging_dependency_ids - required_scalar_ids - external_dependency_ids
        )

        return CudaDotStagingAnalysis(
            dot_plans=dot_plans,
            staging_only_ids=frozenset(staging_only_ids),
        )


@dataclass(frozen=True)
class CudaDotDoubleBufferingPlan:
    dot_result_id: int
    stage_count: int = 2


def match_cuda_dot_double_buffering(
    loop: SSAForRange,
    staging_analysis: CudaDotStagingAnalysis,
) -> CudaDotDoubleBufferingPlan | None:
    if (
        not isinstance(loop.start, Const)
        or type(loop.start.value) is not int
        or loop.start.value != 0
    ):
        return None

    if (
        not isinstance(loop.step, Const)
        or type(loop.step.value) is not int
        or loop.step.value <= 0
    ):
        return None

    if any(isinstance(item, SSAForRange) for item in loop.body):
        return None

    dots = [
        item
        for item in loop.body
        if (
            isinstance(item, SSAOp)
            and item.opcode == "dot"
            and item.result is not None
            and item.result.id in staging_analysis.stageable_dot_ids
        )
    ]

    if len(dots) != 1:
        return None

    dot = dots[0]
    assert dot.result is not None

    lhs, rhs = dot.operands

    if (
        not isinstance(lhs, SSAValue)
        or not isinstance(lhs.ty, BlockType)
        or not isinstance(rhs, SSAValue)
        or not isinstance(rhs.ty, BlockType)
    ):
        return None

    reduction_size = lhs.ty.shape[1]

    if loop.step.value != reduction_size:
        return None

    staging_plan = staging_analysis.plan_for(dot.result.id)

    if staging_plan.lhs.column_offset != loop.index:
        return None

    if staging_plan.rhs.row_offset != loop.index:
        return None

    emitted_ops = [
        item
        for item in loop.body
        if (
            isinstance(item, SSAOp)
            and (
                item.result is None
                or item.result.id not in staging_analysis.staging_only_ids
            )
        )
    ]

    if len(emitted_ops) != 2 or emitted_ops[0] is not dot:
        return None

    accumulation = emitted_ops[1]

    if (
        accumulation.opcode != "add"
        or accumulation.result is None
        or dot.result not in accumulation.operands
    ):
        return None

    if (
        len(loop.carried_inputs) != 1
        or len(loop.carried_args) != 1
        or len(loop.yields) != 1
        or len(loop.results) != 1
    ):
        return None

    if loop.carried_args[0] not in accumulation.operands:
        return None

    if loop.yields[0] != accumulation.result:
        return None

    return CudaDotDoubleBufferingPlan(
        dot_result_id=dot.result.id,
    )
