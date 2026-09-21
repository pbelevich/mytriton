from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cuda_target import CudaTarget
from .trace import BF16, F16, F32, BlockType, ScalarType

if TYPE_CHECKING:
    from .ssa import SSAItem


def prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def cuda_mma_m16n8k8_operand_types(
    target: CudaTarget,
) -> frozenset[ScalarType]:
    """Return low-precision operand types supported by the CUDA target."""

    operand_types: set[ScalarType] = set()

    if target.supports_f16_mma_m16n8k8:
        operand_types.add(F16)

    if target.supports_bf16_mma_m16n8k8:
        operand_types.add(BF16)

    return frozenset(operand_types)


@dataclass(frozen=True)
class CudaMmaM16N8K8Layout:
    """Per-lane fragment layout for PTX mma.m16n8k8."""

    lhs_shape = (16, 8)
    rhs_shape = (8, 8)
    result_shape = (16, 8)
    threads_per_warp = 32
    thread_shape = (4, 8)

    def lane_parts(self, lane: int) -> tuple[int, int]:
        if type(lane) is not int or not 0 <= lane < self.threads_per_warp:
            raise ValueError(f"MMA lane must be between 0 and 31, got {lane}")

        group_id = lane // 4
        thread_id_in_group = lane % 4
        return group_id, thread_id_in_group

    def lhs_coordinates(
        self,
        lane: int,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        group_id, thread_id = self.lane_parts(lane)
        first_column = thread_id * 2

        return (
            (group_id, first_column),
            (group_id, first_column + 1),
            (group_id + 8, first_column),
            (group_id + 8, first_column + 1),
        )

    def rhs_coordinates(
        self,
        lane: int,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
    ]:
        group_id, thread_id = self.lane_parts(lane)
        first_row = thread_id * 2

        return (
            (first_row, group_id),
            (first_row + 1, group_id),
        )

    def accumulator_coordinates(
        self,
        lane: int,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        group_id, thread_id = self.lane_parts(lane)
        first_column = thread_id * 2

        return (
            (group_id, first_column),
            (group_id, first_column + 1),
            (group_id + 8, first_column),
            (group_id + 8, first_column + 1),
        )


@dataclass(frozen=True)
class CudaMmaWarpTileLayout:
    """One logical warp tile composed from m16n8k8 MMA instructions."""

    m: int
    n: int
    k: int

    instruction_layout = CudaMmaM16N8K8Layout()
    threads_per_warp = instruction_layout.threads_per_warp
    thread_shape = instruction_layout.thread_shape

    def __post_init__(self) -> None:
        instruction_m, instruction_n = self.instruction_layout.result_shape
        instruction_k = self.instruction_layout.lhs_shape[1]

        for name, value, multiple in (
            ("M", self.m, instruction_m),
            ("N", self.n, instruction_n),
            ("K", self.k, instruction_k),
        ):
            if type(value) is not int or value <= 0 or value % multiple != 0:
                raise ValueError(
                    f"{name} dimension must be a positive multiple "
                    f"of {multiple}, got {value}"
                )

    @property
    def lhs_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def rhs_shape(self) -> tuple[int, int]:
        return (self.k, self.n)

    @property
    def result_shape(self) -> tuple[int, int]:
        return (self.m, self.n)

    @property
    def instruction_shape(self) -> tuple[int, int, int]:
        instruction_m, instruction_n = self.instruction_layout.result_shape
        instruction_k = self.instruction_layout.lhs_shape[1]
        return (instruction_m, instruction_n, instruction_k)

    @property
    def m_tiles(self) -> int:
        return self.m // self.instruction_shape[0]

    @property
    def n_tiles(self) -> int:
        return self.n // self.instruction_shape[1]

    @property
    def k_tiles(self) -> int:
        return self.k // self.instruction_shape[2]

    @property
    def instruction_count(self) -> int:
        return self.m_tiles * self.n_tiles * self.k_tiles

    @property
    def accumulator_fragments_per_lane(self) -> int:
        return self.m_tiles * self.n_tiles

    @property
    def accumulator_elements_per_lane(self) -> int:
        elements_per_fragment = len(self.instruction_layout.accumulator_coordinates(0))
        return self.accumulator_fragments_per_lane * elements_per_fragment

    def instruction_coordinates(
        self,
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (m_tile, n_tile, k_tile)
            for m_tile in range(self.m_tiles)
            for n_tile in range(self.n_tiles)
            for k_tile in range(self.k_tiles)
        )

    @staticmethod
    def _offset_fragment_coordinates(
        coordinates: tuple[tuple[int, int], ...],
        *,
        row_offset: int,
        column_offset: int,
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (row + row_offset, column + column_offset) for row, column in coordinates
        )

    @staticmethod
    def _require_tile_index(
        name: str,
        index: int,
        count: int,
    ) -> None:
        if type(index) is not int or not 0 <= index < count:
            raise ValueError(
                f"{name} tile index must be between 0 and {count - 1}, got {index}"
            )

    def lhs_fragment_coordinates(
        self,
        *,
        lane: int,
        m_tile: int,
        k_tile: int,
    ) -> tuple[tuple[int, int], ...]:
        self._require_tile_index("M", m_tile, self.m_tiles)
        self._require_tile_index("K", k_tile, self.k_tiles)

        instruction_m, _, instruction_k = self.instruction_shape

        return self._offset_fragment_coordinates(
            self.instruction_layout.lhs_coordinates(lane),
            row_offset=m_tile * instruction_m,
            column_offset=k_tile * instruction_k,
        )

    def rhs_fragment_coordinates(
        self,
        *,
        lane: int,
        k_tile: int,
        n_tile: int,
    ) -> tuple[tuple[int, int], ...]:
        self._require_tile_index("K", k_tile, self.k_tiles)
        self._require_tile_index("N", n_tile, self.n_tiles)

        _, instruction_n, instruction_k = self.instruction_shape

        return self._offset_fragment_coordinates(
            self.instruction_layout.rhs_coordinates(lane),
            row_offset=k_tile * instruction_k,
            column_offset=n_tile * instruction_n,
        )

    def accumulator_fragment_index(
        self,
        *,
        m_tile: int,
        n_tile: int,
    ) -> int:
        self._require_tile_index("M", m_tile, self.m_tiles)
        self._require_tile_index("N", n_tile, self.n_tiles)

        return m_tile * self.n_tiles + n_tile

    def accumulator_fragment_coordinates(
        self,
        *,
        lane: int,
        m_tile: int,
        n_tile: int,
    ) -> tuple[tuple[int, int], ...]:
        self.accumulator_fragment_index(
            m_tile=m_tile,
            n_tile=n_tile,
        )
        instruction_m, instruction_n, _ = self.instruction_shape

        return self._offset_fragment_coordinates(
            self.instruction_layout.accumulator_coordinates(lane),
            row_offset=m_tile * instruction_m,
            column_offset=n_tile * instruction_n,
        )


@dataclass(frozen=True)
class CudaMmaCtaTileLayout:
    """A CTA tile composed from a two-dimensional grid of warp MMA tiles."""

    warp_tile: CudaMmaWarpTileLayout
    warps_m: int
    warps_n: int

    def __post_init__(self) -> None:
        for name, value in (
            ("warps_m", self.warps_m),
            ("warps_n", self.warps_n),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")

        if self.threads_per_block > 1024:
            raise ValueError(
                "CTA MMA tile requires at most 1024 threads, "
                f"got {self.threads_per_block}"
            )

    @property
    def warp_shape(self) -> tuple[int, int]:
        return (self.warps_m, self.warps_n)

    @property
    def warp_count(self) -> int:
        return self.warps_m * self.warps_n

    @property
    def threads_per_block(self) -> int:
        return self.warp_count * self.warp_tile.threads_per_warp

    @property
    def thread_shape(self) -> tuple[int, int]:
        warp_threads_m, warp_threads_n = self.warp_tile.thread_shape
        return (
            self.warps_m * warp_threads_m,
            self.warps_n * warp_threads_n,
        )

    @property
    def lhs_shape(self) -> tuple[int, int]:
        return (
            self.warps_m * self.warp_tile.m,
            self.warp_tile.k,
        )

    @property
    def rhs_shape(self) -> tuple[int, int]:
        return (
            self.warp_tile.k,
            self.warps_n * self.warp_tile.n,
        )

    @property
    def result_shape(self) -> tuple[int, int]:
        return (
            self.warps_m * self.warp_tile.m,
            self.warps_n * self.warp_tile.n,
        )

    def warp_coordinates(self, warp_id: int) -> tuple[int, int]:
        if type(warp_id) is not int or not 0 <= warp_id < self.warp_count:
            raise ValueError(
                f"warp ID must be between 0 and {self.warp_count - 1}, got {warp_id}"
            )

        return divmod(warp_id, self.warps_n)

    def warp_result_offset(self, warp_id: int) -> tuple[int, int]:
        warp_m, warp_n = self.warp_coordinates(warp_id)
        return (
            warp_m * self.warp_tile.m,
            warp_n * self.warp_tile.n,
        )

    def warp_lhs_offset(self, warp_id: int) -> tuple[int, int]:
        warp_m, _ = self.warp_coordinates(warp_id)
        return (
            warp_m * self.warp_tile.m,
            0,
        )

    def warp_rhs_offset(self, warp_id: int) -> tuple[int, int]:
        _, warp_n = self.warp_coordinates(warp_id)
        return (
            0,
            warp_n * self.warp_tile.n,
        )


def cuda_mma_warp_tile_layout(
    op: SSAItem,
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> CudaMmaWarpTileLayout | None:
    """Return the composable warp layout for a supported low-precision dot."""

    from .ssa import SSAOp, SSAValue

    if not isinstance(op, SSAOp) or op.opcode != "dot":
        return None

    if op.result is None or len(op.operands) != 2:
        return None

    lhs, rhs = op.operands

    if not isinstance(lhs, SSAValue) or not isinstance(rhs, SSAValue):
        return None

    if (
        not isinstance(lhs.ty, BlockType)
        or not isinstance(rhs.ty, BlockType)
        or not isinstance(op.result.ty, BlockType)
    ):
        return None

    if lhs.ty.rank != 2 or rhs.ty.rank != 2 or op.result.ty.rank != 2:
        return None

    operand_ty = lhs.ty.element

    if (
        operand_ty not in operand_types
        or rhs.ty.element != operand_ty
        or op.result.ty.element != F32
    ):
        return None

    lhs_m, lhs_k = lhs.ty.shape
    rhs_k, rhs_n = rhs.ty.shape

    if lhs_k != rhs_k:
        return None

    if op.result.ty.shape != (lhs_m, rhs_n):
        return None

    try:
        return CudaMmaWarpTileLayout(
            m=lhs_m,
            n=rhs_n,
            k=lhs_k,
        )
    except ValueError:
        return None


CUDA_MMA_CTA_WARP_TILE_SHAPE = (32, 32)


def cuda_mma_cta_tile_layout(
    op: SSAItem,
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> CudaMmaCtaTileLayout | None:
    """Return a multi-warp CTA layout for a supported low-precision dot."""

    full_tile = cuda_mma_warp_tile_layout(
        op,
        operand_types=operand_types,
    )

    if full_tile is None:
        return None

    warp_m, warp_n = CUDA_MMA_CTA_WARP_TILE_SHAPE

    if full_tile.m % warp_m != 0 or full_tile.n % warp_n != 0:
        return None

    warps_m = full_tile.m // warp_m
    warps_n = full_tile.n // warp_n

    if warps_m * warps_n <= 1:
        return None

    try:
        return CudaMmaCtaTileLayout(
            warp_tile=CudaMmaWarpTileLayout(
                m=warp_m,
                n=warp_n,
                k=full_tile.k,
            ),
            warps_m=warps_m,
            warps_n=warps_n,
        )
    except ValueError:
        return None


def cuda_mma_cta_tile_layouts(
    ssa_ops: list[SSAItem],
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> dict[int, CudaMmaCtaTileLayout]:
    """Collect multi-warp CTA layouts by SSA dot result ID."""

    from .ssa import SSAForRange

    layouts: dict[int, CudaMmaCtaTileLayout] = {}

    for item in ssa_ops:
        if isinstance(item, SSAForRange):
            layouts.update(
                cuda_mma_cta_tile_layouts(
                    item.body,
                    operand_types=operand_types,
                )
            )
            continue

        layout = cuda_mma_cta_tile_layout(
            item,
            operand_types=operand_types,
        )

        if layout is None:
            continue

        assert item.result is not None
        layouts[item.result.id] = layout

    return layouts


def cuda_mma_warp_tile_layouts(
    ssa_ops: list[SSAItem],
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> dict[int, CudaMmaWarpTileLayout]:
    """Collect composable MMA layouts by SSA dot result ID."""

    from .ssa import SSAForRange

    layouts: dict[int, CudaMmaWarpTileLayout] = {}

    for item in ssa_ops:
        if isinstance(item, SSAForRange):
            layouts.update(
                cuda_mma_warp_tile_layouts(
                    item.body,
                    operand_types=operand_types,
                )
            )
            continue

        layout = cuda_mma_warp_tile_layout(
            item,
            operand_types=operand_types,
        )

        if layout is None:
            continue

        assert item.result is not None
        layouts[item.result.id] = layout

    return layouts


def is_cuda_mma_m16n8k8_dot(
    op: SSAItem,
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> bool:
    """Whether an SSA dot matches exactly one m16n8k8 instruction."""

    layout = cuda_mma_warp_tile_layout(
        op,
        operand_types=operand_types,
    )

    return layout is not None and layout.instruction_count == 1


def cuda_mma_m16n8k8_dot_result_ids(
    ssa_ops: list[SSAItem],
    *,
    operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> frozenset[int]:
    """Collect tensor-core-compatible dot result IDs recursively."""

    from .ssa import SSAForRange

    result_ids: set[int] = set()

    for item in ssa_ops:
        if isinstance(item, SSAForRange):
            result_ids.update(
                cuda_mma_m16n8k8_dot_result_ids(
                    item.body,
                    operand_types=operand_types,
                )
            )
            continue

        if not is_cuda_mma_m16n8k8_dot(item, operand_types=operand_types):
            continue

        assert item.result is not None
        result_ids.add(item.result.id)

    return frozenset(result_ids)


@dataclass(frozen=True)
class CudaTileLayout:
    """Mapping from logical tile dimensions to CUDA thread dimensions."""

    logical_shape: tuple[int, ...]
    thread_axes: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.logical_shape) <= 2:
            raise ValueError(
                f"CUDA tile layouts support rank 1 or 2, got {self.logical_shape}"
            )

        if any(type(dim) is not int or dim <= 0 for dim in self.logical_shape):
            raise ValueError(
                f"logical tile dimensions must be positive integers, "
                f"got {self.logical_shape}"
            )

        if len(self.logical_shape) != len(self.thread_axes):
            raise ValueError(
                "logical shape and thread axes must have the same rank, "
                f"got {self.logical_shape} and {self.thread_axes}"
            )

        mapped_axes = [axis for axis in self.thread_axes if axis is not None]

        if any(axis < 0 or axis >= len(self.logical_shape) for axis in mapped_axes):
            raise ValueError(
                f"CUDA thread axes must be valid logical axes, got {self.thread_axes}"
            )

        if len(mapped_axes) != len(set(mapped_axes)):
            raise ValueError(f"CUDA thread axes must be unique, got {self.thread_axes}")


@dataclass(frozen=True)
class CudaCooperativeTileLayout:
    """Distribution of a logical tile across all threads in a CUDA block."""

    logical_shape: tuple[int, ...]
    threads_per_block: int
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.logical_shape) <= 2:
            raise ValueError(
                f"cooperative CUDA tiles support rank 1 or 2, got {self.logical_shape}"
            )

        if any(type(dim) is not int or dim <= 0 for dim in self.logical_shape):
            raise ValueError(
                f"logical tile dimensions must be positive integers, "
                f"got {self.logical_shape}"
            )

        if (
            type(self.threads_per_block) is not int
            or not 1 <= self.threads_per_block <= 1024
        ):
            raise ValueError(
                f"threads per block must be an integer between 1 and 1024, "
                f"got {self.threads_per_block}"
            )

        expected_axes = tuple(range(len(self.logical_shape)))
        if tuple(sorted(self.order)) != expected_axes:
            raise ValueError(
                f"order must be a permutation of {expected_axes}, got {self.order}"
            )

    @property
    def size(self) -> int:
        return prod(self.logical_shape)

    @property
    def iterations_per_thread(self) -> int:
        return (self.size + self.threads_per_block - 1) // self.threads_per_block

    def linear_index(self, thread_index: int, iteration: int) -> int:
        if not 0 <= thread_index < self.threads_per_block:
            raise ValueError(
                f"invalid thread index {thread_index} for "
                f"{self.threads_per_block} threads"
            )

        if not 0 <= iteration < self.iterations_per_thread:
            raise ValueError(
                f"invalid cooperative iteration {iteration}; expected "
                f"0 <= iteration < {self.iterations_per_thread}"
            )

        return thread_index + iteration * self.threads_per_block

    def contains(self, linear_index: int) -> bool:
        return 0 <= linear_index < self.size

    def coordinates(self, linear_index: int) -> tuple[int, ...]:
        if not self.contains(linear_index):
            raise ValueError(
                f"linear index {linear_index} is outside tile {self.logical_shape}"
            )

        coordinates = [0] * len(self.logical_shape)
        remaining = linear_index

        for axis in self.order:
            coordinates[axis] = remaining % self.logical_shape[axis]
            remaining //= self.logical_shape[axis]

        return tuple(coordinates)


@dataclass(frozen=True)
class CudaRegisterTileLayout:
    """Distribution of a logical rank-2 tile across threads and registers."""

    logical_shape: tuple[int, ...]
    thread_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, shape in (
            ("logical tile", self.logical_shape),
            ("CUDA thread", self.thread_shape),
        ):
            if len(shape) != 2:
                raise ValueError(f"{name} shape must have rank 2, got {shape}")

            if any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError(
                    f"{name} dimensions must be positive integers, got {shape}"
                )

        for logical_dim, thread_dim in zip(
            self.logical_shape,
            self.thread_shape,
            strict=True,
        ):
            if logical_dim % thread_dim != 0:
                raise ValueError(
                    "register tile requires logical dimensions divisible "
                    "by CUDA thread dimensions, "
                    f"got {self.logical_shape} and {self.thread_shape}"
                )

    @property
    def register_shape(self) -> tuple[int, ...]:
        return tuple(
            logical_dim // thread_dim
            for logical_dim, thread_dim in zip(
                self.logical_shape,
                self.thread_shape,
                strict=True,
            )
        )

    @property
    def registers_per_thread(self) -> int:
        return prod(self.register_shape)

    def logical_coordinate(
        self,
        *,
        thread_coordinate: tuple[int, ...],
        register_coordinate: tuple[int, ...],
    ) -> tuple[int, ...]:
        for name, coordinate, shape in (
            ("thread", thread_coordinate, self.thread_shape),
            ("register", register_coordinate, self.register_shape),
        ):
            if len(coordinate) != 2 or any(
                type(index) is not int or index < 0 or index >= dim
                for index, dim in zip(
                    coordinate,
                    shape,
                    strict=True,
                )
            ):
                raise ValueError(
                    f"invalid {name} coordinate {coordinate} for shape {shape}"
                )

        return tuple(
            thread_index + register_index * thread_dim
            for thread_index, register_index, thread_dim in zip(
                thread_coordinate,
                register_coordinate,
                self.thread_shape,
                strict=True,
            )
        )


@dataclass(frozen=True)
class CudaKernelLayout:
    """Logical output tile and physical CUDA thread organization."""

    output_tile_shape: tuple[int, ...]
    thread_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, shape in (
            ("output tile", self.output_tile_shape),
            ("CUDA thread", self.thread_shape),
        ):
            if not shape:
                raise ValueError(f"{name} shape must have at least one dimension")

            if len(shape) > 2:
                raise ValueError(f"{name} shape supports rank 1 or 2, got {shape}")

            if any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError(
                    f"{name} dimensions must be positive integers, got {shape}"
                )

        if len(self.output_tile_shape) != len(self.thread_shape):
            raise ValueError(
                "output tile and CUDA thread shapes must have the same rank, "
                f"got {self.output_tile_shape} and {self.thread_shape}"
            )

        threads_per_block = prod(self.thread_shape)
        if threads_per_block > 1024:
            raise ValueError(
                "CUDA threads per block must be between 1 and 1024, "
                f"got {threads_per_block}"
            )

    @property
    def rank(self) -> int:
        return len(self.thread_shape)

    @property
    def is_rank2(self) -> bool:
        return self.rank == 2

    @property
    def threads_per_block(self) -> int:
        return prod(self.thread_shape)

    def register_tile_layout(self) -> CudaRegisterTileLayout:
        return CudaRegisterTileLayout(
            logical_shape=self.output_tile_shape,
            thread_shape=self.thread_shape,
        )

    def tile_layout(
        self,
        logical_shape: tuple[int, ...],
        *,
        broadcast_axes: tuple[int, ...] = (),
    ) -> CudaTileLayout:
        if len(logical_shape) != self.rank:
            raise ValueError(
                f"logical tile rank must match CUDA thread rank: "
                f"{logical_shape} vs {self.thread_shape}"
            )

        if any(axis < 0 or axis >= self.rank for axis in broadcast_axes):
            raise ValueError(
                f"invalid broadcast axes {broadcast_axes} for {logical_shape}"
            )

        thread_axes: list[int | None] = []

        for axis, (logical_dim, thread_dim) in enumerate(
            zip(logical_shape, self.thread_shape, strict=True)
        ):
            if axis in broadcast_axes:
                if logical_dim != 1:
                    raise ValueError(
                        f"broadcast axis {axis} must have size 1, got {logical_shape}"
                    )
                thread_axes.append(None)
            elif logical_dim == thread_dim:
                thread_axes.append(axis)
            elif logical_dim == 1:
                thread_axes.append(None)
            else:
                raise ValueError(
                    f"cannot project logical tile {logical_shape} onto "
                    f"CUDA thread shape {self.thread_shape}"
                )

        return CudaTileLayout(
            logical_shape=logical_shape,
            thread_axes=tuple(thread_axes),
        )

    def cooperative_tile_layout(
        self,
        logical_shape: tuple[int, ...],
        *,
        order: tuple[int, ...] | None = None,
    ) -> CudaCooperativeTileLayout:
        if len(logical_shape) != self.rank:
            raise ValueError(
                f"cooperative tile rank must match CUDA kernel rank: "
                f"{logical_shape} vs {self.thread_shape}"
            )

        if order is None:
            order = tuple(reversed(range(len(logical_shape))))

        return CudaCooperativeTileLayout(
            logical_shape=logical_shape,
            threads_per_block=self.threads_per_block,
            order=order,
        )


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()

    max_rank = max(len(shape) for shape in shapes)
    padded = [(1,) * (max_rank - len(shape)) + shape for shape in shapes]

    dims = []
    for dim_values in zip(*padded, strict=True):
        non_ones = {dim for dim in dim_values if dim != 1}
        if len(non_ones) > 1:
            rendered = ", ".join(
                "x".join(str(dim) for dim in shape) for shape in shapes
            )
            raise ValueError(f"cannot broadcast shapes: {rendered}")

        dims.append(next(iter(non_ones), 1))

    return tuple(dims)


def store_block_shapes(ssa_ops: list[SSAItem]) -> list[tuple[int, ...]]:
    """Collect block shapes that participate in observable memory writes."""

    from .ssa import SSAForRange, SSAValue

    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(store_block_shapes(op.body))
            continue

        if op.opcode != "store":
            continue

        for operand in op.operands:
            if isinstance(operand, SSAValue) and isinstance(operand.ty, BlockType):
                shapes.append(operand.ty.shape)

    return shapes


def reduction_block_shapes(ssa_ops: list[SSAItem]) -> list[tuple[int, ...]]:
    """Collect input shapes of reductions that require cooperative threads."""

    from .ssa import SSAForRange, SSAValue

    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(reduction_block_shapes(op.body))
            continue

        if op.opcode not in {"sum", "max", "min"}:
            continue

        operand = op.operands[0]
        if isinstance(operand, SSAValue) and isinstance(operand.ty, BlockType):
            shapes.append(operand.ty.shape)

    return shapes


def dot_result_shapes(
    ssa_ops: list[SSAItem],
) -> list[tuple[int, ...]]:
    """Collect logical output shapes produced by tl.dot."""

    from .ssa import SSAForRange, SSAValue

    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(dot_result_shapes(op.body))
            continue

        if (
            op.opcode == "dot"
            and isinstance(op.result, SSAValue)
            and isinstance(op.result.ty, BlockType)
        ):
            shapes.append(op.result.ty.shape)

    return shapes


def _infer_cuda_kernel_tile_shape(ssa_ops: list[SSAItem]) -> tuple[int, ...]:
    shapes = store_block_shapes(ssa_ops)

    if not shapes:
        return (1,)

    if any(len(shape) > 2 for shape in shapes):
        rendered = ", ".join(str(shape) for shape in shapes)
        raise ValueError(
            f"CUDA lowering supports only rank-1/rank-2 blocks, got {rendered}"
        )

    rank2_shapes = [shape for shape in shapes if len(shape) == 2]
    rank1_shapes = [shape for shape in shapes if len(shape) == 1]

    if rank2_shapes:
        if rank1_shapes:
            rendered = ", ".join(str(shape) for shape in shapes)
            raise ValueError(
                "CUDA lowering does not support mixed rank-1/rank-2 "
                f"store domains, got {rendered}"
            )

        block_shape = broadcast_shapes(*rank2_shapes)

        if len(block_shape) != 2:
            raise ValueError(f"expected rank-2 CUDA block shape, got {block_shape}")

        return block_shape

    widths = {shape[0] for shape in rank1_shapes}
    if len(widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

    return (next(iter(widths)),)


CUDA_DOT_MAX_THREADS = 32


def _divisors(dim: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in range(1, dim + 1) if dim % candidate == 0)


def _infer_cuda_dot_thread_shape(
    output_tile_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if len(output_tile_shape) != 2:
        raise ValueError(f"CUDA dot output must have rank 2, got {output_tile_shape}")

    rows, columns = output_tile_shape
    candidates = [
        (thread_rows, thread_columns)
        for thread_rows in _divisors(rows)
        for thread_columns in _divisors(columns)
        if (thread_rows * thread_columns <= CUDA_DOT_MAX_THREADS)
    ]

    if not candidates:
        raise ValueError(
            f"cannot fit CUDA dot tile {output_tile_shape} "
            f"into {CUDA_DOT_MAX_THREADS} threads"
        )

    return max(
        candidates,
        key=lambda shape: (
            prod(shape),
            min(shape),
            shape[1],
        ),
    )


def _infer_cuda_thread_shape(
    output_tile_shape: tuple[int, ...],
    reduction_shapes: list[tuple[int, ...]],
    dot_shapes: list[tuple[int, ...]],
) -> tuple[int, ...]:
    if reduction_shapes:
        if any(len(shape) != 1 for shape in reduction_shapes):
            rendered = ", ".join(str(shape) for shape in reduction_shapes)
            raise ValueError(f"CUDA reductions require rank-1 inputs, got {rendered}")

        reduction_widths = {shape[0] for shape in reduction_shapes}
        if len(reduction_widths) != 1:
            rendered = ", ".join(str(width) for width in sorted(reduction_widths))
            raise ValueError(
                f"CUDA reductions require one thread width, got: {rendered}"
            )

        reduction_width = next(iter(reduction_widths))

        if len(output_tile_shape) == 1:
            output_width = output_tile_shape[0]
            if output_width not in (1, reduction_width):
                raise ValueError(
                    f"reduction width {reduction_width} does not match "
                    f"output tile width {output_width}"
                )
            return (reduction_width,)

        if prod(output_tile_shape) != reduction_width:
            raise ValueError(
                f"reduction width {reduction_width} does not match "
                f"output tile size {prod(output_tile_shape)}"
            )

        return output_tile_shape

    if len(output_tile_shape) == 2 and output_tile_shape in dot_shapes:
        return _infer_cuda_dot_thread_shape(output_tile_shape)

    return output_tile_shape


def cuda_kernel_layout(
    ssa_ops: list[SSAItem],
    *,
    mma_operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> CudaKernelLayout:
    output_tile_shape = _infer_cuda_kernel_tile_shape(ssa_ops)

    cta_thread_shapes = {
        layout.thread_shape
        for layout in cuda_mma_cta_tile_layouts(
            ssa_ops,
            operand_types=mma_operand_types,
        ).values()
        if layout.result_shape == output_tile_shape
    }

    if len(cta_thread_shapes) > 1:
        rendered = ", ".join(str(shape) for shape in sorted(cta_thread_shapes))
        raise ValueError(
            f"CUDA lowering requires one multi-warp CTA thread shape, got: {rendered}"
        )

    thread_shape: tuple[int, ...]

    if cta_thread_shapes:
        thread_shape = next(iter(cta_thread_shapes))
    else:
        thread_shape = _infer_cuda_thread_shape(
            output_tile_shape,
            reduction_block_shapes(ssa_ops),
            dot_result_shapes(ssa_ops),
        )

    return CudaKernelLayout(
        output_tile_shape=output_tile_shape,
        thread_shape=thread_shape,
    )


def cuda_threads_per_block(
    ssa_ops: list[SSAItem],
    *,
    mma_operand_types: frozenset[ScalarType] = frozenset((F16,)),
) -> int:
    layout = cuda_kernel_layout(
        ssa_ops,
        mma_operand_types=mma_operand_types,
    )
    threads = layout.threads_per_block
    if not 1 <= threads <= 1024:
        raise ValueError(
            f"CUDA threads per block must be between 1 and 1024, got {threads}"
        )
    return threads
