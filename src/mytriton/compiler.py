import inspect
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import (
    Any,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    cast,
    overload,
)

from .ast_frontend import trace
from .block_shapes import (
    CudaMatmulSchedule,
    cuda_threads_per_block,
    infer_cuda_matmul_schedule,
)
from .cuda_codegen import SSACUDACodegen
from .cuda_target import DEFAULT_CUDA_TARGET, CudaTarget
from .cuda_utils import (
    CudaKernelCache,
    CudaUnavailableError,
    cuda_chip,
    cuda_execution_required,
    execute_cuda_if_needed,
    execute_mlir_cubin_if_needed,
)
from .mlir_codegen import MLIRCodegen, compile_mlir_source_to_cubin
from .optim import ConstantFoldPass, CSEPass, DCEPass, PassManager
from .runtime_specialization import runtime_specialization
from .ssa import SSAItem, SSALowering
from .ssa_verification import SSAVerifier
from .trace import (
    BF16,
    F16,
    ScalarType,
    is_constexpr_annotation,
    make_runtime_params,
)

P = ParamSpec("P")
Meta: TypeAlias = dict[str, Any]
LaunchDimensions: TypeAlias = int | Sequence[int]
Grid: TypeAlias = LaunchDimensions | Callable[[Meta], LaunchDimensions]
CompilationResult: TypeAlias = tuple[list[Any], list[SSAItem], str]


class KernelLauncher(Protocol[P]):
    @overload
    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> CompilationResult: ...

    @overload
    def __call__(
        self,
        *args: Any,
        num_warps: int,
        **kwargs: Any,
    ) -> CompilationResult: ...


Backend: TypeAlias = Literal["cuda", "mlir"]


def _cuda_mma_operand_types(
    target: CudaTarget,
) -> frozenset[ScalarType]:
    operand_types: set[ScalarType] = set()

    if target.supports_f16_mma_m16n8k8:
        operand_types.add(F16)

    if target.supports_bf16_mma_m16n8k8:
        operand_types.add(BF16)

    return frozenset(operand_types)


def _constexpr_key(
    meta: dict[str, object],
) -> tuple[tuple[str, type[object], object], ...]:
    supported = (bool, int, float, str)
    entries = []
    for name, value in meta.items():
        if not isinstance(value, supported):
            raise TypeError(f"{name}: constexpr value must be bool, int, float, or str")

        normalized = float(value).hex() if isinstance(value, float) else value
        entries.append((name, type(value), normalized))

    return tuple(entries)


@dataclass(frozen=True)
class _CompilationArtifact:
    ops: tuple[Any, ...]
    ssa_ops: tuple[SSAItem, ...]
    src: str
    threads_per_block: int
    cubin: bytes | None = None


class CompiledKernel(Generic[P]):
    def __init__(self, fn: Callable[P, Any]) -> None:
        self.fn = fn
        signature = inspect.signature(fn)
        annotations = inspect.get_annotations(fn, eval_str=True)
        parameters = [
            parameter.replace(
                annotation=annotations.get(name, parameter.annotation),
            )
            for name, parameter in signature.parameters.items()
        ]
        self.signature = signature.replace(parameters=parameters)
        self.compilation_cache: dict[tuple[object, ...], _CompilationArtifact] = {}
        self.cuda_cache: CudaKernelCache = {}

    def clear_cache(self) -> None:
        self.compilation_cache.clear()
        self.cuda_cache.clear()

    def _resolve_backend(self) -> Backend:
        selected = os.environ.get("MYTRITON_BACKEND", "cuda")
        if selected not in {"cuda", "mlir"}:
            raise ValueError(
                f"unsupported backend {selected!r}; expected 'cuda' or 'mlir'"
            )
        return cast(Backend, selected)

    def __getitem__(
        self,
        grid: Grid,
    ) -> KernelLauncher[P]:

        def launch(
            *args: Any,
            **kwargs: Any,
        ) -> CompilationResult:
            num_warps: int | None = kwargs.pop(
                "num_warps",
                None,
            )

            if num_warps is not None and (
                type(num_warps) is not int
                or num_warps <= 0
                or num_warps & (num_warps - 1)
            ):
                raise ValueError(
                    f"num_warps must be a positive power of two, got {num_warps}"
                )

            if num_warps is not None and num_warps > 32:
                raise ValueError(
                    f"CUDA kernels support at most 32 warps per block, got {num_warps}"
                )

            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()

            meta: Meta = {
                name: bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if is_constexpr_annotation(parameter.annotation)
            }

            launch_grid = grid(meta) if callable(grid) else grid
            if isinstance(launch_grid, int):
                launch_grid = (launch_grid,)
            launch_grid = tuple(launch_grid)

            if not 1 <= len(launch_grid) <= 3 or any(
                type(size) is not int or size <= 0 for size in launch_grid
            ):
                raise ValueError(f"invalid launch grid: {launch_grid}")

            runtime_args = tuple(
                bound.arguments[name]
                for name, parameter in self.signature.parameters.items()
                if not is_constexpr_annotation(parameter.annotation)
            )
            params = make_runtime_params(self.signature, bound.arguments)
            specializations = {
                param.name: runtime_specialization(value)
                for param, value in zip(params, runtime_args, strict=True)
            }
            backend = self._resolve_backend()
            chip = None
            cuda_target = DEFAULT_CUDA_TARGET
            if backend == "cuda":
                if cuda_execution_required(runtime_args, backend_name="CUDA"):
                    chip = cuda_chip(runtime_args)
                    cuda_target = CudaTarget.from_chip(chip)
                else:
                    chip = cuda_target.chip
            else:
                try:
                    chip = cuda_chip(runtime_args)
                except CudaUnavailableError:
                    chip = "sm_80"

            if backend != "cuda" and num_warps is not None:
                raise ValueError(
                    "num_warps launch option is currently supported "
                    "only by the CUDA backend"
                )

            supported_mma_operand_types = _cuda_mma_operand_types(cuda_target)

            cache_key = (
                backend,
                chip,
                tuple(param.ty for param in params),
                tuple(specializations[param.name] for param in params),
                _constexpr_key(meta),
                num_warps,
            )

            artifact = self.compilation_cache.get(cache_key)
            if artifact is None:
                trace_fn = trace
                ops, params = trace_fn(
                    self.fn,
                    self.signature,
                    bound.arguments,
                    runtime_params=params,
                )
                ssa_ops = SSALowering().lower(ops)

                matmul_schedule: CudaMatmulSchedule | None = None
                if num_warps is not None:
                    matmul_schedule = infer_cuda_matmul_schedule(
                        ssa_ops,
                        num_warps=num_warps,
                        operand_types=supported_mma_operand_types,
                    )
                    if matmul_schedule is None:
                        raise ValueError(
                            "num_warps currently requires a "
                            "tensor-core-compatible tl.dot"
                        )

                threads_per_block = cuda_threads_per_block(
                    ssa_ops,
                    matmul_schedule=matmul_schedule,
                    mma_operand_types=supported_mma_operand_types,
                )

                # The optimizer assumes lowering produced valid SSA.
                verifier = SSAVerifier(threads_per_block)
                verifier.verify(ssa_ops)

                # PassManager re-runs the verifier after each rewrite pass, so
                # a broken optimization fails before CUDA codegen sees it.
                ssa_ops = PassManager(
                    passes=[
                        ConstantFoldPass(),
                        CSEPass(),
                        DCEPass(),
                    ],
                    verifier=verifier,
                ).run(ssa_ops)

                # Recompute after optimization: DCE/CSE may remove the vector
                # ops that originally determined the CUDA block size.
                if num_warps is not None:
                    matmul_schedule = infer_cuda_matmul_schedule(
                        ssa_ops,
                        num_warps=num_warps,
                        operand_types=supported_mma_operand_types,
                    )
                    if matmul_schedule is None:
                        raise ValueError(
                            "optimized SSA no longer contains a "
                            "tensor-core-compatible tl.dot"
                        )

                threads_per_block = cuda_threads_per_block(
                    ssa_ops,
                    matmul_schedule=matmul_schedule,
                    mma_operand_types=supported_mma_operand_types,
                )
                SSAVerifier(threads_per_block).verify(ssa_ops)

                cubin = None
                if backend == "mlir":
                    src = MLIRCodegen().generate(self.fn.__name__, ssa_ops, params)
                else:
                    src = SSACUDACodegen(target=cuda_target).generate(
                        self.fn.__name__,
                        ssa_ops,
                        params,
                        matmul_schedule=matmul_schedule,
                        runtime_specializations=specializations,
                    )

                artifact = _CompilationArtifact(
                    ops=tuple(ops),
                    ssa_ops=tuple(ssa_ops),
                    src=src,
                    threads_per_block=threads_per_block,
                    cubin=cubin,
                )
                self.compilation_cache[cache_key] = artifact

            if backend == "mlir":
                needs_execution = cuda_execution_required(
                    runtime_args, backend_name="MLIR"
                )
                if artifact.cubin is None and needs_execution:
                    assert chip is not None
                    artifact = replace(
                        artifact,
                        cubin=compile_mlir_source_to_cubin(artifact.src, chip=chip),
                    )
                    self.compilation_cache[cache_key] = artifact

                if needs_execution:
                    assert artifact.cubin is not None
                    execute_mlir_cubin_if_needed(
                        kernel_cache=self.cuda_cache,
                        cubin=artifact.cubin,
                        kernel_name=self.fn.__name__,
                        launch_grid=launch_grid,
                        threads_per_block=artifact.threads_per_block,
                        runtime_args=runtime_args,
                    )
            else:
                execute_cuda_if_needed(
                    kernel_cache=self.cuda_cache,
                    cuda_src=artifact.src,
                    kernel_name=self.fn.__name__,
                    launch_grid=launch_grid,
                    threads_per_block=artifact.threads_per_block,
                    runtime_args=runtime_args,
                )

            return (
                list(deepcopy(artifact.ops)),
                list(deepcopy(artifact.ssa_ops)),
                artifact.src,
            )

        return cast(KernelLauncher[P], launch)


def jit(fn: Callable[P, Any]) -> CompiledKernel[P]:
    return CompiledKernel(fn)
