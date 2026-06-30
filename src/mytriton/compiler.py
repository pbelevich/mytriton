import inspect
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, ParamSpec, TypeAlias, cast

from .cuda_codegen import SSACUDACodegen
from .cuda_utils import (
    CudaKernelCache,
    cuda_chip,
    execute_cuda_if_needed,
    execute_mlir_cuda_binary_if_needed,
)
from .mlir_backend import (
    MLIRLoweringResult,
    executable_gpu_to_binary_stages,
    extract_gpu_binary,
    mlir_available,
    run_mlir_pipeline_stages,
)
from .mlir_codegen import SSAGPUExecutableMLIRCodegen
from .optim import ConstantFoldPass, CSEPass, DCEPass, PassManager
from .ssa import SSALowering, SSAOp
from .ssa_verification import SSAVerifier
from .trace import (
    Param,
    VectorType,
    is_constexpr_annotation,
    make_runtime_params,
    trace,
)

P = ParamSpec("P")
Meta: TypeAlias = dict[str, Any]
LaunchDimensions: TypeAlias = int | Sequence[int]
Grid: TypeAlias = LaunchDimensions | Callable[[Meta], LaunchDimensions]
CompilationResult: TypeAlias = tuple[list[Any], list[SSAOp], str]
Backend: TypeAlias = Literal["cuda", "mlir"]


def _cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    vector_widths = {
        op.result.ty.size
        for op in ssa_ops
        if op.result is not None and isinstance(op.result.ty, VectorType)
    }

    if not vector_widths:
        return 1

    if len(vector_widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(vector_widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

    threads_per_block = next(iter(vector_widths))
    if not 1 <= threads_per_block <= 1024:
        raise ValueError(
            "CUDA threads per block must be between 1 and 1024, "
            f"got {threads_per_block}"
        )

    return threads_per_block


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
    ssa_ops: tuple[SSAOp, ...]
    cuda_src: str
    threads_per_block: int
    params: tuple[Param, ...]


@dataclass(frozen=True)
class _MLIRArtifact:
    source: str
    lowering: MLIRLoweringResult
    cubin: bytes | None


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
        self.last_mlir_source: str | None = None
        self.last_mlir_lowering: MLIRLoweringResult | None = None
        self.last_mlir_cubin: bytes | None = None

    def _clear_mlir_artifacts(self) -> None:
        self.last_mlir_source = None
        self.last_mlir_lowering = None
        self.last_mlir_cubin = None

    def clear_cache(self) -> None:
        self.compilation_cache.clear()
        self.cuda_cache.clear()
        self._clear_mlir_artifacts()

    def _resolve_backend(self) -> Backend:
        selected = os.environ.get("MYTRITON_BACKEND", "cuda")
        if selected not in {"cuda", "mlir"}:
            raise ValueError(
                f"unsupported backend {selected!r}; expected 'cuda' or 'mlir'"
            )

        return cast(Backend, selected)

    def _run_mlir_if_needed(
        self,
        *,
        artifact: _CompilationArtifact,
        launch_grid: tuple[int, ...],
        chip: str,
    ) -> _MLIRArtifact:
        self._clear_mlir_artifacts()

        if not mlir_available():
            raise RuntimeError(
                "MLIR backend requested, but MLIR bindings are unavailable"
            )

        mlir_src = SSAGPUExecutableMLIRCodegen().generate(
            self.fn.__name__,
            list(artifact.ssa_ops),
            list(artifact.params),
            grid_x=launch_grid[0],
            block_x=artifact.threads_per_block,
        )
        lowering = run_mlir_pipeline_stages(
            mlir_src,
            executable_gpu_to_binary_stages(chip=chip),
        )

        self.last_mlir_source = mlir_src
        self.last_mlir_lowering = lowering
        if not lowering.ok:
            raise RuntimeError(
                "MLIR backend failed to lower executable GPU module: "
                f"{lowering.first_error}"
            )

        cubin = extract_gpu_binary(lowering.final_output)
        self.last_mlir_cubin = cubin
        return _MLIRArtifact(source=mlir_src, lowering=lowering, cubin=cubin)

    def __getitem__(self, grid: Grid) -> Callable[P, CompilationResult]:

        def launch(*args: P.args, **kwargs: P.kwargs) -> CompilationResult:
            backend = self._resolve_backend()

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
            cache_key = (
                tuple(param.ty for param in params),
                _constexpr_key(meta),
            )

            artifact = self.compilation_cache.get(cache_key)
            if artifact is None:
                ops, params = trace(
                    self.fn,
                    self.signature,
                    bound.arguments,
                    runtime_params=params,
                )
                ssa_ops = SSALowering().lower(ops)
                threads_per_block = _cuda_threads_per_block(ssa_ops)

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
                threads_per_block = _cuda_threads_per_block(ssa_ops)
                SSAVerifier(threads_per_block).verify(ssa_ops)

                cuda_src = SSACUDACodegen().generate(
                    self.fn.__name__,
                    ssa_ops,
                    params,
                )
                artifact = _CompilationArtifact(
                    ops=tuple(ops),
                    ssa_ops=tuple(ssa_ops),
                    cuda_src=cuda_src,
                    threads_per_block=threads_per_block,
                    params=tuple(params),
                )
                self.compilation_cache[cache_key] = artifact

            try:
                chip = cuda_chip()
            except Exception:
                chip = "sm_80"

            if backend == "mlir":
                mlir_artifact = self._run_mlir_if_needed(
                    artifact=artifact,
                    launch_grid=launch_grid,
                    chip=chip,
                )
                if mlir_artifact.cubin is None:
                    raise RuntimeError("MLIR backend did not produce a GPU binary")

                mlir_executed = execute_mlir_cuda_binary_if_needed(
                    kernel_cache=self.cuda_cache,
                    cubin=mlir_artifact.cubin,
                    kernel_name=self.fn.__name__,
                    launch_grid=launch_grid,
                    threads_per_block=artifact.threads_per_block,
                    runtime_args=runtime_args,
                )
                if mlir_executed:
                    return (
                        list(deepcopy(artifact.ops)),
                        list(deepcopy(artifact.ssa_ops)),
                        artifact.cuda_src,
                    )

            if backend == "cuda":
                self._clear_mlir_artifacts()
                execute_cuda_if_needed(
                    kernel_cache=self.cuda_cache,
                    cuda_src=artifact.cuda_src,
                    kernel_name=self.fn.__name__,
                    launch_grid=launch_grid,
                    threads_per_block=artifact.threads_per_block,
                    runtime_args=runtime_args,
                )

            return (
                list(deepcopy(artifact.ops)),
                list(deepcopy(artifact.ssa_ops)),
                artifact.cuda_src,
            )

        return launch


def jit(fn: Callable[P, Any]) -> CompiledKernel[P]:
    return CompiledKernel(fn)
