from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import Enum

FUNC_CLEANUP_PIPELINE = "builtin.module(func.func(cse,canonicalize))"

GPU_CLEANUP_PIPELINE = "builtin.module(gpu.module(gpu.func(cse,canonicalize)))"

GENERIC_CLEANUP_PIPELINE = "builtin.module(any(cse,canonicalize))"

NVVM_ATTACH_TARGET_PIPELINE = (
    "builtin.module(nvvm-attach-target{module=.* chip=sm_80 O=3})"
)

GPU_LOWER_TO_NVVM_PIPELINE = (
    "builtin.module("
    "gpu-lower-to-nvvm-pipeline{"
    "cubin-chip=sm_80 "
    "cubin-features=+ptx80 "
    "opt-level=3"
    "}"
    ")"
)


class MLIRUnavailableError(RuntimeError):
    pass


class MLIRPassPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class MLIRPipelineResult:
    ok: bool
    output: str
    error: str | None = None


class MLIRStageStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class MLIRPipelineStage:
    name: str
    pipeline: str


@dataclass(frozen=True)
class MLIRStageResult:
    name: str
    pipeline: str
    status: MLIRStageStatus
    input_text: str
    output_text: str
    error: str | None = None


@dataclass(frozen=True)
class MLIRLoweringResult:
    stages: tuple[MLIRStageResult, ...]

    @property
    def ok(self) -> bool:
        return all(stage.status == MLIRStageStatus.OK for stage in self.stages)

    @property
    def final_output(self) -> str:
        if not self.stages:
            return ""

        return self.stages[-1].output_text

    @property
    def first_error(self) -> str | None:
        for stage in self.stages:
            if stage.error is not None:
                return stage.error

        return None


def mlir_available() -> bool:
    try:
        import mlir.ir
        import mlir.passmanager  # noqa: F401
    except ImportError:
        return False

    return True


def require_mlir():
    try:
        from mlir.dialects import arith, func, gpu, memref, scf  # noqa: F401
        from mlir.ir import Context, Location, Module
        from mlir.passmanager import PassManager

        # These may be needed for NVVM lowering / target attrs in fuller builds.
        with suppress(ImportError):
            from mlir.dialects import llvm, nvgpu, nvvm  # noqa: F401

    except ImportError as error:
        raise MLIRUnavailableError(
            "MLIR Python bindings are not installed or common dialect bindings "
            "are not importable."
        ) from error

    return Context, Location, Module, PassManager


def parse_mlir_module(mlir_text: str) -> str:
    Context, Location, Module, _ = require_mlir()

    with Context() as _, Location.unknown():
        module = Module.parse(mlir_text)
        return str(module)


def run_mlir_pass_pipeline(mlir_text: str, pipeline: str) -> str:
    Context, Location, Module, PassManager = require_mlir()

    try:
        with Context() as _, Location.unknown():
            module = Module.parse(mlir_text)
            pm = PassManager.parse(pipeline)
            pm.run(module.operation)
            return str(module)
    except Exception as error:
        raise MLIRPassPipelineError(
            f"failed to run MLIR pass pipeline {pipeline!r}: {error}"
        ) from error


def run_mlir_pass_pipelines(mlir_text: str, pipelines: list[str]) -> str:
    output = mlir_text
    for pipeline in pipelines:
        output = run_mlir_pass_pipeline(output, pipeline)
    return output


def try_run_mlir_pass_pipeline(mlir_text: str, pipeline: str) -> MLIRPipelineResult:
    try:
        return MLIRPipelineResult(
            ok=True,
            output=run_mlir_pass_pipeline(mlir_text, pipeline),
        )
    except Exception as error:
        return MLIRPipelineResult(
            ok=False,
            output=mlir_text,
            error=str(error),
        )


def try_run_mlir_pass_pipelines(
    mlir_text: str,
    pipelines: list[str],
) -> MLIRPipelineResult:
    try:
        return MLIRPipelineResult(
            ok=True,
            output=run_mlir_pass_pipelines(mlir_text, pipelines),
        )
    except Exception as error:
        return MLIRPipelineResult(
            ok=False,
            output=mlir_text,
            error=str(error),
        )


def nvvm_attach_target_pipeline(
    *,
    module: str = ".*",
    chip: str = "sm_80",
    opt_level: int = 3,
) -> str:
    return (
        "builtin.module("
        f"nvvm-attach-target{{module={module} chip={chip} O={opt_level}}}"
        ")"
    )


def gpu_lower_to_nvvm_pipeline(
    *,
    chip: str = "sm_80",
    features: str = "+ptx80",
    opt_level: int = 3,
) -> str:
    return (
        "builtin.module("
        "gpu-lower-to-nvvm-pipeline{"
        f"cubin-chip={chip} "
        f"cubin-features={features} "
        f"opt-level={opt_level}"
        "}"
        ")"
    )


def run_mlir_pipeline_stages(
    mlir_text: str,
    stages: list[MLIRPipelineStage],
    *,
    stop_on_error: bool = True,
) -> MLIRLoweringResult:
    results: list[MLIRStageResult] = []
    current = mlir_text
    failed = False

    for stage in stages:
        if failed:
            results.append(
                MLIRStageResult(
                    name=stage.name,
                    pipeline=stage.pipeline,
                    status=MLIRStageStatus.SKIPPED,
                    input_text=current,
                    output_text=current,
                    error="skipped because previous stage failed",
                )
            )
            continue

        try:
            output = run_mlir_pass_pipeline(current, stage.pipeline)

            results.append(
                MLIRStageResult(
                    name=stage.name,
                    pipeline=stage.pipeline,
                    status=MLIRStageStatus.OK,
                    input_text=current,
                    output_text=output,
                    error=None,
                )
            )

            current = output

        except Exception as error:
            failed = True

            results.append(
                MLIRStageResult(
                    name=stage.name,
                    pipeline=stage.pipeline,
                    status=MLIRStageStatus.FAILED,
                    input_text=current,
                    output_text=current,
                    error=str(error),
                )
            )

            if not stop_on_error:
                continue

    return MLIRLoweringResult(tuple(results))


def default_gpu_to_nvvm_stages(
    *,
    chip: str = "sm_80",
    features: str = "+ptx80",
    opt_level: int = 3,
) -> list[MLIRPipelineStage]:
    return [
        MLIRPipelineStage(
            name="gpu-cleanup",
            pipeline=GPU_CLEANUP_PIPELINE,
        ),
        MLIRPipelineStage(
            name="attach-nvvm-target",
            pipeline=nvvm_attach_target_pipeline(
                chip=chip,
                opt_level=opt_level,
            ),
        ),
        MLIRPipelineStage(
            name="gpu-lower-to-nvvm",
            pipeline=gpu_lower_to_nvvm_pipeline(
                chip=chip,
                features=features,
                opt_level=opt_level,
            ),
        ),
    ]


def format_mlir_lowering_report(result: MLIRLoweringResult) -> str:
    lines: list[str] = []

    for index, stage in enumerate(result.stages):
        lines.append(f"=== Stage {index}: {stage.name} ===")
        lines.append(f"status: {stage.status.value}")
        lines.append(f"pipeline: {stage.pipeline}")

        if stage.error is not None:
            lines.append("")
            lines.append("error:")
            lines.append(stage.error)

        lines.append("")

    if result.ok:
        lines.append("MLIR lowering pipeline completed successfully.")
    else:
        lines.append("MLIR lowering pipeline failed.")
        if result.first_error is not None:
            lines.append("")
            lines.append("first error:")
            lines.append(result.first_error)

    return "\n".join(lines)


def executable_gpu_to_binary_stages(
    *,
    chip: str = "sm_80",
    opt_level: int = 3,
) -> list[MLIRPipelineStage]:
    return [
        MLIRPipelineStage(
            name="generic-cleanup",
            pipeline=GENERIC_CLEANUP_PIPELINE,
        ),
        MLIRPipelineStage(
            name="attach-nvvm-target",
            pipeline=nvvm_attach_target_pipeline(
                chip=chip,
                opt_level=opt_level,
            ),
        ),
        MLIRPipelineStage(
            name="convert-gpu-to-nvvm",
            pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm))",
        ),
        MLIRPipelineStage(
            name="gpu-to-llvm",
            pipeline="builtin.module(gpu-to-llvm)",
        ),
        MLIRPipelineStage(
            name="gpu-module-to-binary",
            pipeline="builtin.module(gpu-module-to-binary)",
        ),
    ]
