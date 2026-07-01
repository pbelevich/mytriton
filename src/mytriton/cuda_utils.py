import importlib
from typing import Any, Protocol, TypeGuard

import numpy as np

CudaKernelCache = dict[tuple[object, str], Any]


class _ArrayFlagsLike(Protocol):
    c_contiguous: bool


class _CupyArrayLike(Protocol):
    flags: _ArrayFlagsLike
    ndim: int
    shape: tuple[int, ...]


class CudaUnavailableError(RuntimeError):
    pass


def _cupy():
    try:
        return importlib.import_module("cupy")
    except (ImportError, OSError):
        return None


def _cuda_device_available(cp) -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def cuda_available() -> bool:
    cp = _cupy()
    return cp is not None and _cuda_device_available(cp)


def cuda_module():
    cp = _cupy()
    if cp is None:
        raise CudaUnavailableError("CuPy is not installed")
    if not _cuda_device_available(cp):
        raise CudaUnavailableError("CUDA GPU is not available")
    return cp


def _is_cupy_array(value: object) -> TypeGuard[_CupyArrayLike]:
    module = type(value).__module__
    return module == "cupy" or module.startswith("cupy.")


def _convert_runtime_arg(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return np.int32(value)
    if isinstance(value, (float, np.floating)):
        return np.float32(value)
    return value


def cuda_execution_required(
    runtime_args: tuple[object, ...], *, backend_name: str
) -> bool:
    array_args = [
        value
        for value in runtime_args
        if hasattr(value, "dtype") and hasattr(value, "flags")
    ]
    cupy_array_args = [value for value in array_args if _is_cupy_array(value)]

    if not cupy_array_args:
        return False

    if len(cupy_array_args) != len(array_args):
        raise TypeError(
            f"{backend_name} execution does not support mixed NumPy and CuPy arrays"
        )

    return True


def execute_cuda_if_needed(
    *,
    kernel_cache: CudaKernelCache,
    cuda_src: str,
    kernel_name: str,
    launch_grid: tuple[int, ...],
    threads_per_block: int,
    runtime_args: tuple[object, ...],
) -> None:
    # NumPy calls are compilation-only, including on CUDA machines.
    if not cuda_execution_required(runtime_args, backend_name="CUDA"):
        return

    cp = cuda_module()
    max_threads = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
    if threads_per_block > max_threads:
        raise ValueError(
            f"CUDA block size {threads_per_block} exceeds device limit {max_threads}"
        )

    cache_key = (cuda_src, kernel_name)
    if cache_key not in kernel_cache:
        kernel_cache[cache_key] = cp.RawKernel(
            cuda_src,
            kernel_name,
            options=("--std=c++14",),
        )

    kernel_cache[cache_key](
        launch_grid,
        (threads_per_block,),
        tuple(_convert_runtime_arg(value) for value in runtime_args),
    )


def cuda_chip() -> str:
    cp = cuda_module()
    return f"sm_{cp.cuda.Device().compute_capability}"


def _convert_mlir_memref_args(runtime_args: tuple[object, ...]) -> tuple[object, ...]:
    converted = []

    for value in runtime_args:
        if _is_cupy_array(value):
            array = value
            if array.ndim != 1:
                raise ValueError("MLIR MVP supports only 1D CuPy arrays")
            if not array.flags.c_contiguous:
                raise ValueError("MLIR MVP requires C-contiguous arrays")

            converted.extend(
                [
                    array,  # allocated ptr
                    array,  # aligned ptr
                    np.int64(0),  # offset
                    np.int64(array.shape[0]),
                    np.int64(1),
                ]
            )
        else:
            converted.append(_convert_runtime_arg(value))

    return tuple(converted)


def execute_mlir_cubin_if_needed(
    *,
    kernel_cache: CudaKernelCache,
    cubin: bytes,
    kernel_name: str,
    launch_grid: tuple[int, ...],
    threads_per_block: int,
    runtime_args: tuple[object, ...],
) -> None:
    # NumPy calls are compile-only, same behavior as CUDA backend.
    if not cuda_execution_required(runtime_args, backend_name="MLIR"):
        return

    cp = cuda_module()
    max_threads = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
    if threads_per_block > max_threads:
        raise ValueError(
            f"CUDA block size {threads_per_block} exceeds device limit {max_threads}"
        )

    cache_key = (cubin, kernel_name)
    if cache_key not in kernel_cache:
        module = cp.cuda.function.Module()
        module.load(cubin)
        kernel_cache[cache_key] = module.get_function(kernel_name)

    kernel_cache[cache_key](
        launch_grid,
        (threads_per_block,),
        _convert_mlir_memref_args(runtime_args),
    )
