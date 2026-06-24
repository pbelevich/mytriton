import importlib
from typing import Any

import numpy as np

CudaKernelCache = dict[tuple[str, str], Any]


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


def _is_cupy_array(value: object) -> bool:
    module = type(value).__module__
    return module == "cupy" or module.startswith("cupy.")


def _convert_runtime_arg(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return np.int32(value)
    if isinstance(value, (float, np.floating)):
        return np.float32(value)
    return value


def execute_cuda_if_needed(
    *,
    kernel_cache: CudaKernelCache,
    cuda_src: str,
    kernel_name: str,
    launch_grid: tuple[int, ...],
    threads_per_block: int,
    runtime_args: tuple[object, ...],
) -> None:
    array_args = [
        value
        for value in runtime_args
        if hasattr(value, "dtype") and hasattr(value, "flags")
    ]
    cupy_array_args = [value for value in array_args if _is_cupy_array(value)]

    # NumPy calls are compilation-only, including on CUDA machines.
    if not cupy_array_args:
        return
    if len(cupy_array_args) != len(array_args):
        raise TypeError("CUDA execution does not support mixed NumPy and CuPy arrays")

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
