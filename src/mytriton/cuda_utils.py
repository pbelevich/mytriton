import importlib
from typing import Any, cast

import numpy as np

CudaKernelCache = dict[tuple[object, str], Any]


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


def cuda_chip() -> str:
    cp = cuda_module()
    return f"sm_{cp.cuda.Device().compute_capability}"


def _is_cupy_array(value: object) -> bool:
    module = type(value).__module__
    return module == "cupy" or module.startswith("cupy.")


def _convert_runtime_arg(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return np.int32(value)
    if isinstance(value, (float, np.floating)):
        return np.float32(value)
    return value


def _convert_memref_runtime_args(
    runtime_args: tuple[object, ...],
) -> tuple[object, ...]:
    converted: list[object] = []

    for value in runtime_args:
        if _is_cupy_array(value):
            array = cast(Any, value)
            if array.ndim != 1:
                raise ValueError("MLIR CUDA execution currently supports 1D arrays")
            if not array.flags.c_contiguous:
                raise ValueError("MLIR CUDA execution requires C-contiguous arrays")
            converted.extend(
                [
                    array,
                    array,
                    np.int64(0),
                    np.int64(array.shape[0]),
                    np.int64(1),
                ]
            )
        else:
            converted.append(_convert_runtime_arg(value))

    return tuple(converted)


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


def execute_mlir_cuda_binary_if_needed(
    *,
    kernel_cache: CudaKernelCache,
    cubin: bytes,
    kernel_name: str,
    launch_grid: tuple[int, ...],
    threads_per_block: int,
    runtime_args: tuple[object, ...],
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
            "MLIR CUDA execution does not support mixed NumPy and CuPy arrays"
        )

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
        _convert_memref_runtime_args(runtime_args),
    )
    return True
