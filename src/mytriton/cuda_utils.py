import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, TypeGuard

import numpy as np

from .runtime_args import array_arg_info

CudaKernelCache = dict[tuple[object, str, int], Any]


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


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise CudaUnavailableError(
            "PyTorch is required for Torch CUDA tensor execution"
        ) from error


def _is_cupy_array(value: object) -> TypeGuard[_CupyArrayLike]:
    module = type(value).__module__
    return module == "cupy" or module.startswith("cupy.")


def _convert_runtime_arg(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return np.int32(value)
    if isinstance(value, (float, np.floating)):
        return np.float32(value)
    return value


def _normalize_cuda_array_args(
    cp,
    runtime_args: tuple[object, ...],
) -> tuple[object, ...]:
    normalized = []

    for value in runtime_args:
        info = array_arg_info(value)

        if info is None:
            normalized.append(value)
            continue

        if not info.is_cuda:
            raise TypeError(f"cannot use {info.framework} CPU array as a CUDA argument")

        if info.framework == "cupy":
            normalized.append(value)
            continue

        if info.framework == "torch":
            detach = getattr(value, "detach", None)
            if not callable(detach):
                raise TypeError("Torch CUDA array does not support detach()")

            # A raw kernel launch is not an autograd operation. Detaching makes
            # the tensor exportable through DLPack while preserving its storage.
            normalized.append(cp.from_dlpack(detach()))
            continue

        raise TypeError(f"unsupported CUDA array framework: {info.framework}")

    return tuple(normalized)


def cuda_execution_required(
    runtime_args: tuple[object, ...], *, backend_name: str
) -> bool:
    array_infos = [
        info for value in runtime_args if (info := array_arg_info(value)) is not None
    ]

    if not array_infos:
        return False

    cuda_infos = [info for info in array_infos if info.is_cuda]

    if not cuda_infos:
        return False

    if len(cuda_infos) != len(array_infos):
        raise TypeError(
            f"{backend_name} execution does not support mixed CPU and CUDA arrays"
        )

    frameworks = {info.framework for info in cuda_infos}
    if len(frameworks) != 1:
        rendered = ", ".join(sorted(frameworks))
        raise TypeError(
            f"{backend_name} execution does not support mixed CUDA "
            f"array frameworks: {rendered}"
        )

    device_indices = {info.device_index for info in cuda_infos}
    if len(device_indices) != 1:
        rendered = ", ".join(
            "unknown" if index is None else str(index)
            for index in sorted(
                device_indices,
                key=lambda index: -1 if index is None else index,
            )
        )
        raise TypeError(
            f"{backend_name} execution requires one CUDA device, got: {rendered}"
        )

    return True


@contextmanager
def _cuda_launch_context(
    cp,
    runtime_args: tuple[object, ...],
) -> Iterator[None]:
    cuda_infos = [
        info
        for value in runtime_args
        if ((info := array_arg_info(value)) is not None and info.is_cuda)
    ]

    if not cuda_infos:
        raise RuntimeError("CUDA launch requires at least one CUDA array")

    launch_info = cuda_infos[0]
    device_index = launch_info.device_index

    if device_index is None:
        raise TypeError("CUDA array device index is unavailable")

    with cp.cuda.Device(device_index):
        if launch_info.framework == "torch":
            torch = _torch_module()
            torch_stream = torch.cuda.current_stream(
                device=device_index,
            )

            with cp.cuda.Stream.from_external(torch_stream):
                yield
        else:
            yield


def execute_cuda_if_needed(
    *,
    kernel_cache: CudaKernelCache,
    cuda_src: str,
    kernel_name: str,
    launch_grid: tuple[int, ...],
    threads_per_block: int,
    runtime_args: tuple[object, ...],
) -> None:
    # CPU arrays are compilation-only, including on CUDA machines.
    if not cuda_execution_required(runtime_args, backend_name="CUDA"):
        return

    cp = cuda_module()

    with _cuda_launch_context(cp, runtime_args):
        max_threads = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
        if threads_per_block > max_threads:
            raise ValueError(
                f"CUDA block size {threads_per_block} "
                f"exceeds device limit {max_threads}"
            )

        cache_key = (
            cuda_src,
            kernel_name,
            cp.cuda.Device().id,
        )
        if cache_key not in kernel_cache:
            kernel_cache[cache_key] = cp.RawKernel(
                cuda_src,
                kernel_name,
                options=("--std=c++14",),
            )

        normalized_args = _normalize_cuda_array_args(
            cp,
            runtime_args,
        )

        kernel_cache[cache_key](
            launch_grid,
            (threads_per_block,),
            tuple(_convert_runtime_arg(value) for value in normalized_args),
        )


def cuda_chip(runtime_args: tuple[object, ...] = ()) -> str:
    cuda_infos = [
        info
        for value in runtime_args
        if ((info := array_arg_info(value)) is not None and info.is_cuda)
    ]

    device_index = None
    if cuda_infos:
        cuda_execution_required(runtime_args, backend_name="MLIR")
        device_index = cuda_infos[0].device_index

        if device_index is None:
            raise TypeError("CUDA array device index is unavailable")

    cp = cuda_module()

    if device_index is None:
        return f"sm_{cp.cuda.Device().compute_capability}"

    with cp.cuda.Device(device_index):
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
    # CPU arrays are compilation-only, same behavior as the CUDA backend.
    if not cuda_execution_required(runtime_args, backend_name="MLIR"):
        return

    cp = cuda_module()

    with _cuda_launch_context(cp, runtime_args):
        max_threads = cp.cuda.Device().attributes["MaxThreadsPerBlock"]
        if threads_per_block > max_threads:
            raise ValueError(
                f"CUDA block size {threads_per_block} "
                f"exceeds device limit {max_threads}"
            )

        cache_key = (
            cubin,
            kernel_name,
            cp.cuda.Device().id,
        )
        if cache_key not in kernel_cache:
            module = cp.cuda.function.Module()
            module.load(cubin)
            kernel_cache[cache_key] = module.get_function(kernel_name)

        normalized_args = _normalize_cuda_array_args(
            cp,
            runtime_args,
        )

        kernel_cache[cache_key](
            launch_grid,
            (threads_per_block,),
            _convert_mlir_memref_args(normalized_args),
        )
