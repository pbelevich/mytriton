from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

ArrayFramework = Literal["numpy", "cupy", "torch"]
ArrayDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class RuntimeArrayInfo:
    framework: ArrayFramework
    device: ArrayDevice
    device_index: int | None
    dtype_name: str
    c_contiguous: bool

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


def _module_matches(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _device_index(value: object) -> int | None:
    return value if type(value) is int else None


def array_arg_info(value: object) -> RuntimeArrayInfo | None:
    if isinstance(value, np.ndarray):
        return RuntimeArrayInfo(
            framework="numpy",
            device="cpu",
            device_index=None,
            dtype_name=str(value.dtype),
            c_contiguous=bool(value.flags.c_contiguous),
        )

    module = type(value).__module__

    if _module_matches(module, "cupy"):
        dtype = getattr(value, "dtype", None)
        flags = getattr(value, "flags", None)
        device = getattr(value, "device", None)

        if dtype is None or flags is None or device is None:
            return None

        return RuntimeArrayInfo(
            framework="cupy",
            device="cuda",
            device_index=_device_index(getattr(device, "id", None)),
            dtype_name=str(dtype),
            c_contiguous=bool(getattr(flags, "c_contiguous", False)),
        )

    if _module_matches(module, "torch"):
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        is_contiguous = getattr(value, "is_contiguous", None)

        if dtype is None or device is None or not callable(is_contiguous):
            return None

        device_type = getattr(device, "type", None)
        if device_type not in ("cpu", "cuda"):
            return None

        return RuntimeArrayInfo(
            framework="torch",
            device=cast(ArrayDevice, device_type),
            device_index=_device_index(getattr(device, "index", None)),
            dtype_name=str(dtype).removeprefix("torch."),
            c_contiguous=bool(is_contiguous()),
        )

    return None
