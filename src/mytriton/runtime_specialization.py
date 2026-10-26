from dataclasses import dataclass

import numpy as np

from .runtime_args import array_arg_info

MAX_POINTER_ALIGNMENT = 16
MAX_INTEGER_DIVISIBILITY = 128


@dataclass(frozen=True)
class RuntimeSpecialization:
    """Properties of one runtime argument that are safe to compile against."""

    alignment: int | None = None
    divisibility: int | None = None


def _capped_power_of_two_factor(value: int, *, maximum: int) -> int:
    if value == 0:
        return maximum

    return min(abs(value) & -abs(value), maximum)


def _array_data_pointer(value: object) -> int | None:
    if isinstance(value, np.ndarray):
        return int(value.ctypes.data)

    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        pointer = data_ptr()
        return int(pointer) if isinstance(pointer, int) else None

    cuda_interface = getattr(value, "__cuda_array_interface__", None)
    if isinstance(cuda_interface, dict):
        data = cuda_interface.get("data")
        if isinstance(data, tuple) and data and isinstance(data[0], int):
            return data[0]

    return None


def runtime_specialization(value: object) -> RuntimeSpecialization:
    if array_arg_info(value) is not None:
        pointer = _array_data_pointer(value)
        alignment = (
            None
            if pointer is None
            else _capped_power_of_two_factor(
                pointer,
                maximum=MAX_POINTER_ALIGNMENT,
            )
        )
        return RuntimeSpecialization(alignment=alignment)

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return RuntimeSpecialization(
            divisibility=_capped_power_of_two_factor(
                int(value),
                maximum=MAX_INTEGER_DIVISIBILITY,
            )
        )

    return RuntimeSpecialization()
