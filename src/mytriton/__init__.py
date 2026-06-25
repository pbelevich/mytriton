from .compiler import jit
from .cuda_utils import cuda_available


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def next_power_of_2(value):
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


__all__ = ["cdiv", "cuda_available", "jit", "next_power_of_2"]
