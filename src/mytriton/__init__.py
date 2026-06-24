from .compiler import jit
from .cuda_utils import cuda_available


def cdiv(x, y):
    return (x + y - 1) // y


__all__ = ["cdiv", "cuda_available", "jit"]
