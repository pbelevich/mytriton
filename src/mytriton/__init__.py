from .compiler import jit


def cdiv(x, y):
    return (x + y - 1) // y


__all__ = ["cdiv", "jit"]
