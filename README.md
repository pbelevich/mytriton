# mytriton

`mytriton` is a small symbolic tracer inspired by Triton's Python API.
It converts straight-line Python kernels into a simple expression-tree IR.

## Example

```python
import numpy as np

import mytriton as triton
import mytriton.language as tl


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x_values = tl.load(x + offsets, mask=mask, other=0.0)
    y_values = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, x_values + y_values, mask=mask)


n = 1_000
block = 256
x = np.ones(n, dtype=np.float32)
y = np.ones(n, dtype=np.float32)
out = np.empty_like(x)

ops = add_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
    x,
    y,
    out,
    n,
    BLOCK=block,
)
print(ops)
```

## Current limitations

- Kernels are traced into an expression-tree IR; they are not executed.
- No CPU or GPU code is generated.
- Only straight-line kernels are supported. Symbolic Python control flow is rejected.
- Runtime array arguments must be C-contiguous `float32` arrays.
- The launch grid is evaluated and validated, but is not represented in the IR.
- The IR is not yet typed SSA and has no optimization passes.

## Development

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

Format the project and apply safe lint fixes:

```bash
make format
```

Run the linter, formatter check, type checker, and tests:

```bash
make check
```

To enable checks before every commit, run:

```bash
pre-commit install
```
