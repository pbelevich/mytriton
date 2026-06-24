# mytriton

`mytriton` is a small symbolic tracer inspired by Triton's Python API.
It traces straight-line Python kernels into an expression-tree IR, infers types,
and lowers the result into a small SSA-style IR.

## Example

```python
import numpy as np

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


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

expression_ops, ssa_ops = add_kernel[
    lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
](
    x,
    y,
    out,
    n,
    BLOCK=block,
)

print(expression_ops)
print(SSAPrinter().print_ops(ssa_ops))
```

The first result contains the captured expression-tree operations. The second
contains typed SSA operations. Shared expressions such as `offsets` and `mask`
are lowered once and referenced by their SSA values wherever they are reused.

For example, part of the resulting SSA looks like this:

```text
%2 = arange {start=0, end=256} : vector<256 x i32>
%3 = add %1, %2 : vector<256 x i32>
%4 = addptr x, %3 : vector<256 x ptr<f32>>
%5 = cmp_lt %3, n : vector<256 x bool>
%6 = load %4, %5, 0.0 : vector<256 x f32>
```

## Current limitations

- Kernels are traced and lowered to typed SSA, but they are not executed.
- No CPU or GPU code is generated.
- Only straight-line kernels are supported. Symbolic Python control flow is rejected.
- Runtime array arguments must be C-contiguous `float32` arrays.
- The launch grid is evaluated and validated, but is not represented in the IR.
- The SSA IR has no basic blocks, control-flow representation, or phi nodes yet.
- There are no optimization passes yet.

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
