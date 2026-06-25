# mytriton

`mytriton` is a small symbolic tracer inspired by Triton's Python API.
It traces straight-line Python kernels into an expression-tree IR, infers types,
lowers the result into a small SSA-style IR, and emits CUDA C++ source.

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

expression_ops, ssa_ops, cuda_src = add_kernel[
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
print(cuda_src)
```

The first result contains the captured expression-tree operations. The second
contains typed SSA operations, and the third contains generated CUDA C++ source.
With NumPy arguments, compilation stops there. If the arguments are CuPy arrays
and a CUDA GPU is available, the generated kernel is also compiled and launched.
Shared expressions such as `offsets` and `mask` are lowered once and referenced
by their SSA values wherever they are reused.

For example, part of the resulting SSA looks like this:

```text
%2 = arange {start=0, end=256} : vector<256 x i32>
%3 = add %1, %2 : vector<256 x i32>
%4 = addptr x, %3 : vector<256 x ptr<f32>>
%5 = cmp_lt %3, n : vector<256 x bool>
%6 = load %4, %5, 0.0 : vector<256 x f32>
```

The corresponding CUDA represents each distributed vector element as one value
per CUDA thread. Pointer arithmetic is folded into array indexing:

```cuda
extern "C" __global__
void add_kernel(float* x, float* y, float* out, int n) {
    int v0 = blockIdx.x;
    int v1 = (v0 * 256);
    int v2 = threadIdx.x;
    int v3 = (v1 + v2);
    bool v5 = (v3 < n);
    float v6 = (v5 ? x[v3] : 0.0f);
    float v8 = (v5 ? y[v3] : 0.0f);
    float v9 = (v6 + v8);
    if (v5) {
        out[v3] = v9;
    }
}
```

The test kernels also include a copy, ReLU through `tl.maximum`, leaky ReLU
through `tl.where`, and sigmoid through negation, `tl.exp`, addition, and
division.

## Current limitations

- Generated CUDA source is returned as a string. Execution requires CuPy built
  for the installed CUDA version and an available CUDA GPU; NumPy inputs remain
  compilation-only.
- Only straight-line kernels are supported. Symbolic Python control flow is rejected.
- Runtime array arguments must be C-contiguous `float32` arrays.
- The launch grid is evaluated and used for CUDA execution, but it is not
  represented in the IR.
- CUDA execution uses the SSA vector width as the number of threads per block;
  scalar-only kernels use one thread per block.
- JIT cache entries are specialized by runtime types and constexpr values. Python
  globals and closure values used by a kernel must remain unchanged; call
  `kernel.clear_cache()` after changing them.
- CUDA lowering currently supports program IDs, ranges, basic arithmetic and
  comparison, minimum and maximum, negation, `tl.exp`, `tl.where`, pointer
  addition, masked loads, and masked stores. Floating-point extrema propagate
  NaNs and choose the right-hand operand when values compare equal.
- The SSA IR has no basic blocks, control-flow representation, or phi nodes yet.
- There are no optimization passes yet.

## Development

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

To enable CUDA execution with CUDA 12, install the matching CuPy wheel:

```bash
python -m pip install -e ".[cuda12]"
```

On a GPU test runner, require all CUDA execution tests to run instead of skip:

```bash
MYTRITON_REQUIRE_CUDA=1 python -m pytest
```

The GitHub Actions CUDA job is enabled when the repository variable
`CUDA_RUNNER_ENABLED` is set to `true` and a self-hosted runner has the `gpu`
label.

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
