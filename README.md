# mytriton

`mytriton` is a small compiler inspired by Triton's Python API. It parses a
supported subset of Python kernel source with an AST frontend, builds a symbolic
expression-tree IR, infers types, lowers the result into a small SSA-style IR,
verifies and optimizes that IR, and emits backend source. The default backend
emits CUDA C++ for rank-1 vectors and small rank-2 tiles; an experimental MLIR
backend can lower a small subset of rank-1 kernels through MLIR's GPU/NVVM stack
to a cubin.

## Versions

- [ver1](https://github.com/pbelevich/mytriton/tree/ver1): symbolic tracing,
  Triton-like kernel launch syntax, tests, and CI.
- [ver2](https://github.com/pbelevich/mytriton/tree/ver2): typed SSA lowering
  and type inference for the traced expression-tree IR.
- [ver3](https://github.com/pbelevich/mytriton/tree/ver3): CUDA C++ source
  generation, CuPy-backed compilation, and optional CUDA execution.
- [ver4](https://github.com/pbelevich/mytriton/tree/ver4): math operations and
  activation kernels, including negation, `tl.exp`, `tl.minimum`,
  `tl.maximum`, `tl.where`, ReLU, leaky ReLU, and sigmoid.
- [ver5](https://github.com/pbelevich/mytriton/tree/ver5): SSA verifier and
  optimization pipeline with constant folding, common subexpression
  elimination, and dead-code elimination.
- [ver6](https://github.com/pbelevich/mytriton/tree/ver6): row-wise reductions,
  `tl.sum`/`tl.max`/`tl.min`, 2D matrix add, softmax, `tl.static_range`,
  long-row sum, and a first naive matrix multiplication kernel.
- [ver7](https://github.com/pbelevich/mytriton/tree/ver7): an experimental
  MLIR backend for 1D elementwise kernels, backend-parametrized tests, MLIR GPU
  dialect emission, lowering to cubin, and CuPy-backed cubin execution.
- [ver8](https://github.com/pbelevich/mytriton/tree/ver8): rank-2 block shapes,
  `x[:, None]`/`x[None, :]` expansion, broadcasted 2D masks, CUDA lowering for
  tiled kernels, and a simple rank-2 tiled matrix multiplication kernel.
- [ver9](https://github.com/pbelevich/mytriton/tree/ver9): an AST-based Python
  frontend that replaces direct execution of kernels with symbolic arguments,
  resolves runtime and `constexpr` names, handles the Python syntax used by the
  existing kernels, and unrolls compile-time `range`/`tl.static_range` loops.

## AST frontend

On a JIT cache miss, `mytriton` obtains the decorated function's source with
`inspect`, parses it with Python's `ast` module, and visits the function body.
The frontend does not call the kernel as a regular Python function during
tracing. Instead, it creates an environment in which runtime scalar and pointer
parameters are symbolic values while `tl.constexpr` parameters retain their
concrete Python values.

The AST frontend supports the syntax used by the current kernels: expression
statements, simple and annotated assignments, augmented arithmetic assignments,
function calls, tuples and lists, arithmetic and Boolean `&`, unary signs,
simple `<` and `is` comparisons, constexpr conditional expressions, and the
subscripts needed for `x[:, None]` and `x[None, :]`. Names from globals and
closures are resolved alongside Python builtins, so `tl`, `range`, and helper
functions referenced by a kernel remain available while its AST is visited.

Both Python `range` and `tl.static_range` are accepted when their start, stop,
and step are compile-time integers. Their bodies are expanded by the frontend,
so no loop reaches the expression-tree or SSA IR in this version. For example:

```python
accumulator = 0.0

for k in tl.static_range(0, K):
    accumulator += tl.load(a + k) * tl.load(b + k)
```

Here `K` must be a `tl.constexpr` parameter. Unsupported syntax is rejected with
an `ASTFrontendError` instead of being accidentally evaluated by the Python
interpreter. This explicit source representation also provides the foundation
for adding runtime loops to the IR in a later version.

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

expression_ops, ssa_ops, src = add_kernel[
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
print(src)
```

The first result contains the expression-tree operations built by the AST
frontend. The second contains optimized typed SSA operations, and the third
contains generated source for the selected backend. The default backend is
CUDA, so `src` is CUDA C++. With NumPy arguments, compilation stops there. If
the arguments are CuPy arrays and a CUDA GPU is available, the generated kernel
is also compiled and launched. Shared expressions such as `offsets` and `mask`
are lowered once and referenced by their SSA values wherever they are reused.

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

Rank-2 tiles are expressed by expanding rank-1 ranges:

```python
@triton.jit
def matrix_add_2d_kernel(x, y, out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    lhs = tl.load(x + offsets, mask=mask, other=0.0)
    rhs = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, lhs + rhs, mask=mask)
```

The SSA keeps the tile shape explicit:

```text
%3 = expand_dims %2 {axis=1} : block<16x1 x i32>
%9 = expand_dims %8 {axis=0} : block<1x32 x i32>
%11 = add %5, %10 : block<16x32 x i32>
%15 = and %13, %14 : block<16x32 x bool>
%16 = load %12, %15, 0.0 : block<16x32 x f32>
```

The CUDA backend maps the tile onto one linear CUDA thread block:

```cuda
int tile_i = threadIdx.x / 32;
int tile_j = threadIdx.x % 32;
```

The backend can be selected with `MYTRITON_BACKEND` when running your own
script:

```bash
MYTRITON_BACKEND=cuda python examples_or_your_script.py
MYTRITON_BACKEND=mlir python examples_or_your_script.py
```

The `add_kernel` and `copy_kernel` tests are parameterized over both backends,
so they exercise CUDA and MLIR from the same test body.

With `MYTRITON_BACKEND=mlir`, the same optimized SSA is emitted as MLIR GPU
dialect instead of CUDA C++:

```mlir
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @add_kernel(%x: memref<?xf32>, %y: memref<?xf32>, %out: memref<?xf32>, %n: i32) kernel {
      %bid_x = gpu.block_id x
      %tid_x = gpu.thread_id x
      %block_id_x = arith.index_cast %bid_x : index to i32
      %thread_id_x = arith.index_cast %tid_x : index to i32
      ...
      gpu.return
    }
  }
}
```

For NumPy arguments, the MLIR backend stops after source generation, so MLIR
Python bindings are not required just to inspect the emitted MLIR. For CuPy
arguments, the backend runs a small pass pipeline that attaches an NVVM target,
converts GPU operations to NVVM, emits a GPU binary, extracts the cubin, loads
it through CuPy, and launches it with the same grid and thread-block size used
by the CUDA backend. CuPy arrays are passed using the ranked-memref ABI:
allocated pointer, aligned pointer, offset, size, and stride.

The test kernels also include a copy, 2D matrix add, ReLU through
`tl.maximum`, leaky ReLU through `tl.where`, sigmoid through negation,
`tl.exp`, addition, and division, row-wise `tl.sum`/`tl.max`/`tl.min`
reductions, a numerically stable row-wise softmax, and a long-row sum that uses
`tl.static_range` to unroll several block-sized loads at compile time. The
current tests also include matrix multiplication kernels: an older naive
rank-1-vector version and a rank-2 tiled version that combines a 2D launch grid,
2D block broadcasting, `tl.static_range` over `K`, and masked tile stores.

Before CUDA code generation, the SSA IR is checked by a verifier. The verifier
validates definition order, result declarations, operand types, broadcast
shapes, pointer operations, memory masks, and operation-specific rules such as
`tl.exp` requiring `f32`, `tl.where` lowering to a Boolean `select`,
`expand_dims` preserving element types while inserting a size-1 dimension, and
reductions consuming one power-of-two rank-1 block whose width matches the CUDA
block size.

The verified SSA then runs through a small optimization pipeline:

- constant folding and local simplifications such as `select(true, x, y) -> x`;
- common subexpression elimination for pure operations;
- dead-code elimination.

The verifier runs after every optimization pass so malformed rewrites fail
before CUDA code generation.

## Current limitations

- Generated backend source is returned as a string. Execution requires CuPy
  built for the installed CUDA version and an available CUDA GPU; NumPy inputs
  remain compilation-only.
- `MYTRITON_BACKEND` can be `cuda` or `mlir`. The CUDA backend is the default
  and supports the full current mytriton test language. The MLIR backend is an
  experimental MVP for 1D elementwise kernels. MLIR source generation does not
  require MLIR Python bindings, but MLIR cubin execution does.
- Kernel functions must have source available to `inspect.getsource`; functions
  created dynamically or entered only in an interactive session may not be
  recoverable by the AST frontend.
- Compile-time `range` and `tl.static_range` loops are unrolled by the AST
  frontend. Runtime range bounds, `if`/`while`, `break`/`continue`, `for/else`,
  non-simple assignment targets, and other symbolic Python control flow are not
  supported.
- Runtime array arguments must be C-contiguous `float32` arrays.
- The launch grid is evaluated and used for CUDA execution, but it is not
  represented in the IR.
- CUDA execution uses the SSA rank-1 vector width, or the product of the SSA
  rank-2 tile shape, as the number of threads per block. Scalar-only kernels
  use one thread per block.
- JIT cache entries are specialized by runtime types and constexpr values. Python
  globals and closure values used by a kernel must remain unchanged; call
  `kernel.clear_cache()` after changing them.
- CUDA lowering currently supports program IDs, `tl.arange`, basic arithmetic and
  comparison, Boolean `&`, rank-2 `expand_dims` via `x[:, None]` and
  `x[None, :]`, elementwise minimum and maximum, negation, `tl.exp`,
  `tl.where`, pointer addition, masked loads, masked stores, block-local
  `tl.sum`/`tl.max`/`tl.min` reductions, and compile-time `tl.static_range`
  loops. Reduction lowering internally emits the CUDA shared-memory scratch
  buffers and synchronization needed for block-local reductions. Floating-point
  elementwise extrema propagate NaNs and choose the right-hand operand when
  values compare equal.
- Reductions are currently single-block reductions over the SSA vector width.
  The vector width must be a power of two and must match the CUDA thread block
  size. Larger rows can be handled by statically unrolling multiple loads into
  one block-local partial vector, as in the long-row sum test, but there is no
  multi-block reduction yet.
- Matrix multiplication support is intentionally naive so far. The current
  rank-2 matmul kernel computes one output tile with one CUDA thread per output
  element, but it does not tile through shared memory because there is no
  user-facing shared-memory API yet. It repeatedly reads from global memory and
  uses `tl.static_range` to unroll a constexpr reduction dimension.
- MLIR lowering currently supports only `ptr<f32>` parameters as
  `memref<?xf32>`, scalar `i32`/`f32`/`bool`, `tl.program_id(0)`,
  `tl.arange(0, BLOCK)`, basic arithmetic and `<`, pointer addition, masked
  loads, and masked stores. It intentionally rejects nonzero program axes,
  nonzero `arange` starts, and rank-2 block shapes instead of silently
  generating wrong code. It does not yet support 2D program IDs, reductions,
  `expand_dims`, Boolean `&`, `tl.maximum`, `tl.minimum`, `tl.where`, negation,
  `tl.exp`, `tl.static_range`, or matrix multiplication.
- MLIR execution currently supports only 1D C-contiguous CuPy arrays because it
  builds one-dimensional memref descriptors.
- The SSA IR has no basic blocks, control-flow representation, or phi nodes yet.
- The optimizer is intentionally small. It does local simplification, constant
  folding, common subexpression elimination, and dead-code elimination, but it
  has no control-flow or memory-aware optimization passes yet.

## Development

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

To enable CUDA execution with CUDA 12, install the matching CuPy wheel:

```bash
python -m pip install -e ".[cuda12]"
```

MLIR cubin execution requires Python bindings importable as `mlir.ir` and
`mlir.passmanager`, plus an MLIR build that includes the GPU/NVVM passes needed
by `gpu-module-to-binary`. These bindings are intentionally not listed as a
default or development dependency because MLIR Python packaging depends on the
LLVM/MLIR build or wheel you use.

GitHub Actions runs linting, type checks, unit tests, and CUDA/MLIR codegen
tests, but excludes GPU execution tests:

```bash
python -m pytest -m "not execution"
```

On a GPU machine, run execution tests locally:

```bash
MYTRITON_REQUIRE_CUDA=1 python -m pytest
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
