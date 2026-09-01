# mytriton

`mytriton` is a small compiler inspired by Triton's Python API. It parses a
supported subset of Python kernel source with an AST frontend, builds a symbolic
expression-tree IR, infers types, lowers the result into a small SSA-style IR,
verifies the IR, runs the available optimizations, and emits backend source. The
default backend emits CUDA C++ for rank-1 vectors and small rank-2 tiles; an
experimental MLIR backend can lower a small subset of rank-1 kernels through
MLIR's GPU/NVVM stack to a cubin.

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
- [ver10](https://github.com/pbelevich/mytriton/tree/ver10): structured runtime
  `range` loops with captured outer values, loop-carried variables,
  `iter_args`/`yield` SSA semantics, nested-loop verification, and CUDA `for`
  generation.
- [ver11](https://github.com/pbelevich/mytriton/tree/ver11): `tl.empty`,
  `tl.full`, and `tl.zeros` block factory functions, public `tl.int1`,
  `tl.int32`, and `tl.float32` dtype objects, typed SSA lowering and
  verification for constructed blocks, CUDA backend support, and MLIR lowering
  for `tl.full`/`tl.zeros`.
- [ver12](https://github.com/pbelevich/mytriton/tree/ver12): explicit CUDA tile
  layouts that separate logical output shapes from physical thread
  organization, store-rooted output-layout inference, reduction-aware thread
  layouts, projected layouts for per-thread values, and cooperative layouts
  for distributing arbitrary rank-2 tiles across a CUDA thread block.
- [ver13](https://github.com/pbelevich/mytriton/tree/ver13): public `tl.dot`
  semantics for rank-2 `f32` blocks, expression-tree and typed SSA operations,
  `[M, K] x [K, N] -> [M, N]` type inference, independent SSA verification,
  optimizer purity rules, and an explicit diagnostic for the not-yet-implemented
  CUDA lowering.
- [ver14](https://github.com/pbelevich/mytriton/tree/ver14): CUDA shared-memory
  tile buffers, SSA pattern matching for canonical masked matrix loads,
  cooperative staging of `tl.dot` operands, zero-filled boundary handling,
  block synchronization, and an explicit diagnostic for the deferred
  CUDA-core dot computation.
- [ver15](https://github.com/pbelevich/mytriton/tree/ver15): working
  CUDA-core lowering for canonical `tl.dot` matrix tiles, one register
  accumulator per output thread, an FMA loop over shared-memory operands,
  synchronization before tile reuse, runtime traversal of multiple K-tiles,
  and CUDA correctness tests for masked edge tiles.
- [ver16](https://github.com/pbelevich/mytriton/tree/ver16): per-thread CUDA
  register tiles for `tl.dot` outputs, explicit logical-output-to-thread/register
  mapping, broadcast-aware register arithmetic and pointer construction,
  register-valued loop-carried accumulators, masked multi-result stores, and
  CUDA execution tests in which each thread computes several C elements.
- [ver17](https://github.com/pbelevich/mytriton/tree/ver17): optional PyTorch
  tensor arguments, framework-independent runtime array metadata, zero-copy
  DLPack conversion for CUDA tensors, same-device validation, execution on the
  current PyTorch CUDA stream, and Torch-backed CUDA and MLIR execution tests.

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

When all bounds are compile-time integers, both Python `range` and
`tl.static_range` are expanded by the frontend. For example:

```python
accumulator = 0.0

for k in tl.static_range(0, K):
    accumulator += tl.load(a + k) * tl.load(b + k)
```

Here `K` must be a `tl.constexpr` parameter, so no loop reaches the
expression-tree or SSA IR. Unsupported syntax is rejected with an
`ASTFrontendError` instead of being accidentally evaluated by the Python
interpreter.

## Runtime `for` loops

When a Python `range` has a symbolic start or stop, the AST frontend builds a
structured `ForRange` operation instead of unrolling the body. Runtime loop
bounds must lower to scalar `i32` values, and the step must be a positive
compile-time integer. Sequential and nested runtime loops are supported.

Variables that existed before the loop and are assigned in its body become
loop-carried values. Values from the surrounding scope that are only read by
the body are recorded as captures and lowered before entering the loop region.
Names created only inside the loop do not escape it. For example:

```python
@triton.jit
def runtime_sum_kernel(x, out, K):
    accumulator = 0.0

    for k in range(K):
        accumulator += tl.load(x + k)

    tl.store(out, accumulator)
```

The corresponding SSA uses an induction variable, a region argument initialized
from the value before the loop, and `yield` to carry the updated value into the
next iteration:

```text
%5 = for %0 in range(0, K, 1) iter_args(%1 = 0.0) : f32 {
  %2 = addptr x, %0 : ptr<f32>
  %3 = load %2, none, none : f32
  %4 = add %1, %3 : f32
  yield %4
}
store out, %5, none
```

Here `%1` denotes the accumulator at the start of the current iteration,
`yield %4` supplies its value for the next iteration, and `%5` is the value
available after the loop. CUDA lowering turns this region into a normal C++
`for` loop while preserving the same carried-value semantics.

## Block factory functions

Blocks with a known shape and element type can be constructed without deriving
their shape from another expression:

```python
accumulator = tl.zeros((BM, BN), tl.float32)
twos = tl.full([BM, BN], 2.0, tl.float32)
temporary = tl.empty(BLOCK, tl.float32)
```

The shape may be a positive integer or a non-empty tuple/list of positive
integers. The supported public dtype objects are `tl.int1`, `tl.int32`, and
`tl.float32`. `tl.full` accepts a scalar Boolean, integer, floating-point, or
symbolic runtime value and converts numeric values to the requested numeric
dtype. A block value cannot be used as the fill value.

The constructors remain explicit in SSA, including their normalized shape and
dtype:

```text
%0 = zeros {shape=(8,), dtype=f32} : vector<8 x f32>
%1 = full 2.5 {shape=(8,), dtype=f32} : vector<8 x f32>
%2 = add %0, %1 : vector<8 x f32>
```

In the ordinary elementwise CUDA execution model, each element of a distributed
block is represented by one scalar in its CUDA thread. Consequently `tl.zeros`
emits a zero-initialized per-thread value, `tl.full` emits the fill value in each
thread, and `tl.empty` declares an uninitialized per-thread value. A `tl.dot`
kernel may instead distribute a logical output tile across several registers
per thread; a constructed scalar value such as the initial `tl.zeros`
accumulator is then used to initialize every owned output register. These
factory functions describe logical blocks; they do not allocate CUDA shared
memory. Any computation that consumes a value produced by `tl.empty` observes
undefined contents.

## CUDA tile layouts

Logical block shapes are separate from their physical CUDA execution layouts.
A `BlockType` describes the shape and element type visible in SSA, while
`CudaKernelLayout` records both the logical output tile and the organization of
CUDA threads assigned to it:

```python
layout = CudaKernelLayout(
    output_tile_shape=(64, 64),
    thread_shape=(8, 32),
)
```

This layout represents a 64-by-64 output tile executed by 256 CUDA threads.
Ordinary elementwise lowering still uses one output element per thread, so its
automatically inferred thread shape equals the output tile shape. Reductions
use the separation to retain the wider thread shape required by a scalar
output. `tl.dot` kernels can now use a smaller physical thread shape and assign
several logical output elements to registers owned by each thread.

`CudaRegisterTileLayout` describes that assignment. Its register shape is the
elementwise quotient of the logical output shape and thread shape. Logical
coordinates use a strided mapping:

```text
logical_coordinate =
    thread_coordinate + register_coordinate * thread_shape
```

For an `(8, 8)` output tile executed by `(4, 8)` threads, the register shape is
`(2, 1)`: every thread computes two C elements whose row coordinates differ by
four. A `(16, 16)` output executed by `(4, 8)` threads gives a `(4, 2)` register
tile, or eight results per thread. Logical dimensions must be divisible by
their physical thread dimensions.

A projected `CudaTileLayout` maps logical dimensions directly to CUDA thread
dimensions. Singleton dimensions may be broadcast:

```text
logical (4, 8) -> thread axes (0, 1)
logical (4, 1) -> thread axes (0, none)
logical (1, 8) -> thread axes (none, 1)
```

A `CudaCooperativeTileLayout` describes a different mapping in which all
threads collectively traverse a logical tile. Thread `t` processes linear
indices:

```text
t
t + threads_per_block
t + 2 * threads_per_block
...
```

The linear indices are converted to logical coordinates according to the
layout order. For a row-major rank-2 tile the order is `(1, 0)`, meaning that
the column dimension changes fastest. This cooperative mapping can represent
matrix multiplication operands such as A `[BM, BK]` and B `[BK, BN]` even when
their shapes do not match the output tile or CUDA thread shape.

Version 12 introduces the layout model and its validation. It does not yet emit
cooperative shared-memory loads; those are the next CUDA lowering stage.

## `tl.dot` semantics

Rank-2 `f32` blocks can be combined with the public `tl.dot` operation:

```python
lhs = tl.zeros((BM, BK), tl.float32)
rhs = tl.zeros((BK, BN), tl.float32)
result = tl.dot(lhs, rhs)
```

The operands must have shapes `[M, K]` and `[K, N]`. Their inner dimensions
must match, and the result has shape `[M, N]`:

```text
%0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
%1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
%2 = dot %0, %1 : block<4x8 x f32>
```

The expression-tree type inference and SSA verifier independently check operand
rank, `f32` element types, matching reduction dimensions, and the exact result
type. `dot` is a pure SSA operation, so duplicate operations are eligible for
common subexpression elimination and unused operations can be removed by
dead-code elimination.

Version 13 defines the language and IR semantics only. Version 14 adds
cooperative shared-memory staging for canonical matrix loads, and Version 15
lowers the staged operands to an ordinary CUDA-core multiply-accumulate loop.

## Shared-memory CUDA-core dot

The CUDA backend recognizes canonical matrix tiles loaded for `tl.dot`:

```python
a_values = tl.load(
    a + a_rows * K + a_columns,
    mask=(a_rows < M) & (a_columns < K),
    other=0.0,
)
b_values = tl.load(
    b + b_rows * N + b_columns,
    mask=(b_rows < K) & (b_columns < N),
    other=0.0,
)
result = tl.dot(a_values, b_values)
```

The supported pointer form is `base + rows * row_stride + columns`. Rows and
columns must be built from a scalar tile offset plus an expanded
`tl.arange(0, size)`. Each load must use a two-dimensional bounds mask and
`other=0.0`.

The CUDA staging analysis follows the SSA use-def graph backwards from both
`dot` operands. Operations used only to describe the matrix tiles are removed
from ordinary per-thread scalar lowering, while scalar tile origins and values
shared with the output address remain available.

Each CUDA thread copies linear shared-memory positions

```text
threadIdx.x
threadIdx.x + threads_per_block
threadIdx.x + 2 * threads_per_block
...
```

until the complete A `[BM, BK]` or B `[BK, BN]` tile has been covered. Logical
row and column coordinates determine the global matrix address. Out-of-bounds
positions receive `0.0`, so edge tiles are safe without divergent barriers.
After both cooperative loads, the backend emits `__syncthreads()` so no thread
starts reading a tile before all writes have completed.

For a small output tile, each CUDA thread may still own one output coordinate
`(tile_i, tile_j)` and one `f32` register accumulator. Larger `tl.dot` outputs
use at most 32 physical threads in the current policy and assign a register tile
to every thread. For example, an `(8, 8)` output executed by `(4, 8)` threads
uses two accumulators per thread:

```text
accumulator_0 = 0.0
accumulator_1 = 0.0

for k in range(BK):
    accumulator_0 += shared_a[tile_i, k] * shared_b[k, tile_j]
    accumulator_1 += shared_a[tile_i + 4, k] * shared_b[k, tile_j]
```

Rank-2 broadcast values participate in the same mapping without unnecessary
duplication. A `(BM, 1)` row-offset tile stores registers only along the row
axis, while a `(1, BN)` column-offset tile stores registers only along the
column axis. Binary arithmetic combines them into full `(BM, BN)` register
tiles when necessary. Pointer addition, Boolean masks, and stores select the
matching pointer, value, and mask register for every logical output element.

A second `__syncthreads()` ensures every thread has finished reading the current
shared buffers before a runtime K-loop iteration overwrites them with the next
tiles. Partial K-tiles are zero-filled by the existing load masks.

A complete tiled matmul can therefore accumulate several `tl.dot` results:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k_base in range(0, K, BK):
    # Build and load A [BM, BK] and B [BK, BN] tiles.
    acc = acc + tl.dot(a_values, b_values)

tl.store(output_pointers, acc, mask=output_mask)
```

Version 14 intentionally stops after shared-memory staging. Version 15 adds the
CUDA-core FMA loop, safe shared-buffer reuse, and execution across multiple
K-tiles. Version 16 separates the physical thread tile from the logical dot
output, carries register-tile accumulators through runtime K-loops, and emits a
masked store for every result owned by a thread.

## PyTorch tensor interoperability

Runtime pointer arguments may be NumPy arrays, CuPy arrays, or PyTorch tensors.
All three become the same `ptr<f32>` parameter in typed SSA, so the frontend,
optimizer, and backend source are independent of the Python array framework.

CPU NumPy arrays and CPU PyTorch tensors are compilation-only inputs. A CUDA
PyTorch tensor compiles and executes the kernel directly:

```python
import torch

n = 1_000
block = 256
x = torch.ones(n, device="cuda", dtype=torch.float32)
y = torch.ones(n, device="cuda", dtype=torch.float32)
out = torch.empty_like(x)

add_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
    x,
    y,
    out,
    n,
    BLOCK=block,
)
```

CuPy remains the internal CUDA compiler and launcher. At the runtime boundary,
a Torch CUDA tensor is detached from autograd metadata and exported through
DLPack:

```text
torch.Tensor -- detach -- DLPack -- zero-copy CuPy view
                                           |
                                           v
                                 RawKernel/cubin launch
```

`detach()` does not copy storage. It only makes the raw tensor memory
exportable through DLPack, matching a low-level Triton launch: tensors with
`requires_grad=True` are accepted as pointers, but the launch does not create
an autograd graph or provide a backward operation.

Torch launches run in `torch.cuda.current_stream()` by wrapping it with
`cupy.cuda.Stream.from_external()`. DLPack conversion and kernel execution
happen inside the same stream context, so work queued before and after the
kernel remains correctly ordered without a global device synchronization.

One launch must use either CuPy CUDA arrays or Torch CUDA tensors, not a mixture
of the two frameworks. All array arguments must be on the same CUDA device.
Mixing CPU and CUDA arrays is also rejected. As elsewhere in the current MVP,
runtime arrays must be C-contiguous and have `float32` elements.

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
CUDA, so `src` is CUDA C++. With NumPy or CPU Torch arguments, compilation stops
there. With CuPy arrays or CUDA Torch tensors, the generated kernel is also
compiled and launched. Shared expressions such as `offsets` and `mask` are
lowered once and referenced by their SSA values wherever they are reused.

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

For NumPy or CPU Torch arguments, the MLIR backend stops after source
generation, so MLIR Python bindings are not required just to inspect the
emitted MLIR. For CuPy arrays or CUDA Torch tensors, the backend runs a small
pass pipeline that attaches an NVVM target, converts GPU operations to NVVM,
emits a GPU binary, extracts the cubin, loads it through CuPy, and launches it
with the same grid and thread-block size used by the CUDA backend. CUDA arrays
are passed using the ranked-memref ABI: allocated pointer, aligned pointer,
offset, size, and stride.

The test kernels also include a copy, 2D matrix add, ReLU through
`tl.maximum`, leaky ReLU through `tl.where`, sigmoid through negation,
`tl.exp`, addition, and division, row-wise `tl.sum`/`tl.max`/`tl.min`
reductions, a numerically stable row-wise softmax, and a long-row sum that uses
`tl.static_range` to unroll several block-sized loads at compile time. The
current tests also include matrix multiplication kernels: an older naive
rank-1-vector version and a rank-2 tiled version that combines a 2D launch grid,
2D block broadcasting, and masked tile stores. The rank-2 matmul is covered both
with a compile-time-unrolled `K` and with a runtime `range(K)` that becomes a
CUDA loop. Both versions initialize their rank-2 accumulator directly with
`tl.zeros((BM, BN), tl.float32)`.

Before CUDA code generation, the SSA IR is checked by a verifier. The verifier
validates definition order, result declarations, operand types, broadcast
shapes, pointer operations, memory masks, and operation-specific rules such as
`tl.exp` requiring `f32`, `tl.where` lowering to a Boolean `select`,
`expand_dims` preserving element types while inserting a size-1 dimension, and
reductions consuming one power-of-two rank-1 block whose width matches the CUDA
block size. For runtime loops it also validates region scoping, scalar bounds,
the positive constant step, definition order, and matching types and counts for
carried inputs, region arguments, yielded values, and loop results. For block
factory functions it checks that shapes are non-empty and positive, dtypes are
supported, result block types match the declared shape/dtype, and `tl.full` has
a scalar fill value convertible to the requested dtype. For `tl.dot`, it
requires two rank-2 `f32` operands, matching inner dimensions, and an exact
`[M, N]` rank-2 `f32` result.

Straight-line verified SSA then runs through a small optimization pipeline:

- constant folding and local simplifications such as `select(true, x, y) -> x`;
- common subexpression elimination for pure operations;
- dead-code elimination.

The verifier runs after every optimization pass so malformed rewrites fail
before CUDA code generation. The runtime-loop MVP is fully verified but skips
these rewrite passes because they are not region-aware yet.

## Current limitations

- Generated backend source is returned as a string. Execution requires CuPy
  built for the installed CUDA version and an available CUDA GPU. CUDA launch
  arguments may be homogeneous CuPy arrays or PyTorch CUDA tensors; PyTorch is
  imported only for Torch execution. NumPy arrays and CPU Torch tensors remain
  compilation-only.
- `MYTRITON_BACKEND` can be `cuda` or `mlir`. The CUDA backend is the default
  and supports the full current mytriton test language. The MLIR backend is an
  experimental MVP for 1D elementwise kernels. MLIR source generation does not
  require MLIR Python bindings, but MLIR cubin execution does.
- Kernel functions must have source available to `inspect.getsource`; functions
  created dynamically or entered only in an interactive session may not be
  recoverable by the AST frontend.
- Compile-time `range` and `tl.static_range` loops are unrolled by the AST
  frontend. Runtime `range` supports scalar `i32` bounds and a positive constant
  step. Its induction variable and assignment targets must be simple names, and
  assigning to the induction variable is rejected. `if`/`while`,
  `break`/`continue`, `for/else`, and other symbolic Python control flow are not
  supported.
- Runtime array arguments must be C-contiguous `float32` arrays. One execution
  cannot mix CPU and CUDA arrays, CuPy and Torch CUDA arrays, or arrays from
  different CUDA devices. Raw launches accept Torch tensors with
  `requires_grad=True`, but do not participate in PyTorch autograd.
- The launch grid is evaluated and used for CUDA execution, but it is not
  represented in the IR.
- The CUDA kernel layout is inferred from block-shaped operands of observable
  `store` operations, input widths required by reductions, and `tl.dot` result
  shapes. The current elementwise policy assigns one CUDA thread to each output
  element; reductions may retain a wider thread shape for a scalar output.
  CUDA-core dot kernels use at most 32 threads and can assign several output
  elements to a per-thread rank-2 register tile. Scalar-only kernels use one
  thread per block. The dot thread-count policy is fixed rather than tuned for
  a particular GPU.
- JIT cache entries are specialized by runtime types and constexpr values. Python
  globals and closure values used by a kernel must remain unchanged; call
  `kernel.clear_cache()` after changing them.
- CUDA lowering currently supports program IDs, `tl.arange`, basic arithmetic and
  comparison, Boolean `&`, rank-2 `expand_dims` via `x[:, None]` and
  `x[None, :]`, elementwise minimum and maximum, negation, `tl.exp`,
  `tl.where`, pointer addition, masked loads, masked stores, block-local
  `tl.sum`/`tl.max`/`tl.min` reductions, compile-time `tl.static_range` loops,
  and structured runtime `range` loops, including nested loops and multiple
  carried values. It also supports `tl.empty`, `tl.full`, and `tl.zeros` for
  rank-1 and rank-2 logical blocks. Reduction lowering internally emits the
  CUDA shared-memory scratch buffers and synchronization needed for block-local
  reductions. Floating-point elementwise extrema propagate NaNs and choose the
  right-hand operand when values compare equal. For canonical matrix-load
  operands, `tl.dot` lowering emits shared-memory declarations, cooperative
  masked loads with zero-filled boundaries, a CUDA-core FMA loop, and the
  barriers required before reading and reusing the shared tiles. Runtime
  `range` loops can accumulate multiple K-tiles into one result. Dot outputs,
  their broadcasted row/column coordinates, pointer arithmetic, masks,
  loop-carried accumulators, and stores support several register-resident
  results per CUDA thread.
- Reductions are currently single-block reductions over the SSA vector width.
  The vector width must be a power of two and must match the CUDA thread block
  size. Larger rows can be handled by statically unrolling multiple loads into
  one block-local partial vector, as in the long-row sum test, but there is no
  multi-block reduction yet.
- Matrix multiplication supports a correct tiled CUDA-core implementation for
  canonical `tl.dot` operands. A `[BM, BK]` and B `[BK, BN]` are loaded
  cooperatively into shared memory, each thread computes a rank-2 register tile
  of C elements, and a runtime CUDA loop can traverse the complete K dimension.
  The implementation prioritizes correctness over performance: it has a fixed
  one-warp-at-most dot layout policy and no vectorized loads, shared-memory
  padding or swizzling, double buffering, asynchronous copies, tensor-core
  instructions, or autotuning yet.
  `tl.empty`, `tl.full`, and `tl.zeros` continue to represent logical
  per-thread values rather than shared-memory allocations.
- MLIR lowering currently supports only `ptr<f32>` parameters as
  `memref<?xf32>`, scalar `i32`/`f32`/`bool`, `tl.program_id(0)`,
  `tl.arange(0, BLOCK)`, basic arithmetic and `<`, pointer addition, masked
  loads, masked stores, and scalarized `tl.full`/`tl.zeros` values for rank-1
  blocks. It intentionally rejects `tl.empty`, nonzero program axes, nonzero
  `arange` starts, and rank-2 block shapes instead of silently generating wrong
  code. It does not yet support 2D program IDs, reductions, `expand_dims`,
  Boolean `&`, `tl.maximum`, `tl.minimum`, `tl.where`, negation, `tl.exp`,
  `tl.static_range`, runtime `range`, or matrix multiplication.
- MLIR execution currently supports only 1D C-contiguous CUDA arrays because it
  builds one-dimensional memref descriptors. Torch CUDA tensors are normalized
  to zero-copy CuPy views before those descriptors are constructed.
- The SSA IR has structured `for` regions and loop-carried `iter_args`/`yield`
  values, but it has no general basic blocks, conditional control flow, or phi
  nodes outside this loop representation.
- The optimizer is intentionally small. It does local simplification, constant
  folding, common subexpression elimination, and dead-code elimination, but it
  has no control-flow or memory-aware optimization passes yet. Kernels containing
  runtime loops currently bypass the rewrite pipeline after verification.

## Development

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

To enable CUDA execution with CUDA 12, install the matching CuPy wheel:

```bash
python -m pip install -e ".[cuda12]"
```

PyTorch is an optional runtime integration rather than a project dependency.
Install a PyTorch build matching the local CUDA environment separately. When
PyTorch is available, CUDA tensors can be passed directly to kernels; CuPy is
still required internally for CUDA source compilation and kernel launch.

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
