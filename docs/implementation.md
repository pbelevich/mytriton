# mytriton implementation guide

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
- [ver18](https://github.com/pbelevich/mytriton/tree/ver18): conflict-aware
  shared-memory row padding for A tiles, compile-time-unrolled CUDA-core dot
  loops, two-stage ping-pong buffers for canonical runtime K loops, fewer block
  barriers, and conservative shared-memory budget validation.
- [ver19](https://github.com/pbelevich/mytriton/tree/ver19): public
  `tl.float16`/`tl.bfloat16` dtypes and `tl.cast`, shared numeric-promotion
  rules across type inference, SSA verification, and CUDA lowering,
  low-precision runtime pointer inference, explicit CUDA conversions,
  two-byte shared-memory tiles, and `f16`/`bf16` CUDA-core `tl.dot` with
  `f32` accumulation.
- [ver20](https://github.com/pbelevich/mytriton/tree/ver20): NVIDIA
  tensor-core paths for canonical `f16[16, 8] x f16[8, 8]` and
  `bf16[16, 8] x bf16[8, 8]` dots, architecture-aware lowering, explicit
  warp-fragment layouts, packed 16-bit PTX operands, fused loop-carried `f32`
  accumulation across K-tiles, fragment redistribution for masked stores,
  and CUDA execution tests covering partial boundary tiles.
- [ver21](https://github.com/pbelevich/mytriton/tree/ver21): composable
  one-warp MMA tiles whose M, N, and K dimensions are positive multiples of
  16, 8, and 8, respectively; logical dots are decomposed into grids of
  `mma.sync.m16n8k8` instructions with one FP32 accumulator fragment per
  `16 x 8` output region.
- [ver22](https://github.com/pbelevich/mytriton/tree/ver22): multi-warp CTA
  tiles that partition a larger output tile across a two-dimensional warp
  grid, cooperatively stage and reuse CTA-wide A/B operands, carry independent
  per-warp MMA fragments through the runtime K-loop, and redistribute the
  complete CTA result for masked stores. Tensor Core operands use aligned,
  padded shared-memory layouts and grouped `ldmatrix.x1`, `x2`, and `x4`
  loads, including transposed B-fragment loading.

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
integers. The supported public dtype objects are `tl.int1`, `tl.int32`,
`tl.float16`, `tl.bfloat16`, and `tl.float32`. `tl.full` accepts a scalar
Boolean, integer, floating-point, or symbolic runtime value and converts
numeric values to the requested numeric dtype. A block value cannot be used as
the fill value.

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

Rank-2 `f16`, `bf16`, or `f32` blocks can be combined with the public `tl.dot`
operation:

```python
lhs = tl.zeros((BM, BK), tl.float16)
rhs = tl.zeros((BK, BN), tl.float16)
result = tl.dot(lhs, rhs)
```

The operands must have shapes `[M, K]` and `[K, N]`. Their inner dimensions
must match, and the result has shape `[M, N]`:

```text
%0 = zeros {shape=(4, 16), dtype=f16} : block<4x16 x f16>
%1 = zeros {shape=(16, 8), dtype=f16} : block<16x8 x f16>
%2 = dot %0, %1 : block<4x8 x f32>
```

The expression-tree type inference and SSA verifier independently check operand
rank, floating-point element types, equal operand element types, matching
reduction dimensions, and the exact result type. Both `f16 x f16` and
`bf16 x bf16` dot products produce an `f32` result. `dot` is a pure SSA
operation, so duplicate operations are eligible for common subexpression
elimination and unused operations can be removed by dead-code elimination.

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

## Shared-memory layout optimization

Version 18 makes the shared-memory implementation more explicit. A shared
buffer now records its logical shape separately from its physical row stride:

```text
row_stride = columns + row_padding
stage_size = rows * row_stride
offset(stage, row, column) =
    stage * stage_size + row * row_stride + column
```

This distinction lets the backend change the physical layout without changing
the logical `[rows, columns]` tile seen by `tl.dot`. It also lets one allocation
hold several copies, or stages, of the same logical tile.

CUDA shared memory has 32 banks, and consecutive `f32` values map to
consecutive banks. During the dot loop, several output rows may read the same
column from different rows of A. When the A row stride shares a large common
factor with 32, those reads can repeatedly select the same banks. The backend
detects that case from the CUDA thread layout and adds one padding element to
each A row when it increases the number of distinct banks reached. B remains
unpadded because the current access pattern broadcasts its values rather than
performing the corresponding cross-row access.

For canonical runtime K loops, the backend allocates two stages for both A and
B. Iteration `k_base` selects its stage with:

```cpp
int dot_stage = (k_base / BK) & 1;
```

Consecutive iterations therefore alternate between stage 0 and stage 1. Each
iteration cooperatively fills its selected stage, synchronizes the block, and
then runs the CUDA-core FMA loop over that stage. Since the next iteration
writes the other stage, the trailing shared-buffer reuse barrier can be
omitted. This remains safe for a multi-warp CTA: the next block barrier cannot
complete until every warp has finished the preceding iteration, while the
intervening writes target the other stage.

The optimization is deliberately matched only for the canonical form:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k_base in range(0, K, BK):
    # A uses k_base as its column origin.
    # B uses k_base as its row origin.
    acc = acc + tl.dot(a_values, b_values)
```

The loop must start at zero, advance by the dot reduction width, contain one
stageable dot, and carry and yield only its accumulator. Nested loops and loops
with additional operations use the existing single-stage implementation with
both synchronization barriers. Keeping this conservative fallback prevents an
optimization matcher from changing the semantics of more general loops.

The generated reduction loop also receives `#pragma unroll`, allowing the CUDA
compiler to unroll its compile-time-known `BK` trip count. Before emitting a
kernel, the backend accounts for row padding and every stage and rejects a dot
whose shared allocations exceed the current 48 KiB per-block budget.

The two stages currently provide ping-pong storage and barrier elimination;
they do not yet overlap loading the next tile with computing the current one.
True asynchronous prefetching is deferred until the backend has the alignment
metadata and `cp.async` support needed to implement it safely.

## PyTorch tensor interoperability

Runtime pointer arguments may be NumPy arrays, CuPy arrays, or PyTorch tensors.
The runtime dtype determines whether an array becomes `ptr<f16>`, `ptr<bf16>`,
or `ptr<f32>` in typed SSA, while the frontend, optimizer, and backend source
remain independent of the Python array framework.

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
runtime arrays must be C-contiguous and have `float16`, `bfloat16`, or `float32`
elements.

## Low-precision types and mixed-precision dot

Version 19 adds the public `tl.float16` and `tl.bfloat16` dtype objects. NumPy,
CuPy, and PyTorch `float16` arrays become `ptr<f16>` runtime parameters; PyTorch
`bfloat16` tensors become `ptr<bf16>` parameters. Runtime arrays remain
C-contiguous, and the currently supported element types are `float16`,
`bfloat16`, and `float32`.

Numeric operations use one shared promotion rule:

```text
same + same       -> same
anything + f32    -> f32
f16 + bf16        -> f32
i32 + f16         -> f16
i32 + bf16        -> bf16
```

The same rule is used independently when constructing expression types,
verifying SSA, and emitting CUDA arithmetic and comparisons. Mixed operands are
converted before applying the CUDA operator, rather than converting the result
after a lower-precision operation.

The public `tl.cast` operation performs an explicit elementwise numeric
conversion while preserving the shape of a block:

```python
values = tl.load(source + offsets)
converted = tl.cast(values, tl.float32)
tl.store(destination + offsets, converted)
```

It remains explicit in SSA:

```text
%2 = load %1, none, none : vector<8 x f16>
%3 = cast %2 {dtype=f32} : vector<8 x f32>
```

The CUDA backend conditionally includes `cuda_fp16.h` or `cuda_bf16.h` and uses
the corresponding conversion intrinsics:

```cuda
float f16_value = __half2float(value);
float bf16_value = __bfloat162float(value);
__half rounded_f16 = __float2half_rn(value);
__nv_bfloat16 rounded_bf16 = __float2bfloat16_rn(value);
```

Conversions are emitted independently for every element owned by a CUDA
thread, including kernels whose logical rank-2 output is distributed across
several registers per thread.

Low-precision dot operands remain 16-bit while they are stored in global and
shared memory. Each shared element therefore occupies two bytes instead of
four. The current CUDA-core implementation converts both values to `float`
inside the reduction loop and accumulates into an `f32` register:

```cuda
float accumulator = 0.0f;

for (int k = 0; k < BK; ++k) {
    accumulator +=
        __half2float(shared_a[row * BK + k]) *
        __half2float(shared_b[k * BN + column]);
}
```

The `bf16` path uses `__bfloat162float` in the same way. Masked cooperative
loads explicitly convert their zero fallback to the shared-buffer element type,
avoiding ambiguous CUDA conditional expressions.

This is still an ordinary CUDA-core implementation. Version 19 establishes the
types, conversions, storage widths, and mixed-precision accumulation semantics
needed by a later tensor-core lowering; it does not emit `mma.sync` or WMMA
instructions yet.

## Tensor-core dot

Version 20 introduced the first specialized tensor-core lowering. Its hardware
atom is this dot contract for matching operand types `T`:

```text
A: T[16, 8]
B: T[8, 8]
C: f32[16, 8]
```

The operation is executed by one 32-thread CUDA warp. FP16 uses:

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

on compute capability 7.5 or newer. Ampere (`sm_80`) and newer targets can use
the otherwise identical native BF16 variant:

```text
mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32
```

BF16 falls back to CUDA cores on pre-Ampere targets instead of emitting an
unsupported PTX instruction. CUDA compilation artifacts are cached per
architecture so code selected for `sm_80` cannot be reused on an incompatible
device. Compilation-only calls with CPU arrays conservatively use `sm_75`.

### Composable warp MMA tiles

Version 21 separates the physical `m16n8k8` instruction from the logical tile
computed by one warp. A low-precision dot is eligible when its dimensions are
positive multiples of the instruction dimensions:

```text
A: T[M, K]    M % 16 == 0
B: T[K, N]    N % 8  == 0
C: f32[M, N]  K % 8  == 0
```

`CudaMmaWarpTileLayout` records the logical `(M, N, K)` shape and derives the
number of physical atoms along each axis:

```text
m_tiles = M / 16
n_tiles = N / 8
k_tiles = K / 8

instruction_count = m_tiles * n_tiles * k_tiles
```

For example, a `32 x 16 x 16` dot becomes:

```text
m_tiles = 2
n_tiles = 2
k_tiles = 2
instruction_count = 2 * 2 * 2 = 8
```

The instructions are emitted in `(m_tile, n_tile, k_tile)` order:

```text
(0, 0, 0)  (0, 0, 1)
(0, 1, 0)  (0, 1, 1)
(1, 0, 0)  (1, 0, 1)
(1, 1, 0)  (1, 1, 1)
```

Each `(m_tile, n_tile)` pair owns one independent four-register FP32
accumulator fragment. Instructions with different `k_tile` values update the
same fragment, implementing the reduction over K. The fragment index is:

```text
fragment_index = m_tile * n_tiles + n_tile
```

Consequently, each lane owns:

```text
4 * m_tiles * n_tiles
```

FP32 accumulator registers. The `32 x 16` output above has four fragments and
therefore 16 accumulator registers per lane.

The layout also offsets the physical fragment coordinates into the logical
tile. For instruction `(m_tile, n_tile, k_tile)`, A receives row and K offsets,
B receives K and column offsets, and C receives row and column offsets:

```text
A offset: (m_tile * 16, k_tile * 8)
B offset: (k_tile * 8, n_tile * 8)
C offset: (m_tile * 16, n_tile * 8)
```

The SSA matcher returns a mapping from each eligible dot result ID to its
`CudaMmaWarpTileLayout`. CUDA generation uses this mapping both for standalone
dots and for dots nested in runtime K-loops. Shapes that do not satisfy the
instruction multiples, `f32` operands, and unsupported target/type
combinations retain the CUDA-core implementation.

Within each physical instruction, tensor-core operands are warp fragments
rather than ordinary per-thread register tiles. For lane `lane`, the backend
computes:

```text
group = lane >> 2
thread = lane & 3
```

Each lane owns four A elements:

```text
A[group,     thread * 2]
A[group,     thread * 2 + 1]
A[group + 8, thread * 2]
A[group + 8, thread * 2 + 1]
```

two B elements:

```text
B[thread * 2,     group]
B[thread * 2 + 1, group]
```

and four `f32` accumulator elements at the corresponding output coordinates:

```text
C[group,     thread * 2]
C[group,     thread * 2 + 1]
C[group + 8, thread * 2]
C[group + 8, thread * 2 + 1]
```

#### Operand loading with `ldmatrix`

Version 22 replaces per-element shared-memory reads and manual 16-bit packing
with warp-cooperative `ldmatrix` instructions. The existing cooperative loader
still stages the complete logical A and B tiles from global memory into shared
memory, but the Tensor Core path now gives those buffers a layout suitable for
hardware fragment loads.

Both operand buffers are explicitly aligned to 16 bytes. Their row strides are
multiples of eight 16-bit elements, and an extra eight-element padding is added
when consecutive logical rows would otherwise start in the same shared-memory
bank group. This padding changes only physical shared-memory addresses; logical
matrix coordinates and `tl.dot` semantics remain unchanged.

Before issuing `ldmatrix`, CUDA converts the generic shared-memory pointer to
the 32-bit address representation expected by PTX:

```cuda
unsigned address = static_cast<unsigned>(
    __cvta_generic_to_shared(&shared_tile[row * stride + column])
);
```

The low-level emitter supports the three matrix counts provided by PTX:
`ldmatrix.x1`, `ldmatrix.x2`, and `ldmatrix.x4`. It also supports the
transposing forms used for B.

One A fragment for `mma.m16n8k8` contains two row-major `8 x 8` matrices and
can therefore be loaded with `ldmatrix.x2`:

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
    "{%0, %1}, [%2];"
    : "=r"(a0), "=r"(a1)
    : "r"(address)
    : "memory"
);
```

When two adjacent M fragments are available, the backend combines their four
`8 x 8` matrices into one `ldmatrix.x4`:

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
    : "r"(address)
    : "memory"
);
```

For `x2`, the lower four bits of the lane ID select the 16 required row
addresses. For `x4`, all 32 lanes supply one row address. Register pairs
`(a0, a1)` and `(a2, a3)` are then assigned to the two logical M fragments.
An unpaired final M fragment falls back to `x2`.

The B operand is stored row-major in shared memory, while
`mma.sync.m16n8k8.row.col` expects its B fragment in column-major form.
The backend therefore uses transposing `ldmatrix` instructions.

Up to four adjacent N fragments are grouped into one `x4.trans`:

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
    : "r"(address)
    : "memory"
);
```

Groups of two use `x2.trans`, and a single remaining N fragment uses
`x1.trans`. Each group of eight lanes supplies the row addresses for one
transposed `8 x 8` matrix.

Operand registers are reused across the logical warp tile. An A fragment is
identified by `(m_tile, k_tile)` and is shared by every MMA atom with a
different `n_tile`. A B fragment is identified by `(k_tile, n_tile)` and is
shared by atoms with different `m_tile` values.

For a logical warp tile, the instruction counts are:

```text
A x4 count       = (m_tiles // 2) * k_tiles
A x2 count       = (m_tiles % 2) * k_tiles

B x4.trans count = (n_tiles // 4) * k_tiles
B x2.trans count = ((n_tiles % 4) // 2) * k_tiles
B x1.trans count = (n_tiles % 2) * k_tiles

MMA count        = m_tiles * n_tiles * k_tiles
```

For example, a `32 x 16 x 16` dot contains eight MMA instructions and requires
only two `ldmatrix.x4` instructions for A plus two `ldmatrix.x2.trans`
instructions for B. The previous ungrouped lowering required eight separate
`ldmatrix` instructions, while the Version 21 scalar path materialized and
packed operands separately for every MMA atom.

`ldmatrix` moves raw 16-bit payloads, so the same loading instructions serve
both FP16 and BF16 operands. The operand type remains encoded by the following
`mma.sync` instruction. FP16 remains available on `sm_75+`, while native BF16
MMA still requires `sm_80+`.

A lane supplies two packed A registers and one packed B register to
`mma.sync`. Four `f32` registers contain its accumulator fragment:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5}, "
    "{%6}, "
    "{%0, %1, %2, %3};"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    : "r"(a0), "r"(a1), "r"(b0)
);
```

The `+f` constraints make every accumulator register both an input and an
output. A standalone `tl.dot` initializes every logical C fragment with zero.
Successive K atoms feed the same four registers back into `mma.sync`, producing
the complete reduction rather than independent partial results.

A tiled matrix multiplication carries all of its accumulator fragments across
runtime K-loop iterations:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k_base in range(0, K, BK):
    # For example, load A [32, 16] and B [16, 16] tiles.
    acc = acc + tl.dot(a_values, b_values)

tl.store(output_pointers, acc, mask=output_mask)
```

The conservative double-buffering matcher verifies that the loop starts at
zero, advances by `BK`, contains one stageable dot, and yields exactly the
updated accumulator. For the tensor-core shape, CUDA lowering fuses the SSA
`dot` and `add`:

```text
%dot = dot %a, %b
%next_acc = add %acc, %dot
yield %next_acc
```

into one operation per iteration:

```text
next_acc = mma(a, b, acc)
```

The separate CUDA `add` is not emitted. Instead, the previous loop-carried
fragments initialize the `+f` operands of their corresponding `mma.sync`
instructions, and the instruction results become the accumulator for the next
K-tile. The number of loop-carried registers is derived from the logical warp
layout rather than being fixed at four.

A tensor-core accumulator has a different lane-to-element mapping from the
ordinary output register tile used by pointer arithmetic, masks, and stores.
After the K loop, each lane writes every four-register fragment into its
`16 x 8` region of a row-major shared-memory result tile. A block barrier makes
the complete logical tile visible, after which the existing register-tile
store path reads the appropriate elements and performs masked global stores.
This redistribution also preserves correctness for partial M and N boundary
tiles.

Partial K-tiles reuse the existing masked cooperative loads. Out-of-bounds A
and B elements are written as a zero of the operand type into shared memory, so
the final `mma.sync` may still execute with its complete fixed `m16n8k8` shape.

The runnable [Colab test notebook](../tests/mytriton_colab_tests.ipynb) installs the
current checkout, validates FP16/BF16 GPU support, and streams the complete
pytest output. The CUDA optional dependencies include `ml_dtypes`, which CuPy
needs to consume PyTorch BF16 storage through DLPack.

Version 21 focuses on one-warp composition and uses repeated scalar
shared-memory reads to pack operand fragments. Version 22 replaces those reads
with grouped `ldmatrix` instructions while extending ownership from one warp
to a complete CTA. The result still uses an extra shared-memory redistribution
before the store.

### Multi-warp CTA tiles

Version 22 introduces `CudaMmaCtaTileLayout`, which places identical logical
warp tiles in a two-dimensional grid. The first policy deliberately fixes the
per-warp logical result to `32 x 32`. A compatible larger dot is partitioned as:

```text
warps_m = BM / 32
warps_n = BN / 32
warp_count = warps_m * warps_n
threads_per_block = warp_count * 32
```

The matcher uses the CTA path only when both output dimensions are divisible by
32, the result requires more than one warp, and the CTA fits CUDA's 1,024-thread
limit. Smaller compatible dots continue to use the Version 21 one-warp path;
unsupported types, targets, and shapes retain the CUDA-core fallback.

For the current preferred tile, `BM=64`, `BK=16`, and `BN=64`, the layout is:

```text
CTA result: 64 x 64
warp tile:  32 x 32
warp grid:   2 x 2
block size:  4 warps = 128 threads

warp 0 -> C[ 0:32,  0:32]
warp 1 -> C[ 0:32, 32:64]
warp 2 -> C[32:64,  0:32]
warp 3 -> C[32:64, 32:64]
```

Each warp still executes the same generated `32 x 16 x 32` MMA program. That
program contains 16 static `mma.sync.m16n8k8` instructions: two M atoms, four N
atoms, and two K atoms. All four warps execute those instructions with distinct
warp offsets, so one K-loop iteration performs 64 warp-level MMA instructions
for the CTA-wide result.

The rank-two CUDA prologue decomposes `threadIdx.x` into a warp and lane:

```cuda
int mma_warp_id = threadIdx.x >> 5;
int mma_lane_id = threadIdx.x & 31;
int mma_warp_m = mma_warp_id / 2;
int mma_warp_n = mma_warp_id % 2;
```

The warp coordinates offset every fragment access. A depends on the warp row
but is shared by both warp columns; B depends on the warp column but is shared
by both warp rows:

```text
A warp offset = (mma_warp_m * 32, 0)
B warp offset = (0, mma_warp_n * 32)
C warp offset = (mma_warp_m * 32, mma_warp_n * 32)
```

This is the source of the new data reuse. The four warps cooperatively stage
one `A[64, 16]` tile and one `B[16, 64]` tile instead of loading four unrelated
pairs. The existing cooperative loader automatically distributes those larger
tiles across all 128 threads. Runtime K-loops still use two shared-memory
stages, and all warps agree on the selected stage before the block barrier.

Each `32 x 32` warp result contains eight `16 x 8` accumulator fragments. A
lane therefore owns 32 FP32 accumulator registers regardless of the total CTA
shape. `CudaMmaAccumulatorRef` keeps both the per-warp fragment layout and the
CTA layout: the former determines register names and physical MMA operands,
while the latter supplies the full logical result shape. The same metadata is
attached to the loop-carried value, preventing a one-warp accumulator from
being confused with a multi-warp accumulator of another shape.

The epilogue writes each warp's fragments into its non-overlapping region of a
CTA-wide row-major FP32 shared buffer. After one block barrier, the ordinary
register-tile store mapping reads the complete buffer and performs the existing
masked output store. For a `64 x 64` result the physical thread shape is
`8 x 16`, so each of the 128 threads owns 32 output elements in the final store
mapping. This shared round trip keeps partial M/N boundary tiles correct, at
the cost of 16 KiB of additional shared memory.

Block sizing is architecture-aware. The compiler derives the set of supported
MMA operand types from the selected CUDA target before initial SSA verification,
after optimization, and during CUDA generation. On `sm_80`, BF16 can therefore
select the 128-thread CTA layout. On `sm_75`, the same BF16 dot falls back to
CUDA cores and retains its ordinary thread layout. Keeping these decisions in
sync is essential because the verifier checks distributed `tl.arange` widths
against the selected CUDA block size.

The end-to-end tests cover FP16 execution on `sm_75+`, native BF16 execution on
`sm_80+`, runtime K accumulation, non-multiple matrix boundaries, generated
warp offsets, CTA-wide spill/store redistribution, and the pre-Ampere BF16
fallback.

The [A100 benchmark and Nsight Compute report](../benchmarks/matmul_multi_warp_a100_report.md)
measures the pre-`ldmatrix` four-warp `64 x 16 x 64` BF16 tile at approximately
24 TFLOP/s. That is about 1.8-2.0x faster than the one-warp baseline and shows
the value of CTA-wide operand reuse. The original profile identified scalar
`LDS.U16` fragment loads, shared-memory bank conflicts, long-scoreboard stalls,
and roughly 30% occupancy as important remaining costs. Grouped `ldmatrix`
loads address the scalar fragment loads; the backend still lacks vectorized
global-to-shared copies, a general fragment-compatible swizzle, `cp.async`, a
true overlapped software pipeline, and autotuning.

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

## SSA verification and optimization

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
requires two rank-2 operands with equal `f16`, `bf16`, or `f32` element types,
matching inner dimensions, and an exact `[M, N]` rank-2 `f32` result.

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
- Runtime array arguments must be C-contiguous `float16`, `bfloat16`, or
  `float32` arrays. One execution cannot mix CPU and CUDA arrays, CuPy and Torch
  CUDA arrays, or arrays from different CUDA devices. Raw launches accept Torch
  tensors with `requires_grad=True`, but do not participate in PyTorch autograd.
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
  reductions. It supports explicit numeric `tl.cast`, including register-wise
  conversion of rank-2 tiles, and emits the required CUDA `f16`/`bf16` headers
  and conversion intrinsics. Floating-point elementwise extrema propagate NaNs
  and choose the right-hand operand when values compare equal. For canonical
  matrix-load operands, `tl.dot` lowering emits shared-memory declarations,
  cooperative masked loads with zero-filled boundaries, a CUDA-core FMA loop,
  and the barriers required before reading and reusing the shared tiles.
  Runtime `range` loops can accumulate multiple K-tiles into one result. Dot
  outputs, their broadcasted row/column coordinates, pointer arithmetic, masks,
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
  Operands may use matching `f16`, `bf16`, or `f32` element types; low-precision
  tiles retain their two-byte representation in shared memory and are converted
  to `f32` for CUDA-core multiplication and accumulation.
  Conflict-aware row padding reduces bank conflicts for A, and canonical K
  loops alternate between two shared-memory stages to remove a tile-reuse
  barrier. Eligible `f16`/`bf16` dots whose M, N, and K dimensions are
  divisible by 16, 8, and 8 are composed from Tensor Core
  `mma.sync.m16n8k8` instructions when the CUDA target supports them; other
  operand types, shapes, and targets retain the CUDA-core path. Compatible
  output tiles divisible by `32 x 32` can be partitioned across a
  two-dimensional multi-warp CTA; smaller compatible tiles retain the one-warp
  path. The implementation uses aligned and padded low-precision shared buffers
  together with grouped `ldmatrix.x1`, `x2`, and `x4` operand loads. The current
  fixed warp-tile policy is not autotuned and still has no vectorized
  global-to-shared loads, general shared-memory swizzling, overlapped
  prefetching, or asynchronous copies.
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

The Colab helper uploads the current working tree, including uncommitted
changes, to the runtime connected through VS Code and installs it in editable
mode. This avoids benchmarking an outdated GitHub revision:

```bash
./tools/sync_colab.sh
```

After synchronization, the benchmark wrapper runs the current best Version 22
BF16 configuration (`BM=64`, `BK=16`, `BN=64`) at sizes 4096 and 8192:

```bash
./benchmarks/run_matmul_colab.sh
```

Pass normal `benchmark_matmul.py` arguments to sweep other shapes or tiles.
`benchmarks/profile_matmul_a100.sh` captures the generated CUDA, exact NVRTC
cubin, PTX, SASS, resource usage, and an Nsight Compute report on an A100. Its
default profile tile is also `64x16x64` and can be overridden with
`MYTRITON_PROFILE_TILE`.

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
