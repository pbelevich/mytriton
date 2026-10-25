# Version 24 `ldmatrix` and Swizzled Shared Memory on NVIDIA A100

## Summary

Version 24 replaces scalar FP16/BF16 fragment assembly with warp-cooperative
`ldmatrix.sync.aligned.m8n8` loads and adds a 16-byte XOR swizzle to the
physical shared-memory layout. The global-to-shared writer and the
shared-to-register reader use the same logical-to-physical mapping, so aligned
`uint4` staging remains available while `ldmatrix` sees conflict-free row
addresses.

For BF16 GEMM at `4096 x 4096 x 4096`, the established `64 x 16 x 64`,
four-warp kernel improves from **40.83 TFLOP/s** in Version 23 to
**59.95 TFLOP/s** in Version 24: a **1.47x** speedup. FP16 reaches
**59.88 TFLOP/s**. Both runs report zero sampled error.

The result reaches the lower bound of the planned 60--100 TFLOP/s target when
rounded to one decimal place. It does not reach the upper part of that range.
At `8192 x 8192 x 8192`, both types reach about 49.8 TFLOP/s. The next major
step remains a larger CTA and warp tile rather than more tuning of this
`64 x 64` output tile.

Nsight Compute confirms that the improvement comes from the intended source:

- shared-load bank conflicts fall from 236.68 million to 1.76 million;
- total shared bank conflicts fall from 242.26 million to 11.92 million;
- shared wavefronts fall from 378.64 million to 114.66 million;
- MIO-throttle stalls fall from 2.81 to 0.09 cycles per issued instruction;
- dynamic warp instructions fall by 28.1%;
- the kernel retains 24 KiB of static shared memory and 37.5% theoretical
  occupancy.

## Benchmark metadata

- **Date:** October 24, 2026 EDT / October 25, 2026 UTC
- **Branch:** `version-24-ldmatrix`
- **Implementation commit:** `20564bc75772223dab28d90802c7bc33fdb8f47c`
- **Version 23 base:** `d2d605c622cda4b40912e331a00d6ffb29a3acae`
- **GPU:** NVIDIA A100-SXM4-40GB
- **Compute capability:** 8.0 (`sm_80`)
- **Streaming multiprocessors:** 108
- **PyTorch:** 2.11.0+cu130
- **PyTorch CUDA build:** 13.0
- **CuPy:** 14.0.1
- **Primary tile:** `BM=64`, `BK=16`, `BN=64`
- **Warps per CTA:** 4
- **Stable primary run:** 10 warmup iterations and 100 measured iterations
- **Timing:** CUDA events around the compiled raw kernel; CUDA Graph timing is
  recorded separately

Version 23 reference values come from
[`matmul_vectorized_staging_a100_report.md`](matmul_vectorized_staging_a100_report.md).

## Validation

The complete local suite passed on an RTX 2080:

```text
585 passed, 3 skipped
```

The complete available Colab A100 suite passed with CUDA execution required.
Only MLIR-dependent tests were deselected because the runtime does not provide
MLIR Python bindings:

```text
577 passed, 11 deselected
```

The A100 run covers native SM80 BF16 and FP16, masked boundary tiles,
single-warp MMA, composable warp tiles, multi-warp CTA matmul, and generated
`.x1`, `.x2`, and `.x4` `ldmatrix` forms.

## Primary performance result

### BF16

| Matrix size | Version 23 raw | Version 24 raw | V23 TFLOP/s | V24 TFLOP/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 3.3659 ms | **2.2927 ms** | 40.83 | **59.95** | **1.47x** |
| 8192 x 8192 x 8192 | 28.5911 ms | **22.0815 ms** | 38.46 | **49.79** | **1.29x** |

The corresponding CUDA Graph medians were 2.2743 ms and 22.0457 ms. Maximum
sampled absolute error was `0.000e+00` for both sizes.

### FP16

| Matrix size | Version 23 raw | Version 24 raw | V23 TFLOP/s | V24 TFLOP/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 3.3679 ms | **2.2953 ms** | 40.81 | **59.88** | **1.47x** |
| 8192 x 8192 x 8192 | 28.5957 ms | **22.0749 ms** | 38.45 | **49.81** | **1.30x** |

The corresponding CUDA Graph medians were 2.2753 ms and 22.0442 ms. Maximum
sampled absolute error was `0.000e+00` for both sizes.

The matching FP16 and BF16 behavior is expected: both use two-byte shared
elements, the same swizzle and `ldmatrix` shapes, and corresponding
`mma.sync.m16n8k8` instructions.

## Optimization ablation

The intermediate measurements are important because they separate the
instruction change from the data-layout change.

| Variant, BF16 `4096^3` | Raw TFLOP/s | Shared bytes | Occupancy estimate | Result |
|---|---:|---:|---:|---|
| Version 23 scalar fragments | 40.83 | 24,576 | 37.5% | baseline |
| `ldmatrix`, row-major shared | 40.85 | 24,576 | 37.5% | instruction reduction alone is neutral |
| `ldmatrix`, padded rows | 52.33 | 27,136 | 31.2% | fewer conflicts, but padding costs occupancy |
| `ldmatrix`, XOR-swizzled segments | **59.95** | 24,576 | 37.5% | best result |

The padded experiment proves why a layout optimization must be evaluated as a
whole. Padding makes row starts bank-friendly, but the larger allocation drops
resident CTAs. The XOR layout obtains the bank mapping without consuming any
extra shared memory.

## Physical shared-memory mapping

Each 16-byte `uint4` contains eight BF16 or FP16 values and is treated as one
physical segment. The segment remains contiguous; only its position within a
row changes.

For the default LHS tile, a row has two segments. Rows 4--7 exchange them:

```text
physical_segment = logical_segment XOR ((row >> 2) & 1)
```

For the RHS tile, a row has eight segments:

```text
physical_segment = logical_segment XOR (row & 7)
```

The four-segment case used by `BK=32` uses the equivalent bit permutation:

```text
row_swizzle = (row & 2) | ((row >> 2) & 1)
physical_segment = logical_segment XOR row_swizzle
```

The same `CudaSharedBuffer.element()` mapping is used by vectorized stores,
scalar tail stores, and `ldmatrix` address generation. This is required for
correctness: a swizzled reader with a row-major writer would silently multiply
the wrong matrix values.

## Generated CUDA and SASS

The generated CUDA converts the mapped generic pointer to the 32-bit shared
address required by PTX and emits grouped warp loads:

```cuda
unsigned address = static_cast<unsigned>(
    __cvta_generic_to_shared(&shared[row * stride + physical_column])
);

asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(address)
    : "memory"
);
```

For B, `.trans` converts the row-major logical tile into the column-major
fragment consumed by `mma.sync.m16n8k8.row.col`.

The default tile contains the following static SASS operations:

| SASS operation | Version 23 | Version 24 |
|---|---:|---:|
| `LDS.U16` | 16 | **0** |
| `LDSM` | 0 | **4** |
| `HMMA.1688.F32.BF16` | 16 | 16 |
| `LDG.E.128` | 2 | 2 |
| `STS.128` | 2 | 2 |
| `BAR.SYNC` | 2 | 2 |

Each A/B `LDSM` pair is placed immediately before the eight MMA instructions
that consume it, keeping fragment live ranges bounded.

## Nsight Compute comparison

NCU replay time is diagnostic and is not used as benchmark latency. Version 23
and Version 24 used the same `4096^3` BF16 shape and four-warp tile.

| Metric | Version 23 | Version 24 | Change |
|---|---:|---:|---:|
| Dynamic warp instructions | 716.80 M | **515.65 M** | -28.1% |
| MIO-throttle stall | 2.81 | **0.09** | -96.8% |
| Total shared bank conflicts | 242.26 M | **11.92 M** | -95.1% |
| Shared-load bank conflicts | 236.68 M | **1.76 M** | -99.3% |
| Shared-store bank conflicts | **5.58 M** | 10.16 M | +82.1% |
| Shared wavefronts | 378.64 M | **114.66 M** | -69.7% |
| Long-scoreboard stall | **5.94** | 7.52 | +26.6% |
| Short-scoreboard stall | 2.23 | **0.71** | -68.2% |
| Eligible warps/scheduler | 0.58 | **0.60** | +3.4% |
| Achieved occupancy | **36.03%** | 34.31% | -1.72 pp |
| Theoretical occupancy | 37.5% | 37.5% | unchanged |
| Registers/thread | 64 | 64 | unchanged |
| Static shared memory | 24 KiB | 24 KiB | unchanged |
| NCU duration | 4.22 ms | **2.73 ms** | -35.3% |

Stall values are average warp cycles per issued instruction. The remaining
shared-store conflicts come from the cooperative staging pattern, but they are
small compared with the eliminated `ldmatrix` load conflicts. Long-scoreboard
latency remains the primary exposed stall and motivates the larger tile and
pipeline work planned for subsequent versions.

## Warp-count check

The default remains four warps per CTA:

| `num_warps` | BF16 `4096^3` raw | Registers/thread | Occupancy estimate |
|---:|---:|---:|---:|
| 4 | **59.95 TFLOP/s** | 64 | 37.5% |
| 8 | 51.01 TFLOP/s | 48 | 62.5% |

More occupancy does not compensate for the smaller per-warp output tile and
additional CTA coordination in the eight-warp schedule.

## Interpretation and next step

Version 24 demonstrates that `ldmatrix` and the shared-memory layout are one
optimization, not two independent details. `ldmatrix` alone removed scalar
instructions but left performance at 41 TFLOP/s. Matching the writer and
reader with an XOR-swizzled physical layout raised the same kernel to 60
TFLOP/s without extra storage.

The result reaches only the lower edge of the planned range. The current CTA
still computes a `64 x 64` output tile, so each staged input value is reused
less than it would be in a `128 x 128` CTA. Version 25 should therefore move to
the planned larger CTA and `64 x 64` warp tile before introducing `cp.async`.

## Reproduction

Upload the current checkout:

```bash
./tools/sync_colab.sh
```

Run the stable primary benchmark:

```bash
./benchmarks/run_matmul_colab.sh \
    --sizes 4096 8192 \
    --tiles 64x16x64 \
    --dtype bf16 \
    --num-warps 4 \
    --warmup 10 \
    --repeats 100
```

Capture generated CUDA, PTX, the exact NVRTC cubin, SASS, resource usage, and
the Nsight Compute report:

```bash
./benchmarks/profile_matmul_colab.sh
```

## Conclusion

Version 24 delivers the intended `ldmatrix` lowering and a matching
conflict-free physical shared layout. On A100, the primary BF16 kernel improves
from 40.83 to 59.95 TFLOP/s, shared-load bank conflicts fall by 99.3%, and no
additional shared memory is consumed. The optimization reaches the promised
lower bound while leaving a clear path to higher throughput: increase CTA and
warp tile reuse in Version 25, then overlap staging and compute with
`cp.async` in Version 26.
