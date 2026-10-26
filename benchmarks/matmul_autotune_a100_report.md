# Version 28: CUDA Event Autotuning on A100

Date: 2026-10-26

Branch: `version-28-autotune`

Base: Version 27 commit `976b648`

## Result

Version 28 adds a public Triton-style `Config`/`autotune` API and uses CUDA
events to select a GEMM schedule for each `(M, N, K)` key. It does not introduce
a new CUDA kernel. Its job is to choose among the schedules built in Versions
22–27 without letting Python wrapper overhead distort that choice.

On the connected NVIDIA A100-SXM4-40GB, the tuner selected the smaller
`64x32x64` tile for `1024^3` and `2048^3`, then selected the Version 27 static
`128x32x128`, four-warp pipeline for `4096^3` and `8192^3`.

| BF16 shape | Selected `BMxBKxBN` | Warps | Raw TFLOP/s | torch TFLOP/s | torch ratio |
|---:|:---:|---:|---:|---:|---:|
| `1024^3` | `64x32x64` | 4 | 39.57 | 54.47 | 72.6% |
| `2048^3` | `64x32x64` | 4 | 90.20 | 108.24 | 83.3% |
| `4096^3` | `128x32x128` | 4 | 159.97 | 195.37 | 81.9% |
| `8192^3` | `128x32x128` | 4 | 210.21 | 242.38 | 86.7% |

FP16 `4096^3` selected the same large configuration and reached 158.09
TFLOP/s versus 200.92 TFLOP/s for torch, or 78.7% in that run.

Absolute A100 clocks and torch results varied materially between Colab runs.
Version 27 recorded 187.19 TFLOP/s in its final BF16 `4096^3` sweep and 202.14
TFLOP/s in its best observed run. The Version 28 result should therefore be
read primarily as evidence that the tuner chooses the right existing kernel,
not as a regression or a new kernel-level speedup. Same-run ratios are the
fairest comparison in this report.

## Candidate space

| `BMxBKxBN` | Warps | Purpose |
|:---:|---:|:---|
| `64x16x64` | 4 | small tile, shallow K slice |
| `64x32x64` | 4 | small tile, fewer loop iterations |
| `128x16x128` | 4 | large output tile, shallow K slice |
| `128x32x128` | 4 | Version 27 static pipeline |
| `128x32x128` | 8 | Version 26 dynamic pipeline control |

The exact CUDA-event medians for BF16 `4096^3` were:

| Configuration | Median kernel time |
|:---|---:|
| `64x16x64`, 4 warps | 1.6425 ms |
| `64x32x64`, 4 warps | 1.2237 ms |
| `128x16x128`, 4 warps | 1.3281 ms |
| `128x32x128`, 4 warps | **0.8458 ms** |
| `128x32x128`, 8 warps | 1.0332 ms |

For `1024^3`, the winning `64x32x64` candidate measured 0.0430 ms versus
0.0481 ms for the four-warp `128x32x128` tile. At `2048^3` the respective
times were 0.1792 and 0.1843 ms. The crossover validates a shape-sensitive
cache instead of one global default.

## Why CUDA events are required

The first implementation measured the normal public launcher. That was wrong:
the educational API returns deep copies of the expression tree and SSA after
every launch. At `1024^3`, this Python work took about 1.9–2.0 ms while the GPU
kernel took roughly 0.05 ms. Noise in artifact copying selected an eight-warp
kernel that produced only 140.10 TFLOP/s in the first `4096^3` experiment.

The final implementation compiles and launches a candidate once, obtains its
cached CuPy RawKernel, converts runtime arguments once, and records CUDA events
on the current CuPy or externally-owned PyTorch stream. Compilation, Python IR
copying, DLPack conversion, and device synchronization are outside the measured
interval. The corrected tuner selected the expected four-warp static pipeline.

## Cache semantics

The winner cache is separated by backend and CUDA architecture, CUDA device
index, inferred runtime argument types, and user-provided tuning-key values.
Thus equal dimensions on two devices or with FP16 versus BF16 operands cannot
accidentally share a decision. `clear_cache()` clears tuning decisions,
candidate timings, compilation artifacts, and CUDA kernels.

The winner cache is process-local. Cold CuPy/NVRTC compilation of all five
large generated kernels can take minutes; warm tuning after compilation took
approximately 24–95 ms for the 1024–4096 cases and 556 ms for `8192^3`. A
future version should persist winner records and prune candidates before code
generation.

## Correctness and reproduction

Each reported configuration overwrites the complete output, and the benchmark
checks sampled results against a PyTorch FP32-accumulation reference. The CUDA
execution test also tunes two real FP16 Tensor Core kernels and compares the
final output with NumPy.

```bash
./benchmarks/run_autotune_colab.sh \
    --sizes 1024 2048 4096 8192 \
    --dtype bf16 \
    --warmup 5 \
    --repeats 30
```

The JSON output contains candidate event measurements, the selected config,
cached-wrapper and raw timings, resources, correctness, and device metadata.

## Next bottleneck

Autotuning can avoid a bad existing schedule, but it cannot manufacture a
faster schedule. The large-square winner still uses the Version 27 kernel with
213 registers per thread and 12.5% theoretical occupancy. Closing the final
gap to cuBLAS requires new candidates: lower register pressure, better warp
scheduling, persistent or grouped tiles, and a cheaper public launch path.
