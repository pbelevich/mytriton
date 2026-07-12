from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl
from mytriton.ssa import SSAPrinter


@triton.jit
def matmul_2d_shared_kernel(
    a,
    b,
    c,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    local_m = tl.arange(0, BM)[:, None]
    local_n = tl.arange(0, BN)[None, :]

    offs_m = pid_m * BM + local_m
    offs_n = pid_n * BN + local_n

    c_offsets = offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)

    # Temporary explicit shared-memory hooks.
    # These are not final Triton-like API.
    a_tile = tl._shared_array((BM, BK), dtype=tl.float32)
    b_tile = tl._shared_array((BK, BN), dtype=tl.float32)

    # One rank-1 lane id per CUDA thread.
    # In rank-2 CUDA lowering this becomes threadIdx.x.
    load_idx = tl.arange(0, BM * BN)

    # Symbolic zero tile: block<BM x BN x f32>.
    acc = c_offsets * 0.0

    for k0 in tl.static_range(0, K, BK):
        # ---------------------------------
        # Cooperative load: A[BM, BK] -> shared
        # ---------------------------------
        for base in tl.static_range(0, BM * BK, BM * BN):
            a_sidx = base + load_idx

            a_local_m = a_sidx / BK
            a_local_k = a_sidx - a_local_m * BK

            a_global_m = pid_m * BM + a_local_m
            a_global_k = k0 + a_local_k

            a_global_offsets = a_global_m * K + a_global_k

            a_in_shared = a_sidx < (BM * BK)
            a_mask = a_in_shared & (a_global_m < M) & (a_global_k < K)

            a_values = tl.load(a + a_global_offsets, mask=a_mask, other=0.0)
            tl.store(a_tile + a_sidx, a_values, mask=a_in_shared)

        # ---------------------------------
        # Cooperative load: B[BK, BN] -> shared
        # ---------------------------------
        for base in tl.static_range(0, BK * BN, BM * BN):
            b_sidx = base + load_idx

            b_local_k = b_sidx / BN
            b_local_n = b_sidx - b_local_k * BN

            b_global_k = k0 + b_local_k
            b_global_n = pid_n * BN + b_local_n

            b_global_offsets = b_global_k * N + b_global_n

            b_in_shared = b_sidx < (BK * BN)
            b_mask = b_in_shared & (b_global_k < K) & (b_global_n < N)

            b_values = tl.load(b + b_global_offsets, mask=b_mask, other=0.0)
            tl.store(b_tile + b_sidx, b_values, mask=b_in_shared)

        tl._barrier()

        # ---------------------------------
        # Compute C tile from shared tiles
        # ---------------------------------
        for kk in tl.static_range(0, BK):
            a_values = tl.load(a_tile + local_m * BK + kk)
            b_values = tl.load(b_tile + kk * BN + local_n)

            acc = acc + tl.dot(a_values, b_values)

        # Important: do not let next k0 iteration overwrite shared memory
        # before all threads are done reading current shared tiles.
        tl._barrier()

    tl.store(c + c_offsets, acc, mask=c_mask)


@pytest.mark.codegen
def test_matmul_2d_shared_generates_cuda_source_without_execution(monkeypatch):
    monkeypatch.setenv("MYTRITON_BACKEND", "cuda")

    M, N, K = 17, 19, 20
    BM, BN, BK = 8, 8, 16

    a = np.zeros((M, K), dtype=np.float32)
    b = np.zeros((K, N), dtype=np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    matmul_2d_shared_kernel.clear_cache()
    _, ssa_ops, cuda_src = matmul_2d_shared_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN, BK=BK)

    assert sum(op.opcode == "shared_alloc" for op in ssa_ops) == 2
    assert any(op.opcode == "barrier" for op in ssa_ops)

    expected_ssa = dedent(
        """\
        %0 = program_id {axis=0} : i32
        %1 = mul %0, 8 : i32
        %2 = arange {start=0, end=64} : vector<64 x i32>
        %3 = add 0, %2 : vector<64 x i32>
        %4 = div %3, 16 : vector<64 x i32>
        %5 = add %1, %4 : vector<64 x i32>
        %6 = mul %5, 20 : vector<64 x i32>
        %7 = mul %4, 16 : vector<64 x i32>
        %8 = sub %3, %7 : vector<64 x i32>
        %9 = add 0, %8 : vector<64 x i32>
        %10 = add %6, %9 : vector<64 x i32>
        %11 = addptr a, %10 : vector<64 x ptr<f32>>
        %12 = cmp_lt %3, 128 : vector<64 x bool>
        %13 = cmp_lt %5, M : vector<64 x bool>
        %14 = and %12, %13 : vector<64 x bool>
        %15 = cmp_lt %9, 20 : vector<64 x bool>
        %16 = and %14, %15 : vector<64 x bool>
        %17 = load %11, %16, 0.0 : vector<64 x f32>
        %18 = shared_alloc {shape=(8, 16), dtype=f32} : ptr<shared, f32>
        %19 = addptr %18, %3 : vector<64 x ptr<shared, f32>>
        store %19, %17, %12
        %21 = add 64, %2 : vector<64 x i32>
        %22 = div %21, 16 : vector<64 x i32>
        %23 = add %1, %22 : vector<64 x i32>
        %24 = mul %23, 20 : vector<64 x i32>
        %25 = mul %22, 16 : vector<64 x i32>
        %26 = sub %21, %25 : vector<64 x i32>
        %27 = add 0, %26 : vector<64 x i32>
        %28 = add %24, %27 : vector<64 x i32>
        %29 = addptr a, %28 : vector<64 x ptr<f32>>
        %30 = cmp_lt %21, 128 : vector<64 x bool>
        %31 = cmp_lt %23, M : vector<64 x bool>
        %32 = and %30, %31 : vector<64 x bool>
        %33 = cmp_lt %27, 20 : vector<64 x bool>
        %34 = and %32, %33 : vector<64 x bool>
        %35 = load %29, %34, 0.0 : vector<64 x f32>
        %36 = addptr %18, %21 : vector<64 x ptr<shared, f32>>
        store %36, %35, %30
        %38 = div %3, 8 : vector<64 x i32>
        %39 = add 0, %38 : vector<64 x i32>
        %40 = mul %39, N : vector<64 x i32>
        %41 = program_id {axis=1} : i32
        %42 = mul %41, 8 : i32
        %43 = mul %38, 8 : vector<64 x i32>
        %44 = sub %3, %43 : vector<64 x i32>
        %45 = add %42, %44 : vector<64 x i32>
        %46 = add %40, %45 : vector<64 x i32>
        %47 = addptr b, %46 : vector<64 x ptr<f32>>
        %49 = cmp_lt %39, 20 : vector<64 x bool>
        %50 = and %12, %49 : vector<64 x bool>
        %51 = cmp_lt %45, N : vector<64 x bool>
        %52 = and %50, %51 : vector<64 x bool>
        %53 = load %47, %52, 0.0 : vector<64 x f32>
        %54 = shared_alloc {shape=(16, 8), dtype=f32} : ptr<shared, f32>
        %55 = addptr %54, %3 : vector<64 x ptr<shared, f32>>
        store %55, %53, %12
        %57 = div %21, 8 : vector<64 x i32>
        %58 = add 0, %57 : vector<64 x i32>
        %59 = mul %58, N : vector<64 x i32>
        %61 = mul %57, 8 : vector<64 x i32>
        %62 = sub %21, %61 : vector<64 x i32>
        %63 = add %42, %62 : vector<64 x i32>
        %64 = add %59, %63 : vector<64 x i32>
        %65 = addptr b, %64 : vector<64 x ptr<f32>>
        %67 = cmp_lt %58, 20 : vector<64 x bool>
        %68 = and %30, %67 : vector<64 x bool>
        %69 = cmp_lt %63, N : vector<64 x bool>
        %70 = and %68, %69 : vector<64 x bool>
        %71 = load %65, %70, 0.0 : vector<64 x f32>
        %72 = addptr %54, %21 : vector<64 x ptr<shared, f32>>
        store %72, %71, %30
        barrier
        %73 = arange {start=0, end=8} : vector<8 x i32>
        %74 = expand_dims %73 {axis=1} : block<8x1 x i32>
        %75 = mul %74, 16 : block<8x1 x i32>
        %76 = addptr %18, %75 : block<8x1 x ptr<shared, f32>>
        %77 = addptr %76, 0 : block<8x1 x ptr<shared, f32>>
        %78 = load %77, none, none : block<8x1 x f32>
        %79 = addptr %54, 0 : ptr<shared, f32>
        %81 = expand_dims %73 {axis=0} : block<1x8 x i32>
        %82 = addptr %79, %81 : block<1x8 x ptr<shared, f32>>
        %83 = load %82, none, none : block<1x8 x f32>
        %86 = addptr %76, 1 : block<8x1 x ptr<shared, f32>>
        %87 = load %86, none, none : block<8x1 x f32>
        %88 = addptr %54, 8 : ptr<shared, f32>
        %89 = addptr %88, %81 : block<1x8 x ptr<shared, f32>>
        %90 = load %89, none, none : block<1x8 x f32>
        %93 = addptr %76, 2 : block<8x1 x ptr<shared, f32>>
        %94 = load %93, none, none : block<8x1 x f32>
        %95 = addptr %54, 16 : ptr<shared, f32>
        %96 = addptr %95, %81 : block<1x8 x ptr<shared, f32>>
        %97 = load %96, none, none : block<1x8 x f32>
        %100 = addptr %76, 3 : block<8x1 x ptr<shared, f32>>
        %101 = load %100, none, none : block<8x1 x f32>
        %102 = addptr %54, 24 : ptr<shared, f32>
        %103 = addptr %102, %81 : block<1x8 x ptr<shared, f32>>
        %104 = load %103, none, none : block<1x8 x f32>
        %107 = addptr %76, 4 : block<8x1 x ptr<shared, f32>>
        %108 = load %107, none, none : block<8x1 x f32>
        %109 = addptr %54, 32 : ptr<shared, f32>
        %110 = addptr %109, %81 : block<1x8 x ptr<shared, f32>>
        %111 = load %110, none, none : block<1x8 x f32>
        %114 = addptr %76, 5 : block<8x1 x ptr<shared, f32>>
        %115 = load %114, none, none : block<8x1 x f32>
        %116 = addptr %54, 40 : ptr<shared, f32>
        %117 = addptr %116, %81 : block<1x8 x ptr<shared, f32>>
        %118 = load %117, none, none : block<1x8 x f32>
        %121 = addptr %76, 6 : block<8x1 x ptr<shared, f32>>
        %122 = load %121, none, none : block<8x1 x f32>
        %123 = addptr %54, 48 : ptr<shared, f32>
        %124 = addptr %123, %81 : block<1x8 x ptr<shared, f32>>
        %125 = load %124, none, none : block<1x8 x f32>
        %128 = addptr %76, 7 : block<8x1 x ptr<shared, f32>>
        %129 = load %128, none, none : block<8x1 x f32>
        %130 = addptr %54, 56 : ptr<shared, f32>
        %131 = addptr %130, %81 : block<1x8 x ptr<shared, f32>>
        %132 = load %131, none, none : block<1x8 x f32>
        %135 = addptr %76, 8 : block<8x1 x ptr<shared, f32>>
        %136 = load %135, none, none : block<8x1 x f32>
        %137 = addptr %54, 64 : ptr<shared, f32>
        %138 = addptr %137, %81 : block<1x8 x ptr<shared, f32>>
        %139 = load %138, none, none : block<1x8 x f32>
        %142 = addptr %76, 9 : block<8x1 x ptr<shared, f32>>
        %143 = load %142, none, none : block<8x1 x f32>
        %144 = addptr %54, 72 : ptr<shared, f32>
        %145 = addptr %144, %81 : block<1x8 x ptr<shared, f32>>
        %146 = load %145, none, none : block<1x8 x f32>
        %149 = addptr %76, 10 : block<8x1 x ptr<shared, f32>>
        %150 = load %149, none, none : block<8x1 x f32>
        %151 = addptr %54, 80 : ptr<shared, f32>
        %152 = addptr %151, %81 : block<1x8 x ptr<shared, f32>>
        %153 = load %152, none, none : block<1x8 x f32>
        %156 = addptr %76, 11 : block<8x1 x ptr<shared, f32>>
        %157 = load %156, none, none : block<8x1 x f32>
        %158 = addptr %54, 88 : ptr<shared, f32>
        %159 = addptr %158, %81 : block<1x8 x ptr<shared, f32>>
        %160 = load %159, none, none : block<1x8 x f32>
        %163 = addptr %76, 12 : block<8x1 x ptr<shared, f32>>
        %164 = load %163, none, none : block<8x1 x f32>
        %165 = addptr %54, 96 : ptr<shared, f32>
        %166 = addptr %165, %81 : block<1x8 x ptr<shared, f32>>
        %167 = load %166, none, none : block<1x8 x f32>
        %170 = addptr %76, 13 : block<8x1 x ptr<shared, f32>>
        %171 = load %170, none, none : block<8x1 x f32>
        %172 = addptr %54, 104 : ptr<shared, f32>
        %173 = addptr %172, %81 : block<1x8 x ptr<shared, f32>>
        %174 = load %173, none, none : block<1x8 x f32>
        %177 = addptr %76, 14 : block<8x1 x ptr<shared, f32>>
        %178 = load %177, none, none : block<8x1 x f32>
        %179 = addptr %54, 112 : ptr<shared, f32>
        %180 = addptr %179, %81 : block<1x8 x ptr<shared, f32>>
        %181 = load %180, none, none : block<1x8 x f32>
        %184 = addptr %76, 15 : block<8x1 x ptr<shared, f32>>
        %185 = load %184, none, none : block<8x1 x f32>
        %186 = addptr %54, 120 : ptr<shared, f32>
        %187 = addptr %186, %81 : block<1x8 x ptr<shared, f32>>
        %188 = load %187, none, none : block<1x8 x f32>
        barrier
        %196 = add 16, %8 : vector<64 x i32>
        %197 = add %6, %196 : vector<64 x i32>
        %198 = addptr a, %197 : vector<64 x ptr<f32>>
        %202 = cmp_lt %196, 20 : vector<64 x bool>
        %203 = and %14, %202 : vector<64 x bool>
        %204 = load %198, %203, 0.0 : vector<64 x f32>
        store %19, %204, %12
        %213 = add 16, %26 : vector<64 x i32>
        %214 = add %24, %213 : vector<64 x i32>
        %215 = addptr a, %214 : vector<64 x ptr<f32>>
        %219 = cmp_lt %213, 20 : vector<64 x bool>
        %220 = and %32, %219 : vector<64 x bool>
        %221 = load %215, %220, 0.0 : vector<64 x f32>
        store %36, %221, %30
        %225 = add 16, %38 : vector<64 x i32>
        %226 = mul %225, N : vector<64 x i32>
        %231 = add %226, %45 : vector<64 x i32>
        %232 = addptr b, %231 : vector<64 x ptr<f32>>
        %234 = cmp_lt %225, 20 : vector<64 x bool>
        %235 = and %12, %234 : vector<64 x bool>
        %237 = and %235, %51 : vector<64 x bool>
        %238 = load %232, %237, 0.0 : vector<64 x f32>
        store %55, %238, %12
        %242 = add 16, %57 : vector<64 x i32>
        %243 = mul %242, N : vector<64 x i32>
        %248 = add %243, %63 : vector<64 x i32>
        %249 = addptr b, %248 : vector<64 x ptr<f32>>
        %251 = cmp_lt %242, 20 : vector<64 x bool>
        %252 = and %30, %251 : vector<64 x bool>
        %254 = and %252, %69 : vector<64 x bool>
        %255 = load %249, %254, 0.0 : vector<64 x f32>
        store %72, %255, %30
        barrier
        %260 = load %77, none, none : block<8x1 x f32>
        %263 = load %82, none, none : block<1x8 x f32>
        %267 = load %86, none, none : block<8x1 x f32>
        %270 = load %89, none, none : block<1x8 x f32>
        %274 = load %93, none, none : block<8x1 x f32>
        %277 = load %96, none, none : block<1x8 x f32>
        %281 = load %100, none, none : block<8x1 x f32>
        %284 = load %103, none, none : block<1x8 x f32>
        %288 = load %107, none, none : block<8x1 x f32>
        %291 = load %110, none, none : block<1x8 x f32>
        %295 = load %114, none, none : block<8x1 x f32>
        %298 = load %117, none, none : block<1x8 x f32>
        %302 = load %121, none, none : block<8x1 x f32>
        %305 = load %124, none, none : block<1x8 x f32>
        %309 = load %128, none, none : block<8x1 x f32>
        %312 = load %131, none, none : block<1x8 x f32>
        %316 = load %135, none, none : block<8x1 x f32>
        %319 = load %138, none, none : block<1x8 x f32>
        %323 = load %142, none, none : block<8x1 x f32>
        %326 = load %145, none, none : block<1x8 x f32>
        %330 = load %149, none, none : block<8x1 x f32>
        %333 = load %152, none, none : block<1x8 x f32>
        %337 = load %156, none, none : block<8x1 x f32>
        %340 = load %159, none, none : block<1x8 x f32>
        %344 = load %163, none, none : block<8x1 x f32>
        %347 = load %166, none, none : block<1x8 x f32>
        %351 = load %170, none, none : block<8x1 x f32>
        %354 = load %173, none, none : block<1x8 x f32>
        %358 = load %177, none, none : block<8x1 x f32>
        %361 = load %180, none, none : block<1x8 x f32>
        %365 = load %184, none, none : block<8x1 x f32>
        %368 = load %187, none, none : block<1x8 x f32>
        barrier
        %370 = add %1, %74 : block<8x1 x i32>
        %371 = mul %370, N : block<8x1 x i32>
        %373 = add %42, %81 : block<1x8 x i32>
        %374 = add %371, %373 : block<8x8 x i32>
        %375 = mul %374, 0.0 : block<8x8 x f32>
        %376 = dot %78, %83 : block<8x8 x f32>
        %377 = add %375, %376 : block<8x8 x f32>
        %378 = dot %87, %90 : block<8x8 x f32>
        %379 = add %377, %378 : block<8x8 x f32>
        %380 = dot %94, %97 : block<8x8 x f32>
        %381 = add %379, %380 : block<8x8 x f32>
        %382 = dot %101, %104 : block<8x8 x f32>
        %383 = add %381, %382 : block<8x8 x f32>
        %384 = dot %108, %111 : block<8x8 x f32>
        %385 = add %383, %384 : block<8x8 x f32>
        %386 = dot %115, %118 : block<8x8 x f32>
        %387 = add %385, %386 : block<8x8 x f32>
        %388 = dot %122, %125 : block<8x8 x f32>
        %389 = add %387, %388 : block<8x8 x f32>
        %390 = dot %129, %132 : block<8x8 x f32>
        %391 = add %389, %390 : block<8x8 x f32>
        %392 = dot %136, %139 : block<8x8 x f32>
        %393 = add %391, %392 : block<8x8 x f32>
        %394 = dot %143, %146 : block<8x8 x f32>
        %395 = add %393, %394 : block<8x8 x f32>
        %396 = dot %150, %153 : block<8x8 x f32>
        %397 = add %395, %396 : block<8x8 x f32>
        %398 = dot %157, %160 : block<8x8 x f32>
        %399 = add %397, %398 : block<8x8 x f32>
        %400 = dot %164, %167 : block<8x8 x f32>
        %401 = add %399, %400 : block<8x8 x f32>
        %402 = dot %171, %174 : block<8x8 x f32>
        %403 = add %401, %402 : block<8x8 x f32>
        %404 = dot %178, %181 : block<8x8 x f32>
        %405 = add %403, %404 : block<8x8 x f32>
        %406 = dot %185, %188 : block<8x8 x f32>
        %407 = add %405, %406 : block<8x8 x f32>
        %408 = dot %260, %263 : block<8x8 x f32>
        %409 = add %407, %408 : block<8x8 x f32>
        %410 = dot %267, %270 : block<8x8 x f32>
        %411 = add %409, %410 : block<8x8 x f32>
        %412 = dot %274, %277 : block<8x8 x f32>
        %413 = add %411, %412 : block<8x8 x f32>
        %414 = dot %281, %284 : block<8x8 x f32>
        %415 = add %413, %414 : block<8x8 x f32>
        %416 = dot %288, %291 : block<8x8 x f32>
        %417 = add %415, %416 : block<8x8 x f32>
        %418 = dot %295, %298 : block<8x8 x f32>
        %419 = add %417, %418 : block<8x8 x f32>
        %420 = dot %302, %305 : block<8x8 x f32>
        %421 = add %419, %420 : block<8x8 x f32>
        %422 = dot %309, %312 : block<8x8 x f32>
        %423 = add %421, %422 : block<8x8 x f32>
        %424 = dot %316, %319 : block<8x8 x f32>
        %425 = add %423, %424 : block<8x8 x f32>
        %426 = dot %323, %326 : block<8x8 x f32>
        %427 = add %425, %426 : block<8x8 x f32>
        %428 = dot %330, %333 : block<8x8 x f32>
        %429 = add %427, %428 : block<8x8 x f32>
        %430 = dot %337, %340 : block<8x8 x f32>
        %431 = add %429, %430 : block<8x8 x f32>
        %432 = dot %344, %347 : block<8x8 x f32>
        %433 = add %431, %432 : block<8x8 x f32>
        %434 = dot %351, %354 : block<8x8 x f32>
        %435 = add %433, %434 : block<8x8 x f32>
        %436 = dot %358, %361 : block<8x8 x f32>
        %437 = add %435, %436 : block<8x8 x f32>
        %438 = dot %365, %368 : block<8x8 x f32>
        %439 = add %437, %438 : block<8x8 x f32>
        %440 = addptr c, %374 : block<8x8 x ptr<f32>>
        %441 = cmp_lt %370, M : block<8x1 x bool>
        %442 = cmp_lt %373, N : block<1x8 x bool>
        %443 = and %441, %442 : block<8x8 x bool>
        store %440, %439, %443
        """
    ).rstrip("\n")

    assert SSAPrinter().print_ops(ssa_ops) == expected_ssa

    expected_cuda_src = dedent(
        """\
        extern "C" __global__
        void matmul_2d_shared_kernel(float* a, float* b, float* c, int M, int N) {
            __shared__ float smem18[128];
            __shared__ float smem54[128];

            int tile_i = threadIdx.x / 8;
            int tile_j = threadIdx.x % 8;
            int v0 = blockIdx.x;
            int v1 = (v0 * 8);
            int v3 = (0 + threadIdx.x);
            int v4 = (v3 / 16);
            int v5 = (v1 + v4);
            int v6 = (v5 * 20);
            int v7 = (v4 * 16);
            int v8 = (v3 - v7);
            int v9 = (0 + v8);
            int v10 = (v6 + v9);
            bool v12 = (v3 < 128);
            bool v13 = (v5 < M);
            bool v14 = (v12 && v13);
            bool v15 = (v9 < 20);
            bool v16 = (v14 && v15);
            float v17 = (v16 ? a[v10] : 0.0f);
            if (v12) {
                smem18[v3] = v17;
            }
            int v21 = (64 + threadIdx.x);
            int v22 = (v21 / 16);
            int v23 = (v1 + v22);
            int v24 = (v23 * 20);
            int v25 = (v22 * 16);
            int v26 = (v21 - v25);
            int v27 = (0 + v26);
            int v28 = (v24 + v27);
            bool v30 = (v21 < 128);
            bool v31 = (v23 < M);
            bool v32 = (v30 && v31);
            bool v33 = (v27 < 20);
            bool v34 = (v32 && v33);
            float v35 = (v34 ? a[v28] : 0.0f);
            if (v30) {
                smem18[v21] = v35;
            }
            int v38 = (v3 / 8);
            int v39 = (0 + v38);
            int v40 = (v39 * N);
            int v41 = blockIdx.y;
            int v42 = (v41 * 8);
            int v43 = (v38 * 8);
            int v44 = (v3 - v43);
            int v45 = (v42 + v44);
            int v46 = (v40 + v45);
            bool v49 = (v39 < 20);
            bool v50 = (v12 && v49);
            bool v51 = (v45 < N);
            bool v52 = (v50 && v51);
            float v53 = (v52 ? b[v46] : 0.0f);
            if (v12) {
                smem54[v3] = v53;
            }
            int v57 = (v21 / 8);
            int v58 = (0 + v57);
            int v59 = (v58 * N);
            int v61 = (v57 * 8);
            int v62 = (v21 - v61);
            int v63 = (v42 + v62);
            int v64 = (v59 + v63);
            bool v67 = (v58 < 20);
            bool v68 = (v30 && v67);
            bool v69 = (v63 < N);
            bool v70 = (v68 && v69);
            float v71 = (v70 ? b[v64] : 0.0f);
            if (v30) {
                smem54[v21] = v71;
            }
            __syncthreads();
            int v74 = tile_i;
            int v75 = (v74 * 16);
            float v78 = smem18[(v75 + 0)];
            int v81 = tile_j;
            float v83 = smem54[v81];
            float v87 = smem18[(v75 + 1)];
            float v90 = smem54[(8 + v81)];
            float v94 = smem18[(v75 + 2)];
            float v97 = smem54[(16 + v81)];
            float v101 = smem18[(v75 + 3)];
            float v104 = smem54[(24 + v81)];
            float v108 = smem18[(v75 + 4)];
            float v111 = smem54[(32 + v81)];
            float v115 = smem18[(v75 + 5)];
            float v118 = smem54[(40 + v81)];
            float v122 = smem18[(v75 + 6)];
            float v125 = smem54[(48 + v81)];
            float v129 = smem18[(v75 + 7)];
            float v132 = smem54[(56 + v81)];
            float v136 = smem18[(v75 + 8)];
            float v139 = smem54[(64 + v81)];
            float v143 = smem18[(v75 + 9)];
            float v146 = smem54[(72 + v81)];
            float v150 = smem18[(v75 + 10)];
            float v153 = smem54[(80 + v81)];
            float v157 = smem18[(v75 + 11)];
            float v160 = smem54[(88 + v81)];
            float v164 = smem18[(v75 + 12)];
            float v167 = smem54[(96 + v81)];
            float v171 = smem18[(v75 + 13)];
            float v174 = smem54[(104 + v81)];
            float v178 = smem18[(v75 + 14)];
            float v181 = smem54[(112 + v81)];
            float v185 = smem18[(v75 + 15)];
            float v188 = smem54[(120 + v81)];
            __syncthreads();
            int v196 = (16 + v8);
            int v197 = (v6 + v196);
            bool v202 = (v196 < 20);
            bool v203 = (v14 && v202);
            float v204 = (v203 ? a[v197] : 0.0f);
            if (v12) {
                smem18[v3] = v204;
            }
            int v213 = (16 + v26);
            int v214 = (v24 + v213);
            bool v219 = (v213 < 20);
            bool v220 = (v32 && v219);
            float v221 = (v220 ? a[v214] : 0.0f);
            if (v30) {
                smem18[v21] = v221;
            }
            int v225 = (16 + v38);
            int v226 = (v225 * N);
            int v231 = (v226 + v45);
            bool v234 = (v225 < 20);
            bool v235 = (v12 && v234);
            bool v237 = (v235 && v51);
            float v238 = (v237 ? b[v231] : 0.0f);
            if (v12) {
                smem54[v3] = v238;
            }
            int v242 = (16 + v57);
            int v243 = (v242 * N);
            int v248 = (v243 + v63);
            bool v251 = (v242 < 20);
            bool v252 = (v30 && v251);
            bool v254 = (v252 && v69);
            float v255 = (v254 ? b[v248] : 0.0f);
            if (v30) {
                smem54[v21] = v255;
            }
            __syncthreads();
            float v260 = smem18[(v75 + 0)];
            float v263 = smem54[v81];
            float v267 = smem18[(v75 + 1)];
            float v270 = smem54[(8 + v81)];
            float v274 = smem18[(v75 + 2)];
            float v277 = smem54[(16 + v81)];
            float v281 = smem18[(v75 + 3)];
            float v284 = smem54[(24 + v81)];
            float v288 = smem18[(v75 + 4)];
            float v291 = smem54[(32 + v81)];
            float v295 = smem18[(v75 + 5)];
            float v298 = smem54[(40 + v81)];
            float v302 = smem18[(v75 + 6)];
            float v305 = smem54[(48 + v81)];
            float v309 = smem18[(v75 + 7)];
            float v312 = smem54[(56 + v81)];
            float v316 = smem18[(v75 + 8)];
            float v319 = smem54[(64 + v81)];
            float v323 = smem18[(v75 + 9)];
            float v326 = smem54[(72 + v81)];
            float v330 = smem18[(v75 + 10)];
            float v333 = smem54[(80 + v81)];
            float v337 = smem18[(v75 + 11)];
            float v340 = smem54[(88 + v81)];
            float v344 = smem18[(v75 + 12)];
            float v347 = smem54[(96 + v81)];
            float v351 = smem18[(v75 + 13)];
            float v354 = smem54[(104 + v81)];
            float v358 = smem18[(v75 + 14)];
            float v361 = smem54[(112 + v81)];
            float v365 = smem18[(v75 + 15)];
            float v368 = smem54[(120 + v81)];
            __syncthreads();
            int v370 = (v1 + v74);
            int v371 = (v370 * N);
            int v373 = (v42 + v81);
            int v374 = (v371 + v373);
            float v375 = (v374 * 0.0f);
            float v376 = 0.0f;
            v376 += (v78 * v83);
            float v377 = (v375 + v376);
            float v378 = 0.0f;
            v378 += (v87 * v90);
            float v379 = (v377 + v378);
            float v380 = 0.0f;
            v380 += (v94 * v97);
            float v381 = (v379 + v380);
            float v382 = 0.0f;
            v382 += (v101 * v104);
            float v383 = (v381 + v382);
            float v384 = 0.0f;
            v384 += (v108 * v111);
            float v385 = (v383 + v384);
            float v386 = 0.0f;
            v386 += (v115 * v118);
            float v387 = (v385 + v386);
            float v388 = 0.0f;
            v388 += (v122 * v125);
            float v389 = (v387 + v388);
            float v390 = 0.0f;
            v390 += (v129 * v132);
            float v391 = (v389 + v390);
            float v392 = 0.0f;
            v392 += (v136 * v139);
            float v393 = (v391 + v392);
            float v394 = 0.0f;
            v394 += (v143 * v146);
            float v395 = (v393 + v394);
            float v396 = 0.0f;
            v396 += (v150 * v153);
            float v397 = (v395 + v396);
            float v398 = 0.0f;
            v398 += (v157 * v160);
            float v399 = (v397 + v398);
            float v400 = 0.0f;
            v400 += (v164 * v167);
            float v401 = (v399 + v400);
            float v402 = 0.0f;
            v402 += (v171 * v174);
            float v403 = (v401 + v402);
            float v404 = 0.0f;
            v404 += (v178 * v181);
            float v405 = (v403 + v404);
            float v406 = 0.0f;
            v406 += (v185 * v188);
            float v407 = (v405 + v406);
            float v408 = 0.0f;
            v408 += (v260 * v263);
            float v409 = (v407 + v408);
            float v410 = 0.0f;
            v410 += (v267 * v270);
            float v411 = (v409 + v410);
            float v412 = 0.0f;
            v412 += (v274 * v277);
            float v413 = (v411 + v412);
            float v414 = 0.0f;
            v414 += (v281 * v284);
            float v415 = (v413 + v414);
            float v416 = 0.0f;
            v416 += (v288 * v291);
            float v417 = (v415 + v416);
            float v418 = 0.0f;
            v418 += (v295 * v298);
            float v419 = (v417 + v418);
            float v420 = 0.0f;
            v420 += (v302 * v305);
            float v421 = (v419 + v420);
            float v422 = 0.0f;
            v422 += (v309 * v312);
            float v423 = (v421 + v422);
            float v424 = 0.0f;
            v424 += (v316 * v319);
            float v425 = (v423 + v424);
            float v426 = 0.0f;
            v426 += (v323 * v326);
            float v427 = (v425 + v426);
            float v428 = 0.0f;
            v428 += (v330 * v333);
            float v429 = (v427 + v428);
            float v430 = 0.0f;
            v430 += (v337 * v340);
            float v431 = (v429 + v430);
            float v432 = 0.0f;
            v432 += (v344 * v347);
            float v433 = (v431 + v432);
            float v434 = 0.0f;
            v434 += (v351 * v354);
            float v435 = (v433 + v434);
            float v436 = 0.0f;
            v436 += (v358 * v361);
            float v437 = (v435 + v436);
            float v438 = 0.0f;
            v438 += (v365 * v368);
            float v439 = (v437 + v438);
            bool v441 = (v370 < M);
            bool v442 = (v373 < N);
            bool v443 = (v441 && v442);
            if (v443) {
                c[v374] = v439;
            }
        }
        """
    ).rstrip("\n")

    assert cuda_src == expected_cuda_src


def test_matmul_2d_shared_executes_with_cupy_when_cuda_is_available():
    cp = pytest.importorskip("cupy")

    if not triton.cuda_available():
        pytest.skip("CUDA GPU is not available")

    M, N, K = 17, 19, 20
    BM, BN, BK = 8, 8, 16

    a = cp.arange(M * K, dtype=cp.float32).reshape(M, K)
    b = cp.arange(K * N, dtype=cp.float32).reshape(K, N)
    c = cp.zeros((M, N), dtype=cp.float32)

    matmul_2d_shared_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )
    ](a, b, c, M, N, K=K, BM=BM, BN=BN, BK=BK)

    cp.cuda.Stream.null.synchronize()

    expected = a @ b
    cp.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)
