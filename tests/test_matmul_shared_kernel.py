from textwrap import dedent

import numpy as np
import pytest

import mytriton as triton
import mytriton.language as tl


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

            acc = acc + a_values * b_values

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
            float v78 = (true ? smem18[(v75 + 0)] : 0.0f);
            int v81 = tile_j;
            float v83 = (true ? smem54[v81] : 0.0f);
            float v87 = (true ? smem18[(v75 + 1)] : 0.0f);
            float v90 = (true ? smem54[(8 + v81)] : 0.0f);
            float v94 = (true ? smem18[(v75 + 2)] : 0.0f);
            float v97 = (true ? smem54[(16 + v81)] : 0.0f);
            float v101 = (true ? smem18[(v75 + 3)] : 0.0f);
            float v104 = (true ? smem54[(24 + v81)] : 0.0f);
            float v108 = (true ? smem18[(v75 + 4)] : 0.0f);
            float v111 = (true ? smem54[(32 + v81)] : 0.0f);
            float v115 = (true ? smem18[(v75 + 5)] : 0.0f);
            float v118 = (true ? smem54[(40 + v81)] : 0.0f);
            float v122 = (true ? smem18[(v75 + 6)] : 0.0f);
            float v125 = (true ? smem54[(48 + v81)] : 0.0f);
            float v129 = (true ? smem18[(v75 + 7)] : 0.0f);
            float v132 = (true ? smem54[(56 + v81)] : 0.0f);
            float v136 = (true ? smem18[(v75 + 8)] : 0.0f);
            float v139 = (true ? smem54[(64 + v81)] : 0.0f);
            float v143 = (true ? smem18[(v75 + 9)] : 0.0f);
            float v146 = (true ? smem54[(72 + v81)] : 0.0f);
            float v150 = (true ? smem18[(v75 + 10)] : 0.0f);
            float v153 = (true ? smem54[(80 + v81)] : 0.0f);
            float v157 = (true ? smem18[(v75 + 11)] : 0.0f);
            float v160 = (true ? smem54[(88 + v81)] : 0.0f);
            float v164 = (true ? smem18[(v75 + 12)] : 0.0f);
            float v167 = (true ? smem54[(96 + v81)] : 0.0f);
            float v171 = (true ? smem18[(v75 + 13)] : 0.0f);
            float v174 = (true ? smem54[(104 + v81)] : 0.0f);
            float v178 = (true ? smem18[(v75 + 14)] : 0.0f);
            float v181 = (true ? smem54[(112 + v81)] : 0.0f);
            float v185 = (true ? smem18[(v75 + 15)] : 0.0f);
            float v188 = (true ? smem54[(120 + v81)] : 0.0f);
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
            float v260 = (true ? smem18[(v75 + 0)] : 0.0f);
            float v263 = (true ? smem54[v81] : 0.0f);
            float v267 = (true ? smem18[(v75 + 1)] : 0.0f);
            float v270 = (true ? smem54[(8 + v81)] : 0.0f);
            float v274 = (true ? smem18[(v75 + 2)] : 0.0f);
            float v277 = (true ? smem54[(16 + v81)] : 0.0f);
            float v281 = (true ? smem18[(v75 + 3)] : 0.0f);
            float v284 = (true ? smem54[(24 + v81)] : 0.0f);
            float v288 = (true ? smem18[(v75 + 4)] : 0.0f);
            float v291 = (true ? smem54[(32 + v81)] : 0.0f);
            float v295 = (true ? smem18[(v75 + 5)] : 0.0f);
            float v298 = (true ? smem54[(40 + v81)] : 0.0f);
            float v302 = (true ? smem18[(v75 + 6)] : 0.0f);
            float v305 = (true ? smem54[(48 + v81)] : 0.0f);
            float v309 = (true ? smem18[(v75 + 7)] : 0.0f);
            float v312 = (true ? smem54[(56 + v81)] : 0.0f);
            float v316 = (true ? smem18[(v75 + 8)] : 0.0f);
            float v319 = (true ? smem54[(64 + v81)] : 0.0f);
            float v323 = (true ? smem18[(v75 + 9)] : 0.0f);
            float v326 = (true ? smem54[(72 + v81)] : 0.0f);
            float v330 = (true ? smem18[(v75 + 10)] : 0.0f);
            float v333 = (true ? smem54[(80 + v81)] : 0.0f);
            float v337 = (true ? smem18[(v75 + 11)] : 0.0f);
            float v340 = (true ? smem54[(88 + v81)] : 0.0f);
            float v344 = (true ? smem18[(v75 + 12)] : 0.0f);
            float v347 = (true ? smem54[(96 + v81)] : 0.0f);
            float v351 = (true ? smem18[(v75 + 13)] : 0.0f);
            float v354 = (true ? smem54[(104 + v81)] : 0.0f);
            float v358 = (true ? smem18[(v75 + 14)] : 0.0f);
            float v361 = (true ? smem54[(112 + v81)] : 0.0f);
            float v365 = (true ? smem18[(v75 + 15)] : 0.0f);
            float v368 = (true ? smem54[(120 + v81)] : 0.0f);
            __syncthreads();
            int v370 = (v1 + v74);
            int v371 = (v370 * N);
            int v373 = (v42 + v81);
            int v374 = (v371 + v373);
            float v375 = (v374 * 0.0f);
            float v376 = (v78 * v83);
            float v377 = (v375 + v376);
            float v378 = (v87 * v90);
            float v379 = (v377 + v378);
            float v380 = (v94 * v97);
            float v381 = (v379 + v380);
            float v382 = (v101 * v104);
            float v383 = (v381 + v382);
            float v384 = (v108 * v111);
            float v385 = (v383 + v384);
            float v386 = (v115 * v118);
            float v387 = (v385 + v386);
            float v388 = (v122 * v125);
            float v389 = (v387 + v388);
            float v390 = (v129 * v132);
            float v391 = (v389 + v390);
            float v392 = (v136 * v139);
            float v393 = (v391 + v392);
            float v394 = (v143 * v146);
            float v395 = (v393 + v394);
            float v396 = (v150 * v153);
            float v397 = (v395 + v396);
            float v398 = (v157 * v160);
            float v399 = (v397 + v398);
            float v400 = (v164 * v167);
            float v401 = (v399 + v400);
            float v402 = (v171 * v174);
            float v403 = (v401 + v402);
            float v404 = (v178 * v181);
            float v405 = (v403 + v404);
            float v406 = (v185 * v188);
            float v407 = (v405 + v406);
            float v408 = (v260 * v263);
            float v409 = (v407 + v408);
            float v410 = (v267 * v270);
            float v411 = (v409 + v410);
            float v412 = (v274 * v277);
            float v413 = (v411 + v412);
            float v414 = (v281 * v284);
            float v415 = (v413 + v414);
            float v416 = (v288 * v291);
            float v417 = (v415 + v416);
            float v418 = (v295 * v298);
            float v419 = (v417 + v418);
            float v420 = (v302 * v305);
            float v421 = (v419 + v420);
            float v422 = (v309 * v312);
            float v423 = (v421 + v422);
            float v424 = (v316 * v319);
            float v425 = (v423 + v424);
            float v426 = (v323 * v326);
            float v427 = (v425 + v426);
            float v428 = (v330 * v333);
            float v429 = (v427 + v428);
            float v430 = (v337 * v340);
            float v431 = (v429 + v430);
            float v432 = (v344 * v347);
            float v433 = (v431 + v432);
            float v434 = (v351 * v354);
            float v435 = (v433 + v434);
            float v436 = (v358 * v361);
            float v437 = (v435 + v436);
            float v438 = (v365 * v368);
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
