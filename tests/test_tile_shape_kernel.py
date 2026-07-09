import inspect
from textwrap import dedent

import mytriton.language as tl
from mytriton.ssa import SSALowering, SSAPrinter
from mytriton.ssa_verification import SSAVerifier
from mytriton.trace import I32, PTR_F32, Param, trace


def tile_shape_kernel(out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    offs_m = tl.arange(0, BM)[:, None]
    offs_n = tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    tl.store(out + offsets, offsets, mask=mask)


def test_tile_shape_kernel_lowers_rank2_offsets_without_cuda_execution():
    signature = inspect.signature(tile_shape_kernel)
    bound = signature.bind(object(), 7, 11, BM=16, BN=32)
    params = [
        Param("out", PTR_F32),
        Param("M", I32),
        Param("N", I32),
    ]

    ops, _ = trace(
        tile_shape_kernel,
        signature,
        bound.arguments,
        runtime_params=params,
    )
    ssa_ops = SSALowering().lower(ops)

    SSAVerifier(block_size=16 * 32).verify(ssa_ops)

    assert SSAPrinter().print_ops(ssa_ops) == dedent(
        """\
        %0 = arange {start=0, end=16} : vector<16 x i32>
        %1 = expand_dims %0 {axis=1} : block<16x1 x i32>
        %2 = mul %1, N : block<16x1 x i32>
        %3 = arange {start=0, end=32} : vector<32 x i32>
        %4 = expand_dims %3 {axis=0} : block<1x32 x i32>
        %5 = add %2, %4 : block<16x32 x i32>
        %6 = addptr out, %5 : block<16x32 x ptr<f32>>
        %7 = cmp_lt %1, M : block<16x1 x bool>
        %8 = cmp_lt %4, N : block<1x32 x bool>
        %9 = and %7, %8 : block<16x32 x bool>
        store %6, %5, %9
        """
    ).rstrip("\n")
