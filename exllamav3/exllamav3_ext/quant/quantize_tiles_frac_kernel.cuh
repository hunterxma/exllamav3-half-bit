#pragma once

// Fractional-rate trellis quantizer (experimental). The bit stream advances by KA bits into even
// positions and KB bits into odd positions of the L-weight tail-biting ring, so a tile occupies
// L * (KA + KB) / 2 bits, e.g. (1, 2) = 1.5 bits per weight. The 16-bit window and the codebooks are
// those of quantize_tiles_kernel; only the node widths alternate: the node between positions i-1
// and i is the (16 - D(i))-bit overlap of their windows, D(i) = KA for even i, KB for odd i. Every
// step still evaluates all 65536 window values, grouped as 2^(16-KOUT) out-nodes x 2^KOUT
// candidates whose in-node is the window's top 16-KIN bits. Generic path: costs in global scratch
// (like K = 1), one block of 512 threads per tile, L = 256 only.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

#define QTF_NT 512

// mul1 codebook only: the fractional rates exist for new quants, the 3INST/MCG codebooks are legacy
template <int KIN, int KOUT>
__device__ __forceinline__ void qt_frac_step
(
    const half w,
    const half* __restrict__ costs_in,
    half* __restrict__ costs_out,
    uint16_t* __restrict__ hist_row,
    const bool first,
    const int pre_state,
    const bool write_hist,
    const int thread
)
{
    // Minimum over the 2^KOUT candidates of each out-edge pair as one packed integer, (half cost bits << 16) | in-edge
    // (costs are non-negative halves, so unsigned order is cost order): branch-free, like the sm_120 integer kernels
    constexpr int Kr = 16 - KOUT;
    constexpr int max_q = 1 << KOUT;
    constexpr int edges_out = 65536 >> KOUT;
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(H_INF);
    constexpr uint32_t product_step = 0x83DCD12Du << Kr;

    for (int out_edge_idx = 2 * thread; out_edge_idx < edges_out; out_edge_idx += 2 * QTF_NT)
    {
        uint32_t product0 = out_edge_idx * 0x83DCD12Du;
        uint32_t product1 = product0 + 0x83DCD12Du;
        uint32_t best0 = 0xffffffffu, best1 = 0xffffffffu;
        #pragma unroll
        for (int k = 0; k < max_q; ++k)
        {
            const int in_edge_idx = ((k << Kr) | out_edge_idx) >> KIN;
            const half2 dh2 = __hsub2(decode_mul1_product_2(product0, product1), w2);
            product0 += product_step;
            product1 += product_step;
            half2 err2;
            if (first)
            {
                err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge_idx != pre_state) err2 = inf2;
            }
            else
                err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]));
            best0 = min(best0, ((uint32_t) __half_as_ushort(__low2half(err2)) << 16) | (uint32_t) in_edge_idx);
            best1 = min(best1, ((uint32_t) __half_as_ushort(__high2half(err2)) << 16) | (uint32_t) in_edge_idx);
        }
        reinterpret_cast<half2*>(costs_out)[out_edge_idx >> 1] =
            __halves2half2(__ushort_as_half((uint16_t) (best0 >> 16)), __ushort_as_half((uint16_t) (best1 >> 16)));
        if (write_hist)
            reinterpret_cast<uint32_t*>(hist_row)[out_edge_idx >> 1] = (best0 & 0xffffu) | (best1 << 16);
    }
}

// D(i) = KA + bit (i mod 16) of MASK: a period-16 pattern of KA- and (KA+1)-bit steps, so the rate is
// KA + popcount(MASK) / 16 in steps of 1/16 bit (MASK 0xAAAA = alternating = KA + 0.5)
template <int KA, uint32_t MASK, int L = 256>
__global__ __launch_bounds__(QTF_NT, 2)
void quantize_tiles_frac_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr,
    const half2* __restrict__ lut
)
{
    static_assert(L % 32 == 0, "the 16-position pattern must tile both halves of the ring");
    constexpr int NT = QTF_NT;
    constexpr int NW = NT / 32;
    constexpr int KB = KA + 1;
    constexpr int edges_max = 65536 >> KA;      // widest node space (out-width of the narrower shift)
    // Both passes end on position L - 1 or L / 2 - 1, whose out-width is set by D(0) (ring wrap)
    constexpr int edges_last = 65536 >> (KA + (MASK & 1));

    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint16_t* temp_edges = temp_edges_ptr + (size_t) L * edges_max * tile_idx;

    half* sh_input_tile = (half*) sh; sh += L * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);

    // Cost rows in shared memory from KA = 2 up (2 x 16384 half = 64 KB at most); KA = 1 keeps the caller's global scratch
    half* temp_costs;
    if constexpr (QTF_SMEM_COSTS(KA)) { temp_costs = (half*) sh; sh += 2 * edges_max * sizeof(half); }
    else temp_costs = temp_costs_ptr + (size_t) 2 * edges_max * tile_idx;
    half* temp_costs_inc = temp_costs + edges_max;

    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);
    __syncthreads();

    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    auto forward = [&](int roll, int pre_state)
    {
        for (int i = 0; i < L; ++i)
        {
            const int ri = ring(i, roll);
            half* t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;
            const bool first = i == 0;
            // The first pass only traces back to position 0, so its history for the rolled second
            // half (ri >= L / 2) is never read
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint16_t* hist_row = temp_edges + (size_t) edges_max * ri;
            const half w = sh_input_tile[ri];
            // KIN = D(ri), KOUT = D(ri + 1)
            const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
            switch (din * 2 + dout)
            {
                case 0: qt_frac_step<KA, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                case 1: qt_frac_step<KA, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                case 2: qt_frac_step<KB, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
                default: qt_frac_step<KB, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, thread); break;
            }
            __syncthreads();
        }
    };

    auto argmin_cost = [&]()
    {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges_last; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t) __half_as_ushort(temp_costs[e]) << 16) | rank;
            best = min(best, key);
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        if ((thread & 31) == 0)
            ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            #pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int prev_edge = (int) temp_edges[(size_t) edges_max * ri + edge];
                const int kin = KA + ((MASK >> (ri & 15)) & 1);
                const int encoded = ((prev_edge << kin) | edge) & 0xFFFF;
                edge = prev_edge;
                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_3inst<2>(encoded));
                }
                else if (ri == 0) break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    forward(L / 2, -1);
    int end_state = backward(L / 2, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}
