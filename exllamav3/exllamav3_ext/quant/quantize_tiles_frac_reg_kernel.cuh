#pragma once

// Fractional-rate trellis quantizer, register-resident variant of quantize_tiles_frac_kernel (same search, same
// result bit for bit). The generic kernel decodes every one of the 65536 window values at every step, and the
// lane that scans a window depends on the step's out-width, so nothing can be kept. Here every lane owns a fixed
// set of 32 window pairs (w, w + 1) that serves both out-widths of the ring, so the codebook values are decoded
// once per tile into registers (like the integer kernels' K >= 2 paths), and a step only adds costs and takes
// minima. One block per SM (qtfr_threads).
//
// Window w = (kA << (16 - KA)) | oA: oA is the KA-step out-node (16 - KA bits), its top bit tb = bit 15 - KA, and
// the KB-step (KB = KA + 1) out-node is oA without that bit. A KA-step reduces over the windows of one oA pair, a
// KB-step over those of two oA pairs that differ only in tb.
//  - KA >= 5 (rates 5.5 .. 7.5): an oA pair has 2^KA >= 32 candidates; GA = 2^(KA - 5) lanes share it, lane
//    holding kA = sub + GA * j, j = 0..31. Lane id = (pbase << (log2 GA + 1)) | (tb << log2 GA) | sub, so a KA-step
//    reduces over the GA lanes of a sub group and a KB-step over 2 GA lanes (shuffles within a warp).
//  - KA <= 4 (rates 2.5 .. 4.5): a lane holds NPB = 2^(4 - KA) pbases, both tb, all 2^KA kA: every reduction is
//    in-lane.
// min() of the packed (cost << 16 | in-edge) keys is order independent, so any grouping gives the same minima.

#include <cuda_fp16.h>
#include <type_traits>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

// NT threads per block, one block per SM: 1024 threads hold 32 window pairs per lane, 512 hold 64 (more registers
// per lane, half the warps). The host picks NT per rate (quantize.cu, measured with science/frac_reg_test.py)
template <int KA, uint32_t MASK, int NT, int L = 256>
__global__ __launch_bounds__(NT, 1)
void quantize_tiles_frac_reg_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr,
    const half2* __restrict__ lut
)
{
    static_assert(KA >= 2 && KA <= 7, "cost rows in shared memory (KA >= 2), lane groups within a warp (KA <= 9)");
    static_assert(L % 32 == 0, "the 16-position pattern must tile both halves of the ring");
    static_assert(NT == 512 || NT == 1024);
    constexpr int NW = NT / 32;
    constexpr int PPL = 32768 / NT;                 // window pairs per lane
    constexpr int LPPL = NT == 1024 ? 5 : 6;
    constexpr int KB = KA + 1;
    constexpr int edges_max = 65536 >> KA;
    constexpr int edges_last = 65536 >> (KA + (MASK & 1));
    constexpr int TBS = 15 - KA;                    // bit of oA that the KB out-node drops
    constexpr bool WIDE = KA >= LPPL;
    constexpr int GA = WIDE ? (1 << (KA - LPPL)) : 1;       // lanes per oA pair
    constexpr int LGA = WIDE ? KA - LPPL : 0;
    constexpr int NPB = WIDE ? 1 : (1 << (LPPL - 1 - KA));  // pbases per lane
    constexpr int NKA = WIDE ? PPL : (1 << KA);              // kA values per (pbase, tb) in this lane
    static_assert(NPB * (WIDE ? 1 : 2) * NKA == PPL, "PPL window pairs per lane");
    static_assert(2 * GA <= 32, "lane groups within a warp");

    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    // Traceback history: one byte per out-edge, the winning candidate's top KOUT window bits (the in-edge follows
    // from it and the out-edge), half the stores of the generic kernel's 16-bit in-edges. The caller's 16-bit
    // scratch is used as bytes
    uint8_t* temp_edges = reinterpret_cast<uint8_t*>(temp_edges_ptr) + (size_t) L * edges_max * tile_idx;

    half* sh_input_tile = (half*) sh; sh += L * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);
    half* temp_costs = (half*) sh; sh += 2 * edges_max * sizeof(half);
    half* temp_costs_inc = temp_costs + edges_max;

    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);

    // This lane's windows. WIDE: one oA pair (pbase, tb), kA = sub + GA * j. Else NPB pbases x tb x all kA
    const int sub = WIDE ? thread & (GA - 1) : 0;
    const int tb_lane = WIDE ? (thread >> LGA) & 1 : 0;
    const int pbase0 = WIDE ? thread >> (LGA + 1) : thread * NPB;
    auto window = [&](int b, int t, int j) -> uint32_t   // j < NKA
    {
        const int kA = WIDE ? sub + GA * j : j;
        const int tb = WIDE ? tb_lane : t;
        const int pa = (tb << (TBS - 1)) | (pbase0 + b);
        return ((uint32_t) kA << (16 - KA)) | ((uint32_t) pa << 1);
    };
    constexpr int NT_ = WIDE ? 1 : 2;
    half2 values[NPB][NT_][NKA];
    #pragma unroll
    for (int b = 0; b < NPB; ++b)
        #pragma unroll
        for (int t = 0; t < NT_; ++t)
            #pragma unroll
            for (int j = 0; j < NKA; ++j)
            {
                const uint32_t w = window(b, t, j);
                values[b][t][j] = decode_3inst_2<2>(w, w + 1);
            }
    __syncthreads();

    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    // One step with in-width KIN and out-width KOUT
    auto step = [&](auto kin_c, auto kout_c, const half wv, const bool first, const int pre_state,
                    uint8_t* hist_row, const bool write_hist)
    {
        constexpr int KIN = decltype(kin_c)::value;
        constexpr int KOUT = decltype(kout_c)::value;
        const half2 w2 = __half2half2(wv);
        const half2 inf2 = __half2half2(H_INF);
        const half* costs_in = temp_costs_inc;
        half* costs_out = temp_costs;
        auto eval = [&](int b, int t, int j, uint32_t& best0, uint32_t& best1)
        {
            const uint32_t in_edge = window(b, t, j) >> KIN;
            const half2 dh2 = __hsub2(values[b][t][j], w2);
            half2 err2;
            if (first)
            {
                err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && (int) in_edge != pre_state) err2 = inf2;
            }
            else
                err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge]));
            best0 = min(best0, ((uint32_t) __half_as_ushort(__low2half(err2)) << 16) | in_edge);
            best1 = min(best1, ((uint32_t) __half_as_ushort(__high2half(err2)) << 16) | in_edge);
        };
        auto store = [&](uint32_t out_edge, uint32_t best0, uint32_t best1)
        {
            reinterpret_cast<half2*>(costs_out)[out_edge >> 1] =
                __halves2half2(__ushort_as_half((uint16_t) (best0 >> 16)), __ushort_as_half((uint16_t) (best1 >> 16)));
            if (write_hist)
            {
                // k = window >> (16 - KOUT) = in_edge >> (16 - KOUT - KIN)
                constexpr int ks = 16 - KOUT - KIN;
                reinterpret_cast<uint16_t*>(hist_row)[out_edge >> 1] =
                    (uint16_t) ((((best0 & 0xffffu) >> ks) & 0xffu) | ((((best1 & 0xffffu) >> ks) & 0xffu) << 8));
            }
        };
        constexpr uint32_t out_mask = (1u << (16 - KOUT)) - 1;

        if constexpr (WIDE)
        {
            uint32_t best0 = 0xffffffffu, best1 = 0xffffffffu;
            #pragma unroll
            for (int j = 0; j < NKA; ++j) eval(0, 0, j, best0, best1);
            // KOUT = KA: the GA lanes of this oA pair; KOUT = KB: also the tb partner group
            constexpr int span = KOUT == KA ? GA : 2 * GA;
            #pragma unroll
            for (int o = 1; o < span; o <<= 1)
            {
                best0 = min(best0, __shfl_xor_sync(0xffffffffu, best0, o));
                best1 = min(best1, __shfl_xor_sync(0xffffffffu, best1, o));
            }
            if ((thread & (span - 1)) == 0) store(window(0, 0, 0) & out_mask, best0, best1);
        }
        else if constexpr (KOUT == KA)
        {
            #pragma unroll
            for (int b = 0; b < NPB; ++b)
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                {
                    uint32_t best0 = 0xffffffffu, best1 = 0xffffffffu;
                    #pragma unroll
                    for (int j = 0; j < NKA; ++j) eval(b, t, j, best0, best1);
                    store(window(b, t, 0) & out_mask, best0, best1);
                }
        }
        else
        {
            #pragma unroll
            for (int b = 0; b < NPB; ++b)
            {
                uint32_t best0 = 0xffffffffu, best1 = 0xffffffffu;
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                    #pragma unroll
                    for (int j = 0; j < NKA; ++j) eval(b, t, j, best0, best1);
                store(window(b, 0, 0) & out_mask, best0, best1);
            }
        }
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
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint8_t* hist_row = temp_edges + (size_t) edges_max * ri;
            const half wv = sh_input_tile[ri];
            const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
            using A = std::integral_constant<int, KA>;
            using B = std::integral_constant<int, KB>;
            switch (din * 2 + dout)
            {
                case 0: step(A{}, A{}, wv, first, pre_state, hist_row, write_hist); break;
                case 1: step(A{}, B{}, wv, first, pre_state, hist_row, write_hist); break;
                case 2: step(B{}, A{}, wv, first, pre_state, hist_row, write_hist); break;
                default: step(B{}, B{}, wv, first, pre_state, hist_row, write_hist); break;
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
                const int k = (int) temp_edges[(size_t) edges_max * ri + edge];
                const int kin = KA + ((MASK >> (ri & 15)) & 1);
                const int kout = KA + ((MASK >> ((ri + 1) & 15)) & 1);
                const int encoded = (k << (16 - kout)) | edge;
                const int prev_edge = encoded >> kin;
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
