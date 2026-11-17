#include "exl3_moe_coop_instances.cuh"
#include "../exl3_moe_coop_kernel.cuh"

// Gate/up kernels for a GPU rate tier's two rates in one launch (exl3_moe_coop_a_kernel with bits2), mul1 codebook.
// Only the pairs built models use are instantiated; any other pair keeps one launch per rate
MoeCoopKernel exl3_moe_coop_kernel_a_dual(int K1, bool h1, int K2, bool h2, int Hi, bool wide)
{
    using namespace exl3_moe_coop_ns;
    #define DUAL(k1, hh1, k2, hh2) \
        if (K1 == k1 && h1 == hh1 && K2 == k2 && h2 == hh2) \
        { \
            const int s1 = smem_a_bytes<k1, hh1>(Hi), s2 = smem_a_bytes<k2, hh2>(Hi); \
            const int smem = s1 > s2 ? s1 : s2; \
            if (wide) return { (void*) exl3_moe_coop_a_kernel<k1, 2, true, hh1, k2, hh2>, smem }; \
            return { (void*) exl3_moe_coop_a_kernel<k1, 2, false, hh1, k2, hh2>, smem }; \
        }
    DUAL(4, true, 5, false)     // 4.5 / 5 bpw (MX T5, V2F-24 layers 8-29)
    DUAL(5, false, 4, true)
    #undef DUAL
    return { nullptr, 0 };
}
