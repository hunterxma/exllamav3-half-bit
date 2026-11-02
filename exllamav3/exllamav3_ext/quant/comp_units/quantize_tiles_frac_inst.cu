#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_frac_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a1_maaaa() { return quantize_tiles_frac_kernel<1, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a2_maaaa() { return quantize_tiles_frac_kernel<2, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa() { return quantize_tiles_frac_kernel<3, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a4_maaaa() { return quantize_tiles_frac_kernel<4, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a5_maaaa() { return quantize_tiles_frac_kernel<5, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a6_maaaa() { return quantize_tiles_frac_kernel<6, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a7_maaaa() { return quantize_tiles_frac_kernel<7, 0xaaaau>; }
