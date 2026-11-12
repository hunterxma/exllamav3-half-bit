#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_frac_kernel.cuh"
#include "../quantize_tiles_frac_reg_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a1_maaaa() { return quantize_tiles_frac_kernel<1, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a2_maaaa() { return quantize_tiles_frac_kernel<2, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa() { return quantize_tiles_frac_kernel<3, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a4_maaaa() { return quantize_tiles_frac_kernel<4, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a5_maaaa() { return quantize_tiles_frac_kernel<5, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a6_maaaa() { return quantize_tiles_frac_kernel<6, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a7_maaaa() { return quantize_tiles_frac_kernel<7, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a2_maaaa_t512() { return quantize_tiles_frac_reg_kernel<2, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a2_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<2, 0xaaaau, 1024>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a3_maaaa_t512() { return quantize_tiles_frac_reg_kernel<3, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a3_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<3, 0xaaaau, 1024>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a4_maaaa_t512() { return quantize_tiles_frac_reg_kernel<4, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a4_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<4, 0xaaaau, 1024>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a5_maaaa_t512() { return quantize_tiles_frac_reg_kernel<5, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a5_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<5, 0xaaaau, 1024>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a6_maaaa_t512() { return quantize_tiles_frac_reg_kernel<6, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a6_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<6, 0xaaaau, 1024>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a7_maaaa_t512() { return quantize_tiles_frac_reg_kernel<7, 0xaaaau, 512>; }
fp_quantize_tiles_kernel quantize_tiles_frac_reg_kernel_a7_maaaa_t1024() { return quantize_tiles_frac_reg_kernel<7, 0xaaaau, 1024>; }
