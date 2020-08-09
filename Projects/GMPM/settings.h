#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <array>

namespace mn {

using ivec3 = vec<int, 3>;
using vec3 = vec<float, 3>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

/// sand = Drucker Prager Plasticity, StvkHencky Elasticity
enum class material_e { JFluid = 0, FixedCorotated, Sand, NACC, Total };

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
constexpr int g_device_cnt = 2;
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::FixedCorotated};
  return type;
}
constexpr int g_total_frame_cnt = 60;

#define GBPCB 16
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = GBPCB;
constexpr int g_particle_batch_capacity = 128;

#define MODEL_PPC 8.f
constexpr float g_model_ppc = MODEL_PPC;
constexpr float cfl = 0.5f;

// background_grid
#define BLOCK_BITS 2
#define DOMAIN_BITS 8
#define DXINV (1.f * (1 << DOMAIN_BITS))
constexpr int g_domain_bits = DOMAIN_BITS;
constexpr int g_domain_size = (1 << DOMAIN_BITS);
constexpr float g_bc = 2;
constexpr float g_dx = 1.f / DXINV;
constexpr float g_dx_inv = DXINV;
constexpr float g_D_inv = 4.f * DXINV * DXINV;
constexpr int g_blockbits = BLOCK_BITS;
constexpr int g_blocksize = (1 << BLOCK_BITS);
constexpr int g_blockmask = ((1 << BLOCK_BITS) - 1);
constexpr int g_blockvolume = (1 << (BLOCK_BITS * 3));
constexpr int g_grid_bits = (DOMAIN_BITS - BLOCK_BITS);
constexpr int g_grid_size = (1 << (DOMAIN_BITS - BLOCK_BITS));

// particle
#define MAX_PPC 128
constexpr int g_max_ppc = MAX_PPC;
constexpr int g_bin_capacity = 32;
constexpr int g_particle_num_per_block = (MAX_PPC * (1 << (BLOCK_BITS * 3)));

// material parameters
#define DENSITY 1e3
#define YOUNGS_MODULUS 5e3
#define POISSON_RATIO 0.4f

//
constexpr float g_gravity = -9.8f;

/// only used on host
constexpr int g_max_particle_num = 1000000;
constexpr int g_max_active_block = 10000; /// 62500 bytes for active mask
constexpr std::size_t
calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
  return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity);
}
constexpr std::size_t g_max_particle_bin = g_max_particle_num / g_bin_capacity;
constexpr std::size_t g_max_halo_block = 4000;

} // namespace config

using BlockDomain = compact_domain<char, config::g_blocksize,
                                   config::g_blocksize, config::g_blocksize>;
using GridDomain = compact_domain<int, config::g_grid_size, config::g_grid_size,
                                  config::g_grid_size>;
using GridBufferDomain = compact_domain<int, config::g_max_active_block>;

} // namespace mn

#endif