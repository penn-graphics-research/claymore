#ifndef __PARTICLE_BUFFER_CUH_
#define __PARTICLE_BUFFER_CUH_
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include <MnBase/Meta/Polymorphism.h>
#include <MnSystem/Cuda/HostUtils.hpp>

namespace mn {

using ParticleBinDomain = aligned_domain<char, config::g_bin_capacity>;
using ParticleBufferDomain = compact_domain<int, config::g_max_particle_bin>;
using ParticleArrayDomain = compact_domain<int, config::g_max_particle_num>;

using particle_bin4_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f32_, f32_, f32_,
               f32_>; ///< J, pos
using particle_bin12_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f32_, f32_, f32_, f32_,
               f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>; ///< pos, F
using particle_bin13_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f32_, f32_, f32_, f32_,
               f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_,
               f32_>; ///< pos, F, logJp
template <material_e mt> struct particle_bin_;
template <> struct particle_bin_<material_e::JFluid> : particle_bin4_ {};
template <>
struct particle_bin_<material_e::FixedCorotated> : particle_bin12_ {};
template <> struct particle_bin_<material_e::Sand> : particle_bin13_ {};
template <> struct particle_bin_<material_e::NACC> : particle_bin13_ {};

template <typename ParticleBin>
using particle_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleBufferDomain, attrib_layout::aos, ParticleBin>;
using particle_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f32_, f32_, f32_>;

template <material_e mt>
struct ParticleBufferImpl : Instance<particle_buffer_<particle_bin_<mt>>> {
  static constexpr material_e materialType = mt;
  using base_t = Instance<particle_buffer_<particle_bin_<mt>>>;

  template <typename Allocator>
  ParticleBufferImpl(Allocator allocator, std::size_t count)
      : base_t{spawn<particle_buffer_<particle_bin_<mt>>, orphan_signature>(
            allocator, count)},
        _numActiveBlocks{0}, _ppcs{nullptr}, _ppbs{nullptr},
        _cellbuckets{nullptr}, _blockbuckets{nullptr}, _binsts{nullptr} {}
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > this->_capacity)
      this->resize(allocator, capacity);
  }
  template <typename Allocator>
  void reserveBuckets(Allocator allocator, std::size_t numBlockCnt) {
    if (_binsts) {
      allocator.deallocate(_ppcs, sizeof(int) * _numActiveBlocks *
                                      config::g_blockvolume);
      allocator.deallocate(_ppbs, sizeof(int) * _numActiveBlocks);
      allocator.deallocate(_cellbuckets, sizeof(int) * _numActiveBlocks *
                                             config::g_blockvolume *
                                             config::g_max_ppc);
      allocator.deallocate(_blockbuckets, sizeof(int) * _numActiveBlocks *
                                              config::g_particle_num_per_block);
      allocator.deallocate(_binsts, sizeof(int) * _numActiveBlocks);
    }
    _numActiveBlocks = numBlockCnt;
    _ppcs = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                      config::g_blockvolume);
    _ppbs = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks);
    _cellbuckets =
        (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                  config::g_blockvolume * config::g_max_ppc);
    _blockbuckets = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                              config::g_particle_num_per_block);
    _binsts = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks);
    resetPpcs();
  }
  void resetPpcs() {
    checkCudaErrors(cudaMemset(
        _ppcs, 0, sizeof(int) * _numActiveBlocks * config::g_blockvolume));
  }
  void copy_to(ParticleBufferImpl &other, std::size_t blockCnt,
               cudaStream_t stream) const {
    checkCudaErrors(cudaMemcpyAsync(other._binsts, _binsts,
                                    sizeof(int) * (blockCnt + 1),
                                    cudaMemcpyDefault, stream));
    checkCudaErrors(cudaMemcpyAsync(other._ppbs, _ppbs, sizeof(int) * blockCnt,
                                    cudaMemcpyDefault, stream));
  }
  __forceinline__ __device__ void add_advection(Partition<1> &table,
                                                Partition<1>::key_t cellid,
                                                int dirtag,
                                                int pidib) noexcept {
    using namespace config;
    Partition<1>::key_t blockid = cellid / g_blocksize;
    int blockno = table.query(blockid);
#if 1
    if (blockno == -1) {
      ivec3 offset{};
      dir_components(dirtag, offset);
      printf("loc(%d, %d, %d) dir(%d, %d, %d) pidib(%d)\n", cellid[0],
             cellid[1], cellid[2], offset[0], offset[1], offset[2], pidib);
      return;
    }
#endif
    int cellno = ((cellid[0] & g_blockmask) << (g_blockbits << 1)) |
                 ((cellid[1] & g_blockmask) << g_blockbits) |
                 (cellid[2] & g_blockmask);
    int pidic = atomicAdd(_ppcs + blockno * g_blockvolume + cellno, 1);
    _cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                 pidic] = (dirtag * g_particle_num_per_block) | pidib;
  }

  std::size_t _numActiveBlocks;
  int *_ppcs, *_ppbs;
  int *_cellbuckets, *_blockbuckets;
  int *_binsts;
};

template <material_e mt> struct ParticleBuffer;
template <>
struct ParticleBuffer<material_e::JFluid>
    : ParticleBufferImpl<material_e::JFluid> {
  using base_t = ParticleBufferImpl<material_e::JFluid>;
  float rho = DENSITY;
  float volume = (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  float mass = (DENSITY / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                (1 << DOMAIN_BITS) / MODEL_PPC);
  float bulk = 4e4;
  float gamma = 7.15f;
  float visco = 0.01f;
  void updateParameters(float density, float vol, float b, float g, float v) {
    rho = density;
    volume = vol;
    mass = volume * density;
    bulk = b;
    gamma = g;
    visco = v;
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated>
    : ParticleBufferImpl<material_e::FixedCorotated> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated>;
  float rho = DENSITY;
  float volume = (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  float mass = (DENSITY / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                (1 << DOMAIN_BITS) / MODEL_PPC);
  float E = YOUNGS_MODULUS;
  float nu = POISSON_RATIO;
  float lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  float mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  void updateParameters(float density, float vol, float E, float nu) {
    rho = density;
    volume = vol;
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
  }
  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::Sand> : ParticleBufferImpl<material_e::Sand> {
  using base_t = ParticleBufferImpl<material_e::Sand>;
  static constexpr float rho = DENSITY;
  static constexpr float volume =
      (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  static constexpr float mass =
      (DENSITY / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  static constexpr float E = YOUNGS_MODULUS;
  static constexpr float nu = POISSON_RATIO;
  static constexpr float lambda =
      YOUNGS_MODULUS * POISSON_RATIO /
      ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  static constexpr float mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  static constexpr float logJp0 = 0.f;
  static constexpr float frictionAngle = 30.f;
  static constexpr float cohesion = 0.f;
  static constexpr float beta = 1.f;
  // std::sqrt(2.f/3.f) * 2.f * std::sin(30.f/180.f*3.141592741f)
  // 						/ (3.f -
  // std::sin(30.f/180.f*3.141592741f))
  static constexpr float yieldSurface =
      0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f);
  static constexpr bool volumeCorrection = true;

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::NACC> : ParticleBufferImpl<material_e::NACC> {
  using base_t = ParticleBufferImpl<material_e::NACC>;
  float rho = DENSITY;
  float volume = (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  float mass = (DENSITY / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                (1 << DOMAIN_BITS) / MODEL_PPC);
  float E = YOUNGS_MODULUS;
  float nu = POISSON_RATIO;
  float lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  float mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  float frictionAngle = 45.f;
  float bm = 2.f / 3.f * (YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))) +
             (YOUNGS_MODULUS * POISSON_RATIO /
              ((1 + POISSON_RATIO) *
               (1 - 2 * POISSON_RATIO))); ///< bulk modulus, kappa
  float xi = 0.8f;                        ///< hardening factor
  static constexpr float logJp0 = -0.01f;
  float beta = 0.5f;
  static constexpr float mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr float M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr float Msqr = 3.423772074299613;
  static constexpr bool hardeningOn = true;

  void updateParameters(float density, float vol, float E, float nu, float be,
                        float x) {
    rho = density;
    volume = vol;
    mass = volume * density;
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
    bm =
        2.f / 3.f * (E / (2 * (1 + nu))) + (E * nu / ((1 + nu) * (1 - 2 * nu)));
    beta = be;
    xi = x;
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

/// conversion
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
using particle_buffer_t =
    variant<ParticleBuffer<material_e::JFluid>,
            ParticleBuffer<material_e::FixedCorotated>,
            ParticleBuffer<material_e::Sand>, ParticleBuffer<material_e::NACC>>;

struct ParticleArray : Instance<particle_array_> {
  using base_t = Instance<particle_array_>;
  ParticleArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  ParticleArray(base_t &&instance) { static_cast<base_t &>(*this) = instance; }
};

} // namespace mn

#endif