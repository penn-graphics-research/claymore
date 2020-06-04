#ifndef __PARTICLE_BUFFER_CUH_
#define __PARTICLE_BUFFER_CUH_
#include "settings.h"
#include <MnBase/Meta/Polymorphism.h>

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
  ParticleBufferImpl(Allocator allocator)
      : base_t{spawn<particle_buffer_<particle_bin_<mt>>, orphan_signature>(
            allocator)} {}
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > this->_capacity)
      this->resize(allocator, capacity);
  }
};

template <material_e mt> struct ParticleBuffer;
template <>
struct ParticleBuffer<material_e::JFluid>
    : ParticleBufferImpl<material_e::JFluid> {
  using base_t = ParticleBufferImpl<material_e::JFluid>;
  static constexpr float rho = DENSITY;
  static constexpr float volume =
      (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  static constexpr float mass =
      (DENSITY / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  static constexpr float bulk = 4e4;
  static constexpr float gamma = 7.15f;
  static constexpr float visco = 0.01f;
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated>
    : ParticleBufferImpl<material_e::FixedCorotated> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated>;
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
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
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
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
};

template <>
struct ParticleBuffer<material_e::NACC> : ParticleBufferImpl<material_e::NACC> {
  using base_t = ParticleBufferImpl<material_e::NACC>;
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

  static constexpr float frictionAngle = 45.f;
  static constexpr float bm =
      2.f / 3.f * (YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))) +
      (YOUNGS_MODULUS * POISSON_RATIO /
       ((1 + POISSON_RATIO) *
        (1 - 2 * POISSON_RATIO)));  ///< bulk modulus, kappa
  static constexpr float xi = 0.8f; ///< hardening factor
  static constexpr float logJp0 = -0.01f;
  static constexpr float beta = 0.5f;
  static constexpr float mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr float M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr float Msqr = 3.423772074299613;
  static constexpr bool hardeningOn = true;

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}
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
};

} // namespace mn

#endif