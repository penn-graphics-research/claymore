#ifndef PARTICLE_BUFFER_CUH
#define PARTICLE_BUFFER_CUH
#include <MnBase/Meta/Polymorphism.h>

#include "settings.h"

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define PRINT_NEGATIVE_BLOGNOS 1

namespace mn {

using ParticleBinDomain	   = AlignedDomain<char, config::G_BIN_CAPACITY>;
using ParticleBufferDomain = CompactDomain<int, config::G_MAX_PARTICLE_BIN>;
using ParticleArrayDomain  = CompactDomain<int, config::G_MAX_PARTICLE_NUM>;
//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reporst variable errors fro template arguments
using particle_bin4_  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;														///< J, pos
using particle_bin12_ = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;		///< pos, F
using particle_bin13_ = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, ParticleBinDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_, f32_>;///< pos, F, logJp
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

template<MaterialE Mt>
struct particle_bin_;
template<>
struct particle_bin_<MaterialE::J_FLUID> : particle_bin4_ {};
template<>
struct particle_bin_<MaterialE::FIXED_COROTATED> : particle_bin12_ {};
template<>
struct particle_bin_<MaterialE::SAND> : particle_bin13_ {};
template<>
struct particle_bin_<MaterialE::NACC> : particle_bin13_ {};

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reporst variable errors fro template arguments
template<typename ParticleBin>
using particle_buffer_ = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, ParticleBufferDomain, attrib_layout::AOS, ParticleBin>;
using particle_array_  = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, ParticleArrayDomain, attrib_layout::AOS, f32_, f32_, f32_>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

template<MaterialE Mt>
struct ParticleBufferImpl : Instance<particle_buffer_<particle_bin_<Mt>>> {
	static constexpr MaterialE MATERIAL_TYPE = Mt;
	using base_t							 = Instance<particle_buffer_<particle_bin_<Mt>>>;

	std::size_t num_active_blocks;
	int* cell_particle_counts;
	int* particle_bucket_sizes;
	int* cellbuckets;
	int* blockbuckets;
	int* bin_offsets;

	template<typename Allocator>
	explicit ParticleBufferImpl(Allocator allocator)
		: base_t {spawn<particle_buffer_<particle_bin_<Mt>>, orphan_signature>(allocator)}
		, num_active_blocks()
		, cell_particle_counts(nullptr)
		, particle_bucket_sizes(nullptr)
		, cellbuckets(nullptr)
		, blockbuckets(nullptr)
		, bin_offsets(nullptr) {}

	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > this->capacity)
			this->resize(allocator, capacity);
	}
};

//NOTE: In subsequent classes some parameters are actually const and static, but they are nit declared as such to allow consistent access to the members
//TODO: Maybe write accessors; Maybe create common baseclass(es) for accessing the parameters

template<MaterialE Mt>
struct ParticleBuffer;
template<>
struct ParticleBuffer<MaterialE::J_FLUID> : ParticleBufferImpl<MaterialE::J_FLUID> {
	using base_t = ParticleBufferImpl<MaterialE::J_FLUID>;

	//NOLINTBEGIN(readability-magic-numbers) Parameter definitions
	float rho		= config::DENSITY;
	float volume	= (1.0f / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float mass		= (config::DENSITY / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float bulk		= 4e4;
	float gamma		= 7.15f;
	float viscosity = 0.01f;

	void update_parameters(float density, float vol, float b, float g, float v) {
		rho		  = density;
		volume	  = vol;
		mass	  = volume * density;
		bulk	  = b;
		gamma	  = g;
		viscosity = v;
	}
	//NOLINTEND(readability-magic-numbers)

	template<typename Allocator>
	explicit ParticleBuffer(Allocator allocator)
		: base_t {allocator} {}
};

template<>
struct ParticleBuffer<MaterialE::FIXED_COROTATED> : ParticleBufferImpl<MaterialE::FIXED_COROTATED> {
	using base_t = ParticleBufferImpl<MaterialE::FIXED_COROTATED>;

	//NOLINTBEGIN(readability-magic-numbers) Parameter definitions
	float rho	 = config::DENSITY;
	float volume = (10.f / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float mass	 = (config::DENSITY / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float e		 = config::YOUNGS_MODULUS;
	float nu	 = config::POISSON_RATIO;
	float lambda = config::YOUNGS_MODULUS * config::POISSON_RATIO / ((1 + config::POISSON_RATIO) * (1 - 2 * config::POISSON_RATIO));
	float mu	 = config::YOUNGS_MODULUS / (2 * (1 + config::POISSON_RATIO));

	void update_parameters(float density, float vol, float e, float nu) {
		rho	   = density;
		volume = vol;
		mass   = volume * density;
		lambda = e * nu / ((1 + nu) * (1 - 2 * nu));
		mu	   = e / (2 * (1 + nu));
	}
	//NOLINTEND(readability-magic-numbers)

	template<typename Allocator>
	explicit ParticleBuffer(Allocator allocator)
		: base_t {allocator} {}
};

template<>
struct ParticleBuffer<MaterialE::SAND> : ParticleBufferImpl<MaterialE::SAND> {
	using base_t = ParticleBufferImpl<MaterialE::SAND>;

	//NOLINTBEGIN(readability-magic-numbers) Parameter definitions; consistent naming
	float rho	 = config::DENSITY;
	float volume = (10.f / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float mass	 = (config::DENSITY / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float r		 = config::YOUNGS_MODULUS;
	float nu	 = config::POISSON_RATIO;
	float lambda = config::YOUNGS_MODULUS * config::POISSON_RATIO / ((1 + config::POISSON_RATIO) * (1 - 2 * config::POISSON_RATIO));
	float mu	 = config::YOUNGS_MODULUS / (2 * (1 + config::POISSON_RATIO));

	static constexpr float LOG_JP_0 = 0.0f;
	float friction_angle			= 30.f;
	float cohesion					= 0.0f;
	float beta						= 1.0f;
	// std::sqrt(2.f/3.f) * 2.f * std::sin(30.f/180.f*3.141592741f)
	// 						/ (3.f -
	// std::sin(30.f/180.f*3.141592741f))
	float yield_surface	   = 0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f);
	bool volume_correction = true;
	//NOLINTEND(readability-magic-numbers)

	template<typename Allocator>
	explicit ParticleBuffer(Allocator allocator)
		: base_t {allocator} {}
};

template<>
struct ParticleBuffer<MaterialE::NACC> : ParticleBufferImpl<MaterialE::NACC> {
	using base_t = ParticleBufferImpl<MaterialE::NACC>;

	//NOLINTBEGIN(readability-magic-numbers) Parameter definitions
	float rho	 = config::DENSITY;
	float volume = (1.0f / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float mass	 = (config::DENSITY / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / (1u << config::DOMAIN_BITS) / config::MODEL_PPC);
	float e		 = config::YOUNGS_MODULUS;
	float nu	 = config::POISSON_RATIO;
	float lambda = config::YOUNGS_MODULUS * config::POISSON_RATIO / ((1 + config::POISSON_RATIO) * (1 - 2 * config::POISSON_RATIO));
	float mu	 = config::YOUNGS_MODULUS / (2 * (1 + config::POISSON_RATIO));

	float friction_angle			= 45.f;
	float bm						= 2.f / 3.f * (config::YOUNGS_MODULUS / (2 * (1 + config::POISSON_RATIO))) + (config::YOUNGS_MODULUS * config::POISSON_RATIO / ((1 + config::POISSON_RATIO) * (1 - 2 * config::POISSON_RATIO)));///< bulk modulus, kappa
	float xi						= 0.8f;																																															///< hardening factor
	static constexpr float LOG_JP_0 = -0.01f;
	float beta						= 0.5f;
	float mohr_columb_friction		= 0.503599787772409;//< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
	float m							= 1.850343771924453;//< mohr_columb_friction * (T)dim / sqrt((T)2 / ((T)6 - dim));
	float msqr						= 3.423772074299613;
	bool hardening_on				= true;

	void update_parameters(float density, float vol, float e, float nu, float be, float x) {
		rho	   = density;
		volume = vol;
		mass   = volume * density;
		lambda = e * nu / ((1 + nu) * (1 - 2 * nu));
		mu	   = e / (2 * (1 + nu));
		bm	   = 2.f / 3.f * (e / (2 * (1 + nu))) + (e * nu / ((1 + nu) * (1 - 2 * nu)));
		beta   = be;
		xi	   = x;
	}
	//NOLINTEND(readability-magic-numbers)

	template<typename Allocator>
	explicit ParticleBuffer(Allocator allocator)
		: base_t {allocator} {}
};

/// conversion
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
using particle_buffer_t = variant<ParticleBuffer<MaterialE::J_FLUID>, ParticleBuffer<MaterialE::FIXED_COROTATED>, ParticleBuffer<MaterialE::SAND>, ParticleBuffer<MaterialE::NACC>>;

struct ParticleArray : Instance<particle_array_> {
	using base_t = Instance<particle_array_>;

	ParticleArray() = default;
	explicit ParticleArray(base_t&& instance)//NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) Clang say, that std::move has no effect here
		: base_t(instance) {}
};

}// namespace mn

#endif