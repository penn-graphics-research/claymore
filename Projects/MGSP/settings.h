#ifndef SETTINGS_H
#define SETTINGS_H
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>

#include <array>

#include "partition_domain.h"

namespace mn {

using ivec3	   = vec<int, 3>;
using vec3	   = vec<float, 3>;
using vec9	   = vec<float, 9>;
using vec3x3   = vec<float, 3, 3>;
using vec3x4   = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;

/// sand = Drucker Prager Plasticity, StvkHencky Elasticity
enum class MaterialE {
	J_FLUID = 0,
	FIXED_COROTATED,
	SAND,
	NACC,
	TOTAL
};

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
/// benchmark setup
namespace config {
	constexpr int G_DEVICE_CNT = 2;
	constexpr MaterialE get_material_type(int did) noexcept {
		(void) did;

		return MaterialE::J_FLUID;
	}
	constexpr int G_TOTAL_FRAME_CNT = 60;
	constexpr int NUM_DIMENSIONS	= 3;

	constexpr int GBPCB							   = 16;
	constexpr int G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK = GBPCB;
	constexpr int G_NUM_WARPS_PER_GRID_BLOCK	   = 1;
	constexpr int G_NUM_WARPS_PER_CUDA_BLOCK	   = GBPCB;
	constexpr int G_PARTICLE_BATCH_CAPACITY		   = 128;

	constexpr float MODEL_PPC	= 8.0f;
	constexpr float G_MODEL_PPC = MODEL_PPC;
	constexpr float CFL			= 0.5f;

	// background_grid
	constexpr int BLOCK_BITS	= 2;
	constexpr int DOMAIN_BITS	= 8;
	constexpr float DXINV		= (1.0f * (1u << DOMAIN_BITS));
	constexpr int G_DOMAIN_BITS = DOMAIN_BITS;
	constexpr int G_DOMAIN_SIZE = (1 << DOMAIN_BITS);
	constexpr float G_BC		= 2.0;
	constexpr float G_DX		= 1.f / DXINV;
	constexpr float G_DX_INV	= DXINV;
	constexpr float G_D_INV		= 4.f * DXINV * DXINV;
	constexpr int G_BLOCKBITS	= BLOCK_BITS;
	constexpr int G_BLOCKSIZE	= (1 << BLOCK_BITS);
	constexpr int G_BLOCKMASK	= ((1 << BLOCK_BITS) - 1);
	constexpr int G_BLOCKVOLUME = (1 << (BLOCK_BITS * 3));
	constexpr int G_GRID_BITS	= (DOMAIN_BITS - BLOCK_BITS);
	constexpr int G_GRID_SIZE	= (1 << (DOMAIN_BITS - BLOCK_BITS));

	// partition domains
	//NOLINTBEGIN(readability-magic-numbers) Numbers are used for dividing domains
	constexpr BoxDomain<int, NUM_DIMENSIONS> get_domain(int did) noexcept {
		constexpr int LEN = G_GRID_SIZE / 2;
		BoxDomain<int, NUM_DIMENSIONS> domain {};
		for(int d = 0; d < NUM_DIMENSIONS; ++d) {
			domain.min[d] = 0;
			domain.max[d] = G_GRID_SIZE - 1;
		}
		if constexpr(G_DEVICE_CNT == 1) {
			/// default
		} else if(G_DEVICE_CNT == 2) {
			if(did == 0) {
				domain.max[0] = LEN;
			} else if(did == 1) {//NOLINT(hicpp-multiway-paths-covered) Otherwise unchanged
				domain.min[0] = LEN + 1;
			}
		} else if(G_DEVICE_CNT <= 4 && G_DEVICE_CNT >= 3) {
			domain.min[0] = ((did & 2) != 0) ? LEN + 1 : 0;
			domain.min[2] = ((did & 1) != 0) ? LEN + 1 : 0;
			domain.max[0] = ((did & 2) != 0) ? G_GRID_SIZE - 1 : LEN;
			domain.max[2] = ((did & 1) != 0) ? G_GRID_SIZE - 1 : LEN;
		} else {
			domain.max[0] = domain.max[1] = domain.max[2] = -3;
		}
		return domain;
	}
	//NOLINTEND(readability-magic-numbers)

	// particle
	constexpr int MAX_PPC				   = 128;
	constexpr int G_MAX_PPC				   = MAX_PPC;
	constexpr int G_BIN_CAPACITY		   = 32;
	constexpr int G_PARTICLE_NUM_PER_BLOCK = (MAX_PPC * (1 << (BLOCK_BITS * 3)));

	// material parameters
	constexpr float DENSITY		   = 1e3;
	constexpr float YOUNGS_MODULUS = 5e3;
	constexpr float POISSON_RATIO  = 0.4f;

	//
	constexpr float G_GRAVITY = -9.8f * 0.5f;

	/// only used on host
	constexpr int G_MAX_PARTICLE_NUM = 2000000;
	constexpr int G_MAX_ACTIVE_BLOCK = 12000;/// 62500 bytes for active mask
	constexpr std::size_t calc_particle_bin_count(std::size_t num_active_blocks) noexcept {
		return num_active_blocks * (G_MAX_PPC * G_BLOCKVOLUME / G_BIN_CAPACITY);
	}
	constexpr std::size_t G_MAX_PARTICLE_BIN = G_MAX_PARTICLE_NUM / G_BIN_CAPACITY;
	constexpr std::size_t G_MAX_HALO_BLOCK	 = 4000;

}// namespace config

}// namespace mn

#endif