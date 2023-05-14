#ifndef MULTI_GMPM_KERNELS_CUH
#define MULTI_GMPM_KERNELS_CUH

#include <MnBase/Math/Matrix/MatrixUtils.h>

#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnSystem/Cuda/DeviceUtils.cuh>

#include "boundary_condition.cuh"
#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//TODO: Maybe use names instead of formula signs for better understanding
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers) CUDA does not yet support std::span; Common names for physical formulas; Cannot declare __global__ functions inline
template<typename ParticleArray, typename Partition>
__global__ void activate_blocks(uint32_t particle_counts, ParticleArray particle_array, Partition partition) {
	uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_counts) {
		return;
	}
	ivec3 blockid {static_cast<int>((std::lround(particle_array.val(_0, particle_id) / config::G_DX) - 2) / config::G_BLOCKSIZE), static_cast<int>((std::lround(particle_array.val(_1, particle_id) / config::G_DX) - 2) / config::G_BLOCKSIZE), static_cast<int>((std::lround(particle_array.val(_2, particle_id) / config::G_DX) - 2) / config::G_BLOCKSIZE)};
	partition.insert(blockid);
}
template<typename ParticleArray, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particle_counts, ParticleArray particle_array, Partition partition) {
	uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_counts) {
		return;
	}
	ivec3 coord {int(std::lround(particle_array.val(_0, particle_id) / config::G_DX) - 2), int(std::lround(particle_array.val(_1, particle_id) / config::G_DX) - 2), int(std::lround(particle_array.val(_2, particle_id) / config::G_DX) - 2)};
	int cellno																														   = (coord[0] & config::G_BLOCKMASK) * config::G_BLOCKSIZE * config::G_BLOCKSIZE + (coord[1] & config::G_BLOCKMASK) * config::G_BLOCKSIZE + (coord[2] & config::G_BLOCKMASK);
	coord																															   = coord / static_cast<int>(config::G_BLOCKSIZE);
	auto blockno																													   = partition.query(coord);
	auto particle_id_in_cell																										   = atomicAdd(partition.cell_particle_counts + blockno * config::G_BLOCKVOLUME + cellno, 1);
	partition.cellbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_cell] = static_cast<int>(particle_id);//NOTE:Explicit narrowing conversation.
}
__global__ void cell_bucket_to_block(const int* cell_particle_counts, const int* cellbuckets, int* particle_bucket_sizes, int* buckets) {
	const int cellno		  = static_cast<int>(threadIdx.x) & (config::G_BLOCKVOLUME - 1);
	const int particle_counts = cell_particle_counts[blockIdx.x * config::G_BLOCKVOLUME + cellno];
	for(int particle_id_in_cell = 0; particle_id_in_cell < config::G_MAX_PARTICLES_IN_CELL; particle_id_in_cell++) {
		if(particle_id_in_cell < particle_counts) {
			auto particle_id_in_block													  = atomic_agg_inc<int>(particle_bucket_sizes + blockIdx.x);
			buckets[blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block] = cellbuckets[blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_cell];
		}
		__syncthreads();
	}
}
__global__ void compute_bin_capacity(uint32_t block_count, int const* particle_bucket_sizes, int* bin_sizes) {
	const uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	bin_sizes[blockno] = (particle_bucket_sizes[blockno] + config::G_BIN_CAPACITY - 1) / config::G_BIN_CAPACITY;
}
__global__ void init_adv_bucket(const int* particle_bucket_sizes, int* buckets) {
	int particle_counts = particle_bucket_sizes[blockIdx.x];
	int* bucket			= buckets + static_cast<size_t>(blockIdx.x) * config::G_PARTICLE_NUM_PER_BLOCK;
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		bucket[particle_id_in_block] = (dir_offset(ivec3 {0, 0, 0}) * config::G_PARTICLE_NUM_PER_BLOCK) | particle_id_in_block;
	}
}
template<typename Grid>
__global__ void clear_grid(Grid grid) {
	auto gridblock = grid.ch(_0, blockIdx.x);
	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		gridblock.val_1d(_0, cell_id_in_block) = 0.f;
		gridblock.val_1d(_1, cell_id_in_block) = 0.f;
		gridblock.val_1d(_2, cell_id_in_block) = 0.f;
		gridblock.val_1d(_3, cell_id_in_block) = 0.f;
	}
}
template<typename Partition>
__global__ void register_neighbor_blocks(uint32_t block_count, Partition partition) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	auto blockid = partition.active_keys[blockno];
	for(char i = 0; i < 2; ++i) {
		for(char j = 0; j < 2; ++j) {
			for(char k = 0; k < 2; ++k) {
				partition.insert(ivec3 {blockid[0] + i, blockid[1] + j, blockid[2] + k});
			}
		}
	}
}
template<typename Partition>
__global__ void register_exterior_blocks(uint32_t block_count, Partition partition) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	auto blockid = partition.active_keys[blockno];
	for(char i = -1; i < 2; ++i) {
		for(char j = -1; j < 2; ++j) {
			for(char k = -1; k < 2; ++k) {
				partition.insert(ivec3 {blockid[0] + i, blockid[1] + j, blockid[2] + k});
			}
		}
	}
}
template<typename Grid, typename Partition>
__global__ void rasterize(uint32_t particle_counts, const ParticleArray particle_array, Grid grid, const Partition partition, float dt, float mass) {
	uint32_t particle_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(particle_id >= particle_counts) {
		return;
	}

	vec3 local_pos {particle_array.val(_0, particle_id), particle_array.val(_1, particle_id), particle_array.val(_2, particle_id)};
	vec3 vel;
	vec9 contrib;
	vec9 c;
	vel.set(0.0f), contrib.set(0.f), c.set(0.f);
	contrib = (c * mass - contrib * dt) * config::G_D_INV;
	ivec3 global_base_index {static_cast<int>(std::lround(local_pos[0] * config::G_DX_INV) - 1), static_cast<int>(std::lround(local_pos[1] * config::G_DX_INV) - 1), static_cast<int>(std::lround(local_pos[2] * config::G_DX_INV) - 1)};
	local_pos = local_pos - global_base_index * config::G_DX;
	vec<vec3, 3> dws;
	for(int d = 0; d < 3; ++d) {
		dws[d] = bspline_weight(local_pos[d]);
	}
	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 3; ++j) {
			for(int k = 0; k < 3; ++k) {
				ivec3 offset {i, j, k};
				vec3 xixp		  = offset * config::G_DX - local_pos;
				float w			  = dws[0][i] * dws[1][j] * dws[2][k];
				ivec3 local_index = global_base_index + offset;
				float wm		  = mass * w;
				int blockno		  = partition.query(ivec3 {static_cast<int>(local_index[0] >> config::G_BLOCKBITS), static_cast<int>(local_index[1] >> config::G_BLOCKBITS), static_cast<int>(local_index[2] >> config::G_BLOCKBITS)});
				auto grid_block	  = grid.ch(_0, blockno);
				for(int d = 0; d < 3; ++d) {
					local_index[d] = local_index[d] & config::G_BLOCKMASK;
				}
				atomicAdd(&grid_block.val(_0, local_index[0], local_index[1], local_index[2]), wm);
				atomicAdd(&grid_block.val(_1, local_index[0], local_index[1], local_index[2]), wm * vel[0] + (contrib[0] * xixp[0] + contrib[3] * xixp[1] + contrib[6] * xixp[2]) * w);
				atomicAdd(&grid_block.val(_2, local_index[0], local_index[1], local_index[2]), wm * vel[1] + (contrib[1] * xixp[0] + contrib[4] * xixp[1] + contrib[7] * xixp[2]) * w);
				atomicAdd(&grid_block.val(_3, local_index[0], local_index[1], local_index[2]), wm * vel[2] + (contrib[2] * xixp[0] + contrib[5] * xixp[1] + contrib[8] * xixp[2]) * w);
			}
		}
	}
}

template<typename ParticleArray, typename Partion>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::J_FLUID> particle_buffer, Partion partition) {
	uint32_t blockno	= blockIdx.x;
	int particle_counts = partition.particle_bucket_sizes[blockno];
	auto* bucket		= partition.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_id  = bucket[particle_id_in_block];
		auto particle_bin = particle_buffer.ch(_0, partition.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// pos
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// J
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
	}
}

template<typename ParticleArray, typename Partion>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, Partion partition) {
	uint32_t blockno	= blockIdx.x;
	int particle_counts = partition.particle_bucket_sizes[blockno];
	auto* bucket		= partition.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_id  = bucket[particle_id_in_block];
		auto particle_bin = particle_buffer.ch(_0, partition.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// pos
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
	}
}

template<typename ParticleArray, typename Partion>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::SAND> particle_buffer, Partion partition) {
	uint32_t blockno	= blockIdx.x;
	int particle_counts = partition.particle_bucket_sizes[blockno];
	auto* bucket		= partition.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_id  = bucket[particle_id_in_block];
		auto particle_bin = particle_buffer.ch(_0, partition.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// pos
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
		/// log_jp
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = ParticleBuffer<MaterialE::SAND>::LOG_JP_0;
	}
}

template<typename ParticleArray, typename Partion>
__global__ void array_to_buffer(ParticleArray particle_array, ParticleBuffer<MaterialE::NACC> particle_buffer, Partion partition) {
	uint32_t blockno	= blockIdx.x;
	int particle_counts = partition.particle_bucket_sizes[blockno];
	auto* bucket		= partition.blockbuckets + static_cast<size_t>(blockno) * config::G_PARTICLE_NUM_PER_BLOCK;
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto particle_id  = bucket[particle_id_in_block];
		auto particle_bin = particle_buffer.ch(_0, partition.bin_offsets[blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
		/// pos
		particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_0, particle_id);
		particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_1, particle_id);
		particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = particle_array.val(_2, particle_id);
		/// F
		particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = 1.f;
		particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = 0.f;
		particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = 0.f;
		particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = 1.f;
		/// log_jp
		particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = ParticleBuffer<MaterialE::NACC>::LOG_JP_0;
	}
}

template<typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t block_count, Grid grid, Partition partition, float dt, float* max_vel) {
	const int boundary_condition   = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));
	constexpr int NUM_WARPS		   = config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK * config::G_NUM_WARPS_PER_GRID_BLOCK;
	constexpr unsigned ACTIVE_MASK = 0xffffffff;
	//__shared__ float sh_maxvels[config::G_BLOCKVOLUME * config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK
	/// 32];
	extern __shared__ float sh_maxvels[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here
	int blockno		= static_cast<int>(blockIdx.x) * config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK + static_cast<int>(threadIdx.x) / 32 / config::G_NUM_WARPS_PER_GRID_BLOCK;
	auto blockid	= partition.active_keys[blockno];
	int is_in_bound = ((blockid[0] < boundary_condition || blockid[0] >= config::G_GRID_SIZE - boundary_condition) << 2) | ((blockid[1] < boundary_condition || blockid[1] >= config::G_GRID_SIZE - boundary_condition) << 1) | (blockid[2] < boundary_condition || blockid[2] >= config::G_GRID_SIZE - boundary_condition);
	if(threadIdx.x < NUM_WARPS) {
		sh_maxvels[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	/// within-warp computations
	if(blockno < block_count) {
		auto grid_block = grid.ch(_0, blockno);
		for(int cell_id_in_block = static_cast<int>(threadIdx.x % 32); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += 32) {
			float mass	  = grid_block.val_1d(_0, cell_id_in_block);
			float vel_sqr = 0.0f;
			vec3 vel;
			if(mass > 0.0f) {
				mass = 1.f / mass;

				//int i = (cell_id_in_block >> (config::G_BLOCKBITS << 1)) & config::G_BLOCKMASK;
				//int j = (cell_id_in_block >> config::G_BLOCKBITS) & config::G_BLOCKMASK;
				//int k = cell_id_in_block & config::G_BLOCKMASK;

				vel[0] = grid_block.val_1d(_1, cell_id_in_block);
				vel[1] = grid_block.val_1d(_2, cell_id_in_block);
				vel[2] = grid_block.val_1d(_3, cell_id_in_block);

				vel[0] = is_in_bound & 4 ? 0.0f : vel[0] * mass;
				vel[1] = is_in_bound & 2 ? 0.0f : vel[1] * mass;
				vel[1] += config::G_GRAVITY * dt;
				vel[2] = is_in_bound & 1 ? 0.0f : vel[2] * mass;
				// if (is_in_bound) ///< sticky
				//  vel.set(0.f);

				grid_block.val_1d(_1, cell_id_in_block) = vel[0];
				vel_sqr += vel[0] * vel[0];

				grid_block.val_1d(_2, cell_id_in_block) = vel[1];
				vel_sqr += vel[1] * vel[1];

				grid_block.val_1d(_3, cell_id_in_block) = vel[2];
				vel_sqr += vel[2] * vel[2];
			}
			// unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
			for(int iter = 1; iter % 32; iter <<= 1) {
				float tmp = __shfl_down_sync(ACTIVE_MASK, vel_sqr, iter, 32);
				if((threadIdx.x % 32) + iter < 32) {
					vel_sqr = tmp > vel_sqr ? tmp : vel_sqr;
				}
			}
			if(vel_sqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0) {
				sh_maxvels[threadIdx.x / 32] = vel_sqr;
			}
		}
	}
	__syncthreads();
	/// various assumptions
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			if(sh_maxvels[static_cast<int>(threadIdx.x) + interval] > sh_maxvels[threadIdx.x]) {
				sh_maxvels[threadIdx.x] = sh_maxvels[static_cast<int>(threadIdx.x) + interval];
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		atomic_max(max_vel, sh_maxvels[0]);
	}
}

template<typename Grid, typename Partition, typename Boundary>
__global__ void update_grid_velocity_query_max(uint32_t block_count, Grid grid, Partition partition, float dt, Boundary boundary, float* max_vel) {
	const int boundary_condition   = static_cast<int>(std::floor(config::G_BOUNDARY_CONDITION));
	constexpr int NUM_WARPS		   = config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK * config::G_NUM_WARPS_PER_GRID_BLOCK;
	constexpr unsigned ACTIVE_MASK = 0xffffffff;
	//__shared__ float sh_maxvels[config::G_BLOCKVOLUME * config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK
	/// 32];
	extern __shared__ float sh_maxvels[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here
	int blockno		= static_cast<int>(blockIdx.x) * config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK + static_cast<int>(threadIdx.x) / 32 / config::G_NUM_WARPS_PER_GRID_BLOCK;
	auto blockid	= partition.active_keys[blockno];
	int is_in_bound = ((blockid[0] < boundary_condition || blockid[0] >= config::G_GRID_SIZE - boundary_condition) << 2) | ((blockid[1] < boundary_condition || blockid[1] >= config::G_GRID_SIZE - boundary_condition) << 1) | (blockid[2] < boundary_condition || blockid[2] >= config::G_GRID_SIZE - boundary_condition);
	if(threadIdx.x < NUM_WARPS) {
		sh_maxvels[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	/// within-warp computations
	if(blockno < block_count) {
		auto grid_block = grid.ch(_0, blockno);
		for(int cell_id_in_block = static_cast<int>(threadIdx.x % 32); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += 32) {
			float mass	  = grid_block.val_1d(_0, cell_id_in_block);
			float vel_sqr = 0.0f;
			vec3 vel;
			if(mass > 0.0f) {
				mass = 1.f / mass;

				//int i = (cell_id_in_block >> (config::G_BLOCKBITS << 1)) & config::G_BLOCKMASK;
				//int j = (cell_id_in_block >> config::G_BLOCKBITS) & config::G_BLOCKMASK;
				//int k = cell_id_in_block & config::G_BLOCKMASK;

				vel[0] = grid_block.val_1d(_1, cell_id_in_block);
				vel[1] = grid_block.val_1d(_2, cell_id_in_block);
				vel[2] = grid_block.val_1d(_3, cell_id_in_block);

				vel[0] = is_in_bound & 4 ? 0.0f : vel[0] * mass;
				vel[1] = is_in_bound & 2 ? 0.0f : vel[1] * mass;
				vel[1] += config::G_GRAVITY * dt;
				vel[2] = is_in_bound & 1 ? 0.0f : vel[2] * mass;

				ivec3 cellid {(cell_id_in_block & 0x30) >> 4, (cell_id_in_block & 0xc) >> 2, cell_id_in_block & 0x3};
				boundary.detect_and_resolve_collision(blockid, cellid, 0.f, vel);
				vel_sqr									= vel.dot(vel);
				grid_block.val_1d(_1, cell_id_in_block) = vel[0];
				vel_sqr += vel[0] * vel[0];

				grid_block.val_1d(_2, cell_id_in_block) = vel[1];
				vel_sqr += vel[1] * vel[1];

				grid_block.val_1d(_3, cell_id_in_block) = vel[2];
				vel_sqr += vel[2] * vel[2];
			}
			// unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
			for(int iter = 1; iter % 32; iter <<= 1) {
				float tmp = __shfl_down_sync(ACTIVE_MASK, vel_sqr, iter, 32);
				if((threadIdx.x % 32) + iter < 32) {
					vel_sqr = tmp > vel_sqr ? tmp : vel_sqr;
				}
			}
			if(vel_sqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0) {
				sh_maxvels[threadIdx.x / 32] = vel_sqr;
			}
		}
	}
	__syncthreads();
	/// various assumptions
	for(int interval = NUM_WARPS >> 1; interval > 0; interval >>= 1) {
		if(threadIdx.x < interval) {
			if(sh_maxvels[static_cast<int>(threadIdx.x) + interval] > sh_maxvels[threadIdx.x]) {
				sh_maxvels[threadIdx.x] = sh_maxvels[static_cast<int>(threadIdx.x) + interval];
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		atomic_max(max_vel, sh_maxvels[0]);
	}
}

template<typename Partition, typename Grid>
__global__ void g2p2g(float dt, float new_dt, const ivec3* __restrict__ blocks, const ParticleBuffer<MaterialE::J_FLUID> particle_buffer, ParticleBuffer<MaterialE::J_FLUID> next_particle_buffer, const Partition prev_partition, Partition partition, const Grid grid, Grid next_grid) {
	static constexpr uint64_t NUM_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 3;
	static constexpr uint64_t NUM_VI_IN_ARENA  = NUM_VI_PER_BLOCK << 3;

	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using ViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>*;
	using ViArenaRef  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>&;
	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	extern __shared__ char shmem[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	ViArenaRef __restrict__ g2pbuffer  = *static_cast<ViArena>(static_cast<void*>(static_cast<char*>(shmem)));
	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem) + NUM_VI_IN_ARENA * sizeof(float)));

	ivec3 blockid;
	int src_blockno;
	if(blocks != nullptr) {
		blockid		= blocks[blockIdx.x];
		src_blockno = partition.query(blockid);
	} else {
		if(partition.halo_marks[blockIdx.x]) {
			return;
		}
		blockid = partition.active_keys[blockIdx.x];

		int src_blockno			 = static_cast<int>(blockIdx.x);
		int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
		if(particle_bucket_size == 0) {
			return;
		}
	}

	for(int base = static_cast<int>(threadIdx.x); base < NUM_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		auto grid_block		= grid.ch(_0, blockno);
		int channelid		= static_cast<int>(base % NUM_VI_PER_BLOCK);
		char c				= static_cast<char>(channelid & 0x3f);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}
		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc = base;
		char z	= static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char x								 = static_cast<char>(loc & ARENAMASK);
		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < partition.particle_bucket_sizes[src_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
		int advection_source_blockno;
		int source_pidib;
		ivec3 base_index;
		{
			int advect = partition.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
			dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, base_index);
			base_index += blockid;
			advection_source_blockno = prev_partition.query(base_index);
			source_pidib			 = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
			advection_source_blockno = prev_partition.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY;
		}
		vec3 pos;
		float J;
		{
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			pos[0]					 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
			pos[1]					 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
			pos[2]					 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
			J						 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
		}
		ivec3 local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		vec3 local_pos		   = pos - local_base_index * config::G_DX;
		base_index			   = local_base_index;

		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			float d	   = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0) = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1)			 = 0.75f - d * d;
			d					 = 0.5f + d;
			dws(dd, 2)			 = 0.5f * d * d;
			local_base_index[dd] = ((local_base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}
		vec3 vel;
		vel.set(0.f);
		vec9 C;
		C.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W	  = dws(0, i) * dws(1, j) * dws(2, k);
					vec3 vi {g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)]};
					vel += vi * W;
					C[0] += W * vi[0] * xixp[0];
					C[1] += W * vi[1] * xixp[0];
					C[2] += W * vi[2] * xixp[0];
					C[3] += W * vi[0] * xixp[1];
					C[4] += W * vi[1] * xixp[1];
					C[5] += W * vi[2] * xixp[1];
					C[6] += W * vi[0] * xixp[2];
					C[7] += W * vi[1] * xixp[2];
					C[8] += W * vi[2] * xixp[2];
				}
			}
		}
		pos += vel * dt;

		J = (1 + (C[0] + C[4] + C[8]) * dt * config::G_D_INV) * J;
		if(J < 0.1) {
			J = 0.1;
		}
		vec9 contrib;
		{
			float voln	   = J * particle_buffer.volume;
			float pressure = particle_buffer.bulk * (powf(J, -particle_buffer.gamma) - 1.f);
			{
				contrib[0] = ((C[0] + C[0]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
				contrib[1] = (C[1] + C[3]) * config::G_D_INV * particle_buffer.viscosity * voln;
				contrib[2] = (C[2] + C[6]) * config::G_D_INV * particle_buffer.viscosity * voln;

				contrib[3] = (C[3] + C[1]) * config::G_D_INV * particle_buffer.viscosity * voln;
				contrib[4] = ((C[4] + C[4]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
				contrib[5] = (C[5] + C[7]) * config::G_D_INV * particle_buffer.viscosity * voln;

				contrib[6] = (C[6] + C[2]) * config::G_D_INV * particle_buffer.viscosity * voln;
				contrib[7] = (C[7] + C[5]) * config::G_D_INV * particle_buffer.viscosity * voln;
				contrib[8] = ((C[8] + C[8]) * config::G_D_INV * particle_buffer.viscosity - pressure) * voln;
			}
			contrib = (C * particle_buffer.mass - contrib * new_dt) * config::G_D_INV;
			{
				auto particle_bin													= next_particle_buffer.ch(_0, partition.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
				particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY) = pos[0];
				particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY) = pos[1];
				particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY) = pos[2];
				particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY) = J;
			}
		}

		local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		{
			int dirtag = dir_offset((base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (local_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE));
			partition.add_advection(local_base_index - 1, dirtag, particle_id_in_block);
		}
		// dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			local_pos[dd] = pos[dd] - static_cast<float>(local_base_index[dd]) * config::G_DX;
			float d		  = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0)	  = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1) = 0.75f - d * d;
			d		   = 0.5f + d;
			dws(dd, 2) = 0.5f * d * d;

			local_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + local_base_index[dd] - base_index[dd];
		}
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					pos		= vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W = dws(0, i) * dws(1, j) * dws(2, k);
					auto wm = particle_buffer.mass * W;
					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();
	/// arena no, channel no, cell no
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		char c		  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

template<typename Partition, typename Grid>
__global__ void g2p2g(float dt, float new_dt, const ivec3* __restrict__ blocks, const ParticleBuffer<MaterialE::FIXED_COROTATED> particle_buffer, ParticleBuffer<MaterialE::FIXED_COROTATED> next_particle_buffer, const Partition prev_partition, Partition partition, const Grid grid, Grid next_grid) {
	static constexpr uint64_t NUM_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 3;
	static constexpr uint64_t NUM_VI_IN_ARENA  = NUM_VI_PER_BLOCK << 3;

	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using ViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>*;
	using ViArenaRef  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>&;
	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	extern __shared__ char shmem[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	ViArenaRef __restrict__ g2pbuffer  = *static_cast<ViArena>(static_cast<void*>(static_cast<char*>(shmem)));
	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem) + NUM_VI_IN_ARENA * sizeof(float)));

	ivec3 blockid;
	int src_blockno;
	if(blocks != nullptr) {
		blockid		= blocks[blockIdx.x];
		src_blockno = partition.query(blockid);
	} else {
		if(partition.halo_marks[blockIdx.x]) {
			return;
		}
		blockid = partition.active_keys[blockIdx.x];

		int src_blockno			 = static_cast<int>(blockIdx.x);
		int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
		if(particle_bucket_size == 0) {
			return;
		}
	}

	for(int base = static_cast<int>(threadIdx.x); base < NUM_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		auto grid_block		= grid.ch(_0, blockno);
		int channelid		= static_cast<int>(base % NUM_VI_PER_BLOCK);
		char c				= static_cast<char>(channelid & 0x3f);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}
		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc = static_cast<int>(base);

		char z = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char x								 = static_cast<char>(loc & ARENAMASK);
		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < partition.particle_bucket_sizes[src_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
		int advection_source_blockno;
		int source_pidib;
		ivec3 base_index;
		{
			int advect = partition.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
			dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, base_index);
			base_index += blockid;
			advection_source_blockno = prev_partition.query(base_index);
			source_pidib			 = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
			advection_source_blockno = prev_partition.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY;
		}
		vec3 pos;
		{
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			pos[0]					 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
			pos[1]					 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
			pos[2]					 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
		}
		ivec3 local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		vec3 local_pos		   = pos - local_base_index * config::G_DX;
		base_index			   = local_base_index;

		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			float d	   = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0) = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1)			 = 0.75f - d * d;
			d					 = 0.5f + d;
			dws(dd, 2)			 = 0.5f * d * d;
			local_base_index[dd] = ((local_base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}
		vec3 vel;
		vel.set(0.f);
		vec9 C;
		C.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W	  = dws(0, i) * dws(1, j) * dws(2, k);
					vec3 vi {g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)]};
					vel += vi * W;
					C[0] += W * vi[0] * xixp[0];
					C[1] += W * vi[1] * xixp[0];
					C[2] += W * vi[2] * xixp[0];
					C[3] += W * vi[0] * xixp[1];
					C[4] += W * vi[1] * xixp[1];
					C[5] += W * vi[2] * xixp[1];
					C[6] += W * vi[0] * xixp[2];
					C[7] += W * vi[1] * xixp[2];
					C[8] += W * vi[2] * xixp[2];
				}
			}
		}
		pos += vel * dt;

#pragma unroll 9
		for(int d = 0; d < 9; ++d) {
			dws.val(d) = C[d] * dt * config::G_D_INV + ((d & 0x3) ? 0.f : 1.f);
		}

		vec9 contrib;
		{
			vec9 F;
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			contrib[0]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
			contrib[1]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
			contrib[2]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
			contrib[3]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
			contrib[4]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
			contrib[5]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
			contrib[6]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
			contrib[7]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
			contrib[8]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
			matrix_matrix_multiplication_3d(dws.data_arr(), contrib.data_arr(), F.data_arr());
			{
				auto particle_bin													 = next_particle_buffer.ch(_0, partition.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
				particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[0];
				particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[1];
				particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[2];
				particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[0];
				particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[1];
				particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[2];
				particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[3];
				particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[4];
				particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[5];
				particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[6];
				particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = F[7];
				particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = F[8];
			}
			compute_stress_fixed_corotated(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, F, contrib);
			contrib = (C * particle_buffer.mass - contrib * new_dt) * config::G_D_INV;
		}

		local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		{
			int dirtag = dir_offset((base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (local_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE));
			partition.add_advection(local_base_index - 1, dirtag, particle_id_in_block);
		}
		// dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			local_pos[dd] = pos[dd] - static_cast<float>(local_base_index[dd]) * config::G_DX;
			float d		  = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0)	  = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1) = 0.75f - d * d;
			d		   = 0.5f + d;
			dws(dd, 2) = 0.5f * d * d;

			local_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + local_base_index[dd] - base_index[dd];
		}
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					pos		= vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W = dws(0, i) * dws(1, j) * dws(2, k);
					auto wm = particle_buffer.mass * W;
					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();
	/// arena no, channel no, cell no
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		char c		  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;
		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

template<typename Partition, typename Grid>
__global__ void g2p2g(float dt, float new_dt, const ivec3* __restrict__ blocks, const ParticleBuffer<MaterialE::SAND> particle_buffer, ParticleBuffer<MaterialE::SAND> next_particle_buffer, const Partition prev_partition, Partition partition, const Grid grid, Grid next_grid) {
	static constexpr uint64_t NUM_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 3;
	static constexpr uint64_t NUM_VI_IN_ARENA  = NUM_VI_PER_BLOCK << 3;

	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using ViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>*;
	using ViArenaRef  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>&;
	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	extern __shared__ char shmem[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	ViArenaRef __restrict__ g2pbuffer  = *static_cast<ViArena>(static_cast<void*>(static_cast<char*>(shmem)));
	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem) + NUM_VI_IN_ARENA * sizeof(float)));

	ivec3 blockid;
	int src_blockno;
	if(blocks != nullptr) {
		blockid		= blocks[blockIdx.x];
		src_blockno = partition.query(blockid);
	} else {
		if(partition.halo_marks[blockIdx.x]) {
			return;
		}
		blockid = partition.active_keys[blockIdx.x];

		int src_blockno			 = static_cast<int>(blockIdx.x);
		int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
		if(particle_bucket_size == 0) {
			return;
		}
	}

	for(int base = static_cast<int>(threadIdx.x); base < NUM_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		auto grid_block		= grid.ch(_0, blockno);
		int channelid		= static_cast<int>(base % NUM_VI_PER_BLOCK);
		char c				= static_cast<char>(channelid & 0x3f);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}
		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc = static_cast<int>(base);
		char z	= static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char x								 = static_cast<char>(loc & ARENAMASK);
		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < partition.particle_bucket_sizes[src_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
		int advection_source_blockno;
		int source_pidib;
		ivec3 base_index;
		{
			int advect = partition.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
			dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, base_index);
			base_index += blockid;
			advection_source_blockno = prev_partition.query(base_index);
			source_pidib			 = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
			advection_source_blockno = prev_partition.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY;
		}
		vec3 pos;
		{
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			pos[0]					 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
			pos[1]					 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
			pos[2]					 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
		}
		ivec3 local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		vec3 local_pos		   = pos - local_base_index * config::G_DX;
		base_index			   = local_base_index;

		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			float d	   = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0) = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1)			 = 0.75f - d * d;
			d					 = 0.5f + d;
			dws(dd, 2)			 = 0.5f * d * d;
			local_base_index[dd] = ((local_base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}
		vec3 vel;
		vel.set(0.f);
		vec9 C;
		C.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W	  = dws(0, i) * dws(1, j) * dws(2, k);
					vec3 vi {g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)]};
					vel += vi * W;
					C[0] += W * vi[0] * xixp[0];
					C[1] += W * vi[1] * xixp[0];
					C[2] += W * vi[2] * xixp[0];
					C[3] += W * vi[0] * xixp[1];
					C[4] += W * vi[1] * xixp[1];
					C[5] += W * vi[2] * xixp[1];
					C[6] += W * vi[0] * xixp[2];
					C[7] += W * vi[1] * xixp[2];
					C[8] += W * vi[2] * xixp[2];
				}
			}
		}
		pos += vel * dt;

#pragma unroll 9
		for(int d = 0; d < 9; ++d) {
			dws.val(d) = C[d] * dt * config::G_D_INV + ((d & 0x3) ? 0.f : 1.f);
		}

		vec9 contrib;
		{
			vec9 F;
			float log_jp;
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			contrib[0]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
			contrib[1]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
			contrib[2]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
			contrib[3]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
			contrib[4]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
			contrib[5]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
			contrib[6]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
			contrib[7]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
			contrib[8]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
			log_jp					 = source_particle_bin.val(_12, source_pidib % config::G_BIN_CAPACITY);

			matrix_matrix_multiplication_3d(dws.data_arr(), contrib.data_arr(), F.data_arr());
			compute_stress_sand(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, particle_buffer.cohesion, particle_buffer.beta, particle_buffer.yield_surface, particle_buffer.volume_correction, log_jp, F, contrib);
			{
				auto particle_bin													 = next_particle_buffer.ch(_0, partition.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
				particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[0];
				particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[1];
				particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[2];
				particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[0];
				particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[1];
				particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[2];
				particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[3];
				particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[4];
				particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[5];
				particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[6];
				particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = F[7];
				particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = F[8];
				particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = log_jp;
			}

			contrib = (C * particle_buffer.mass - contrib * new_dt) * config::G_D_INV;
		}

		local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		{
			int dirtag = dir_offset((base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (local_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE));
			partition.add_advection(local_base_index - 1, dirtag, particle_id_in_block);
		}
		// dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			local_pos[dd] = pos[dd] - static_cast<float>(local_base_index[dd]) * config::G_DX;
			float d		  = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0)	  = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1) = 0.75f - d * d;
			d		   = 0.5f + d;
			dws(dd, 2) = 0.5f * d * d;

			local_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + local_base_index[dd] - base_index[dd];
		}
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					pos		= vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W = dws(0, i) * dws(1, j) * dws(2, k);
					auto wm = particle_buffer.mass * W;
					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();
	/// arena no, channel no, cell no
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		char c		  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

template<typename Partition, typename Grid>
__global__ void g2p2g(float dt, float new_dt, const ivec3* __restrict__ blocks, const ParticleBuffer<MaterialE::NACC> particle_buffer, ParticleBuffer<MaterialE::NACC> next_particle_buffer, const Partition prev_partition, Partition partition, const Grid grid, Grid next_grid) {
	static constexpr uint64_t NUM_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 3;
	static constexpr uint64_t NUM_VI_IN_ARENA  = NUM_VI_PER_BLOCK << 3;

	static constexpr uint64_t NUM_M_VI_PER_BLOCK = static_cast<uint64_t>(config::G_BLOCKVOLUME) * 4;
	static constexpr uint64_t NUM_M_VI_IN_ARENA	 = NUM_M_VI_PER_BLOCK << 3;

	static constexpr unsigned ARENAMASK = (config::G_BLOCKSIZE << 1) - 1;
	static constexpr unsigned ARENABITS = config::G_BLOCKBITS + 1;

	using ViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>*;
	using ViArenaRef  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 3>&;
	using MViArena	  = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>*;
	using MViArenaRef = std::array<std::array<std::array<std::array<float, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, config::G_BLOCKSIZE << 1>, 4>&;

	extern __shared__ char shmem[];//NOLINT(modernize-avoid-c-arrays, readability-redundant-declaration) Cannot declare runtime size shared memory as std::array; extern has different meaning here

	ViArenaRef __restrict__ g2pbuffer  = *static_cast<ViArena>(static_cast<void*>(static_cast<char*>(shmem)));
	MViArenaRef __restrict__ p2gbuffer = *static_cast<MViArena>(static_cast<void*>(static_cast<char*>(shmem) + NUM_VI_IN_ARENA * sizeof(float)));

	ivec3 blockid;
	int src_blockno;
	if(blocks != nullptr) {
		blockid		= blocks[blockIdx.x];
		src_blockno = partition.query(blockid);
	} else {
		if(partition.halo_marks[blockIdx.x]) {
			return;
		}
		blockid = partition.active_keys[blockIdx.x];

		int src_blockno			 = static_cast<int>(blockIdx.x);
		int particle_bucket_size = next_particle_buffer.particle_bucket_sizes[src_blockno];
		if(particle_bucket_size == 0) {
			return;
		}
	}

	for(int base = static_cast<int>(threadIdx.x); base < NUM_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		auto grid_block		= grid.ch(_0, blockno);
		int channelid		= static_cast<int>(base % NUM_VI_PER_BLOCK);
		char c				= static_cast<char>(channelid & 0x3f);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val;
		if(channelid == 0) {
			val = grid_block.val_1d(_1, c);
		} else if(channelid == 1) {
			val = grid_block.val_1d(_2, c);
		} else {
			val = grid_block.val_1d(_3, c);
		}
		g2pbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))] = val;
	}
	__syncthreads();
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		int loc = static_cast<int>(base);
		char z	= static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char y = static_cast<char>(loc & ARENAMASK);
		loc >>= ARENABITS;

		char x								 = static_cast<char>(loc & ARENAMASK);
		p2gbuffer[loc >> ARENABITS][x][y][z] = 0.0f;
	}
	__syncthreads();

	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < partition.particle_bucket_sizes[src_blockno]; particle_id_in_block += static_cast<int>(blockDim.x)) {
		int advection_source_blockno;
		int source_pidib;
		ivec3 base_index;
		{
			int advect = partition.blockbuckets[src_blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
			dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, base_index);
			base_index += blockid;
			advection_source_blockno = prev_partition.query(base_index);
			source_pidib			 = advect & (config::G_PARTICLE_NUM_PER_BLOCK - 1);
			advection_source_blockno = prev_partition.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY;
		}
		vec3 pos;
		{
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			pos[0]					 = source_particle_bin.val(_0, source_pidib % config::G_BIN_CAPACITY);
			pos[1]					 = source_particle_bin.val(_1, source_pidib % config::G_BIN_CAPACITY);
			pos[2]					 = source_particle_bin.val(_2, source_pidib % config::G_BIN_CAPACITY);
		}
		ivec3 local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		vec3 local_pos		   = pos - local_base_index * config::G_DX;
		base_index			   = local_base_index;

		vec3x3 dws;
#pragma unroll 3
		for(int dd = 0; dd < 3; ++dd) {
			float d	   = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0) = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1)			 = 0.75f - d * d;
			d					 = 0.5f + d;
			dws(dd, 2)			 = 0.5f * d * d;
			local_base_index[dd] = ((local_base_index[dd] - 1) & config::G_BLOCKMASK) + 1;
		}
		vec3 vel;
		vel.set(0.f);
		vec9 C;
		C.set(0.f);
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					vec3 xixp = vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W	  = dws(0, i) * dws(1, j) * dws(2, k);
					vec3 vi {g2pbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], g2pbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)]};
					vel += vi * W;
					C[0] += W * vi[0] * xixp[0];
					C[1] += W * vi[1] * xixp[0];
					C[2] += W * vi[2] * xixp[0];
					C[3] += W * vi[0] * xixp[1];
					C[4] += W * vi[1] * xixp[1];
					C[5] += W * vi[2] * xixp[1];
					C[6] += W * vi[0] * xixp[2];
					C[7] += W * vi[1] * xixp[2];
					C[8] += W * vi[2] * xixp[2];
				}
			}
		}
		pos += vel * dt;

#pragma unroll 9
		for(int d = 0; d < 9; ++d) {
			dws.val(d) = C[d] * dt * config::G_D_INV + ((d & 0x3) ? 0.f : 1.f);
		}

		vec9 contrib;
		{
			vec9 F;
			float log_jp;
			auto source_particle_bin = particle_buffer.ch(_0, advection_source_blockno);
			contrib[0]				 = source_particle_bin.val(_3, source_pidib % config::G_BIN_CAPACITY);
			contrib[1]				 = source_particle_bin.val(_4, source_pidib % config::G_BIN_CAPACITY);
			contrib[2]				 = source_particle_bin.val(_5, source_pidib % config::G_BIN_CAPACITY);
			contrib[3]				 = source_particle_bin.val(_6, source_pidib % config::G_BIN_CAPACITY);
			contrib[4]				 = source_particle_bin.val(_7, source_pidib % config::G_BIN_CAPACITY);
			contrib[5]				 = source_particle_bin.val(_8, source_pidib % config::G_BIN_CAPACITY);
			contrib[6]				 = source_particle_bin.val(_9, source_pidib % config::G_BIN_CAPACITY);
			contrib[7]				 = source_particle_bin.val(_10, source_pidib % config::G_BIN_CAPACITY);
			contrib[8]				 = source_particle_bin.val(_11, source_pidib % config::G_BIN_CAPACITY);
			log_jp					 = source_particle_bin.val(_12, source_pidib % config::G_BIN_CAPACITY);

			matrix_matrix_multiplication_3d(dws.data_arr(), contrib.data_arr(), F.data_arr());
			compute_stress_nacc(particle_buffer.volume, particle_buffer.mu, particle_buffer.lambda, particle_buffer.bm, particle_buffer.xi, particle_buffer.beta, particle_buffer.msqr, particle_buffer.hardening_on, log_jp, F, contrib);
			{
				auto particle_bin													 = next_particle_buffer.ch(_0, partition.bin_offsets[src_blockno] + particle_id_in_block / config::G_BIN_CAPACITY);
				particle_bin.val(_0, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[0];
				particle_bin.val(_1, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[1];
				particle_bin.val(_2, particle_id_in_block % config::G_BIN_CAPACITY)	 = pos[2];
				particle_bin.val(_3, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[0];
				particle_bin.val(_4, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[1];
				particle_bin.val(_5, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[2];
				particle_bin.val(_6, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[3];
				particle_bin.val(_7, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[4];
				particle_bin.val(_8, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[5];
				particle_bin.val(_9, particle_id_in_block % config::G_BIN_CAPACITY)	 = F[6];
				particle_bin.val(_10, particle_id_in_block % config::G_BIN_CAPACITY) = F[7];
				particle_bin.val(_11, particle_id_in_block % config::G_BIN_CAPACITY) = F[8];
				particle_bin.val(_12, particle_id_in_block % config::G_BIN_CAPACITY) = log_jp;
			}

			contrib = (C * particle_buffer.mass - contrib * new_dt) * config::G_D_INV;
		}

		local_base_index = (pos * config::G_DX_INV + 0.5f).cast<int>() - 1;
		{
			int dirtag = dir_offset((base_index - 1) / static_cast<int>(config::G_BLOCKSIZE) - (local_base_index - 1) / static_cast<int>(config::G_BLOCKSIZE));
			partition.add_advection(local_base_index - 1, dirtag, particle_id_in_block);
		}
		// dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
		for(char dd = 0; dd < 3; ++dd) {
			local_pos[dd] = pos[dd] - static_cast<float>(local_base_index[dd]) * config::G_DX;
			float d		  = (local_pos[dd] - static_cast<float>(std::floor(local_pos[dd] * config::G_DX_INV + 0.5f) - 1) * config::G_DX) * config::G_DX_INV;
			dws(dd, 0)	  = 0.5f * (1.5f - d) * (1.5f - d);
			d -= 1.0f;
			dws(dd, 1) = 0.75f - d * d;
			d		   = 0.5f + d;
			dws(dd, 2) = 0.5f * d * d;

			local_base_index[dd] = (((base_index[dd] - 1) & config::G_BLOCKMASK) + 1) + local_base_index[dd] - base_index[dd];
		}
#pragma unroll 3
		for(char i = 0; i < 3; i++) {
#pragma unroll 3
			for(char j = 0; j < 3; j++) {
#pragma unroll 3
				for(char k = 0; k < 3; k++) {
					pos		= vec3 {static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)} * config::G_DX - local_pos;
					float W = dws(0, i) * dws(1, j) * dws(2, k);
					auto wm = particle_buffer.mass * W;
					atomicAdd(&p2gbuffer[0][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm);
					atomicAdd(&p2gbuffer[1][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] + contrib[6] * pos[2]) * W);
					atomicAdd(&p2gbuffer[2][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] + contrib[7] * pos[2]) * W);
					atomicAdd(&p2gbuffer[3][static_cast<size_t>(static_cast<size_t>(local_base_index[0]) + i)][static_cast<size_t>(static_cast<size_t>(local_base_index[1]) + j)][static_cast<size_t>(static_cast<size_t>(local_base_index[2]) + k)], wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] + contrib[8] * pos[2]) * W);
				}
			}
		}
	}
	__syncthreads();
	/// arena no, channel no, cell no
	for(int base = static_cast<int>(threadIdx.x); base < NUM_M_VI_IN_ARENA; base += static_cast<int>(blockDim.x)) {
		char local_block_id = static_cast<char>(base / NUM_M_VI_PER_BLOCK);
		auto blockno		= partition.query(ivec3 {blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0), blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0), blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
		// auto grid_block = next_grid.template ch<0>(blockno);
		int channelid = static_cast<int>(base & (NUM_M_VI_PER_BLOCK - 1));
		char c		  = static_cast<char>(channelid % config::G_BLOCKVOLUME);

		char cz = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cy = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		char cx = static_cast<char>(channelid & config::G_BLOCKMASK);
		channelid >>= config::G_BLOCKBITS;

		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::G_BLOCKSIZE : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::G_BLOCKSIZE : 0))];
		if(channelid == 0) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
		} else if(channelid == 1) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
		} else if(channelid == 2) {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
		} else {
			atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
		}
	}
}

template<typename Grid>
__global__ void mark_active_grid_blocks(uint32_t block_count, const Grid grid, int* marks) {
	auto idx	= blockIdx.x * blockDim.x + threadIdx.x;
	int blockno = static_cast<int>(idx / config::G_BLOCKVOLUME);
	int cellno	= static_cast<int>(idx % config::G_BLOCKVOLUME);
	if(blockno >= block_count) {
		return;
	}
	if(grid.ch(_0, blockno).val_1d(_0, cellno) != 0.f) {
		marks[blockno] = 1;
	}
}

__global__ void mark_active_particle_blocks(uint32_t block_count, const int* __restrict__ particle_bucket_sizes, int* marks) {
	const std::size_t blockno = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	marks[blockno] = particle_bucket_sizes[blockno] > 0 ? 1 : 0;
}

template<typename Partition>
__global__ void update_partition(uint32_t block_count, const int* __restrict__ source_nos, const Partition partition, Partition next_partition) {
	__shared__ std::size_t source_no[1];//NOLINT(modernize-avoid-c-arrays) Cannot declare shared memory as std::array?
	std::size_t blockno = blockIdx.x;
	if(blockno >= block_count) {
		return;
	}
	if(threadIdx.x == 0) {
		source_no[0]						= source_nos[blockno];
		auto source_blockid					= partition.active_keys[source_no[0]];
		next_partition.active_keys[blockno] = source_blockid;
		next_partition.reinsert(static_cast<int>(blockno));
		next_partition.particle_bucket_sizes[blockno] = partition.particle_bucket_sizes[source_no[0]];
	}
	__syncthreads();

	auto particle_counts = next_partition.particle_bucket_sizes[blockno];
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		next_partition.blockbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block] = partition.blockbuckets[source_no[0] * config::G_PARTICLE_NUM_PER_BLOCK + particle_id_in_block];
	}
}

template<typename Partition, typename Grid>
__global__ void copy_selected_grid_blocks(const ivec3* __restrict__ prev_blockids, const Partition partition, const int* __restrict__ marks, Grid prev_grid, Grid grid) {
	auto blockid = prev_blockids[blockIdx.x];
	if(marks[blockIdx.x]) {
		auto blockno = partition.query(blockid);
		if(blockno == -1) {
			return;
		}
		auto sourceblock					= prev_grid.ch(_0, blockIdx.x);
		auto targetblock					= grid.ch(_0, blockno);
		targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x);
		targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x);
		targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x);
		targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x);
	}
}

template<typename Partition>
__global__ void check_table(uint32_t block_count, Partition partition) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	auto blockid = partition.active_keys[blockno];
	if(partition.query(blockid) != blockno)
		printf("FUCK, partition table is wrong!\n");
}

template<typename Grid>
__global__ void sum_grid_mass(Grid grid, float* sum) {
	atomicAdd(sum, grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}

__global__ void sum_particle_counts(uint32_t count, int* __restrict__ counts, int* sum) {
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count) {
		return;
	}
	atomicAdd(sum, counts[idx]);
}

template<typename Partition>
__global__ void check_partition(uint32_t block_count, Partition partition) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= block_count) {
		return;
	}
	ivec3 blockid = partition.active_keys[idx];
	if(blockid[0] == 0 || blockid[1] == 0 || blockid[2] == 0) {
		printf("\tDAMN, encountered zero block record\n");
	}
	if(partition.query(blockid) != idx) {
		int id	  = partition.query(blockid);
		ivec3 bid = partition.active_keys[id];
		printf(
			"\t\tcheck partition %d, (%d, %d, %d), feedback index %d, (%d, %d, "
			"%d)\n",
			idx,
			(int) blockid[0],
			(int) blockid[1],
			(int) blockid[2],
			id,
			bid[0],
			bid[1],
			bid[2]
		);
	}
}

template<typename Partition, typename Domain>
__global__ void check_partition_domain(uint32_t block_count, int did, Domain const domain, Partition partition) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= block_count) {
		return;
	}
	ivec3 blockid = partition.active_keys[idx];
	if(domain.inside(blockid)) {
		printf("%d-th block (%d, %d, %d) is in domain[%d] (%d, %d, %d)-(%d, %d, %d)\n", idx, blockid[0], blockid[1], blockid[2], did, domain._min[0], domain._min[1], domain._min[2], domain._max[0], domain._max[1], domain._max[2]);
	}
}

template<typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void retrieve_particle_buffer(Partition partition, Partition prev_partition, ParticleBuffer particle_buffer, ParticleArray particle_array, int* parcount) {
	int particle_counts	  = partition.particle_bucket_sizes[blockIdx.x];
	ivec3 blockid		  = partition.active_keys[blockIdx.x];
	auto advection_bucket = partition.blockbuckets + blockIdx.x * config::G_PARTICLE_NUM_PER_BLOCK;
	// auto particle_offset = partition.bin_offsets[blockIdx.x];
	for(int particle_id_in_block = static_cast<int>(threadIdx.x); particle_id_in_block < particle_counts; particle_id_in_block += static_cast<int>(blockDim.x)) {
		auto advect = advection_bucket[particle_id_in_block];
		ivec3 source_blockid;
		dir_components(advect / config::G_PARTICLE_NUM_PER_BLOCK, source_blockid);
		source_blockid += blockid;
		auto advection_source_blockno = prev_partition.query(source_blockid);
		auto source_pidib			  = advect % config::G_PARTICLE_NUM_PER_BLOCK;
		auto source_bin				  = particle_buffer.ch(_0, prev_partition.bin_offsets[advection_source_blockno] + source_pidib / config::G_BIN_CAPACITY);
		auto _source_pidib			  = source_pidib % config::G_BIN_CAPACITY;

		auto particle_id = atomicAdd(parcount, 1);
		/// pos
		particle_array.val(_0, particle_id) = source_bin.val(_0, _source_pidib);
		particle_array.val(_1, particle_id) = source_bin.val(_1, _source_pidib);
		particle_array.val(_2, particle_id) = source_bin.val(_2, _source_pidib);
	}
}
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, readability-identifier-naming, misc-definitions-in-headers)
}// namespace mn

#endif