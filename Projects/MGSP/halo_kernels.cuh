#ifndef HALO_KERNELS_CUH
#define HALO_KERNELS_CUH

#include <MnBase/Math/Matrix/MatrixUtils.h>

#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnSystem/Cuda/DeviceUtils.cuh>

#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"

namespace mn {

using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//TODO: Make magic numbers to constants where suitable
//TODO: Ensure call dimensions and such are small enough to allow narrowing conversations. Or directly use unsigned where possible
//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, misc-definitions-in-headers) CUDA does not yet support std::span; Cannot declare __global__ functions inline
template<typename Partition, typename HaloGridBlocks>
__global__ void mark_overlapping_blocks(uint32_t block_count, int otherdid, const ivec3* __restrict__ incoming_block_ids, Partition partition, uint32_t* count, HaloGridBlocks halo_grid_blocks) {
	uint32_t inc_blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(inc_blockno >= block_count) {
		return;
	}
	auto inc_blockid = incoming_block_ids[inc_blockno];
	auto blockno	 = partition.query(inc_blockid);
	if(blockno >= 0) {
		atomicOr(partition.overlap_marks + blockno, 1 << otherdid);
		auto halono = atomicAdd(count, 1);
		// halo_grid_blocks.val(_1, halono) = inc_blockid;
		halo_grid_blocks.blockids[halono] = inc_blockid;
	}
}

template<typename Partition>
__global__ void collect_blockids_for_halo_reduction(uint32_t particle_block_count, int did, Partition partition) {
	(void) did;

	std::size_t blockno = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if(blockno >= particle_block_count) {
		return;
	}
	auto blockid				  = partition.active_keys[blockno];
	partition.halo_marks[blockno] = 0;
	for(char i = 0; i < 2; ++i) {
		for(char j = 0; j < 2; ++j) {
			for(char k = 0; k < 2; ++k) {
				ivec3 neighborid {blockid[0] + i, blockid[1] + j, blockid[2] + k};
				int neighborno = partition.query(neighborid);
				// if (partition.overlap_marks[neighborno] ^ ((HaloIndex)1 << did)) {
				if(partition.overlap_marks[neighborno]) {
					partition.halo_marks[blockno] = 1;
					auto halono					  = atomicAdd(partition.halo_count, 1);
					partition.halo_blocks[halono] = blockid;
					return;
				}
			}
		}
	}
}

template<typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void collect_grid_blocks(Grid grid, Partition partition, HaloGridBlocks halo_grid_blocks) {
	uint32_t halo_blockno = blockIdx.x;
	// auto halo_blockid = halo_grid_blocks.grid.val(_1, halo_blockno);
	auto halo_blockid = halo_grid_blocks.blockids[halo_blockno];

	auto blockno		= partition.query(halo_blockid);
	auto halo_gridblock = halo_grid_blocks.grid.ch(_0, halo_blockno);
	auto gridblock		= grid.ch(_0, blockno);

	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		halo_gridblock.val_1d(_0, cell_id_in_block) = gridblock.val_1d(_0, cell_id_in_block);
		halo_gridblock.val_1d(_1, cell_id_in_block) = gridblock.val_1d(_1, cell_id_in_block);
		halo_gridblock.val_1d(_2, cell_id_in_block) = gridblock.val_1d(_2, cell_id_in_block);
		halo_gridblock.val_1d(_3, cell_id_in_block) = gridblock.val_1d(_3, cell_id_in_block);
	}
}

template<typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void reduce_grid_blocks(Grid grid, Partition partition, HaloGridBlocks halo_grid_blocks) {
	uint32_t halo_blockno = blockIdx.x;
	// auto halo_blockid = halo_grid_blocks.grid.val(_1, halo_blockno);
	auto halo_blockid	= halo_grid_blocks.blockids[halo_blockno];
	auto blockno		= partition.query(halo_blockid);
	auto halo_gridblock = halo_grid_blocks.grid.ch(_0, halo_blockno);
	auto gridblock		= grid.ch(_0, blockno);

	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		atomicAdd(&gridblock.val_1d(_0, cell_id_in_block), halo_gridblock.val_1d(_0, cell_id_in_block));
		atomicAdd(&gridblock.val_1d(_1, cell_id_in_block), halo_gridblock.val_1d(_1, cell_id_in_block));
		atomicAdd(&gridblock.val_1d(_2, cell_id_in_block), halo_gridblock.val_1d(_2, cell_id_in_block));
		atomicAdd(&gridblock.val_1d(_3, cell_id_in_block), halo_gridblock.val_1d(_3, cell_id_in_block));
	}
}

template<typename Domain, typename Partition, typename HaloParticleBlocks>
__global__ void mark_migration_grid_blocks(uint32_t block_count, Domain const domain, Partition const partition, uint32_t* count, HaloParticleBlocks halo_particle_blocks, int const* active_grid_block_marks) {
	uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockno >= block_count) {
		return;
	}
	if(active_grid_block_marks[blockno]) {
		auto blockid = partition.active_keys[blockno];
		if(domain.within(blockid, ivec3 {0, 0, 0}, ivec3 {1, 1, 1})) {
			// halo_particle_blocks._binpbs[halono] = 0;
			auto halono								= atomicAdd(count, 1);
			halo_particle_blocks._gblockids[halono] = blockid;
		}
	}
}

template<typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void collect_migration_grid_blocks(Grid grid, Partition partition, HaloGridBlocks halo_grid_blocks) {
	uint32_t halo_blockno = blockIdx.x;
	auto halo_blockid	  = halo_grid_blocks._gblockids[halo_blockno];
	auto halo_gridblock	  = halo_grid_blocks._grid.ch(_0, halo_blockno);

	auto blockno   = partition.query(halo_blockid);
	auto gridblock = grid.ch(_0, blockno);

	for(int cell_id_in_block = static_cast<int>(threadIdx.x); cell_id_in_block < config::G_BLOCKVOLUME; cell_id_in_block += static_cast<int>(blockDim.x)) {
		halo_gridblock.val_1d(_0, cell_id_in_block) = gridblock.val_1d(_0, cell_id_in_block);
		halo_gridblock.val_1d(_1, cell_id_in_block) = gridblock.val_1d(_1, cell_id_in_block);
		halo_gridblock.val_1d(_2, cell_id_in_block) = gridblock.val_1d(_2, cell_id_in_block);
		halo_gridblock.val_1d(_3, cell_id_in_block) = gridblock.val_1d(_3, cell_id_in_block);
	}
}
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers, misc-definitions-in-headers)

}// namespace mn

#endif