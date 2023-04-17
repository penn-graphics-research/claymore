#ifndef HALO_BUFFER_CUH
#define HALO_BUFFER_CUH
#include <MnBase/Meta/Polymorphism.h>

#include "grid_buffer.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
//#include <cub/device/device_scan.cuh>

namespace mn {

using HaloGridBlocksDomain = CompactDomain<int, config::G_MAX_HALO_BLOCK>;
using halo_grid_blocks_	   = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, HaloGridBlocksDomain, attrib_layout::SOA, grid_block_>;

using grid_block_ = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;

/// halo grid block
struct HaloGridBlocks {
	struct HaloBuffer {
		Instance<halo_grid_blocks_> grid;
		ivec3* blockids;
	};

	int num_targets;
	uint32_t* counts;
	std::vector<uint32_t> h_counts;
	std::vector<HaloBuffer> buffers;

	explicit HaloGridBlocks(int num_neighbors)
		: num_targets {num_neighbors}
		, counts(nullptr)
		, h_counts(num_neighbors, 0) {
		check_cuda_errors(cudaMalloc(&counts, sizeof(uint32_t) * num_targets));
		buffers.resize(num_targets);
	}
	template<typename Allocator>
	void init_blocks(Allocator allocator, uint32_t block_count) {
		for(int did = 0; did < num_targets; ++did) {
			buffers[did].blockids = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * block_count));
		}
	}
	template<typename Allocator>
	void init_buffer(Allocator allocator, std::vector<uint32_t> counts) {
		for(int did = 0; did < num_targets; ++did) {
			buffers[did].grid.allocate_handle(allocator, counts[did]);
		}
	}
	void reset_counts(cudaStream_t stream) {
		check_cuda_errors(cudaMemsetAsync(counts, 0, sizeof(uint32_t) * num_targets, stream));
	}
	void retrieve_counts(cudaStream_t stream) {
		check_cuda_errors(cudaMemcpyAsync(h_counts.data(), counts, sizeof(uint32_t) * num_targets, cudaMemcpyDefault, stream));
	}
	void send(HaloGridBlocks& other, int src, int dst, cudaStream_t stream) {
		auto cnt = other.h_counts[src] = h_counts[dst];
		//check_cuda_errors(cudaMemcpyAsync( &other.buffers[src].val(_1, 0), &buffers[dst].val(_1, 0), sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.buffers[src].blockids, buffers[dst].blockids, sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
		//check_cuda_errors(cudaMemcpyAsync(&other.buffers[src].grid.ch(_0, 0).val_1d(_0, 0), &buffers[dst].grid.ch(_0, 0).val_1d(_0, 0), grid_block_::size * cnt, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyPeerAsync(&other.buffers[src].grid.ch(_0, 0).val_1d(_0, 0), dst, &buffers[dst].grid.ch(_0, 0).val_1d(_0, 0), src, grid_block_::size * cnt, stream));
		// printf("sending from %d to %d at %llu\n", src, dst,
		//       (unsigned long long)&other.buffers[src].grid.ch(_0, 0).val_1d(_0,
		//       0));
	}
};

}// namespace mn

#endif