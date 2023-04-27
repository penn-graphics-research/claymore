#ifndef GRID_BUFFER_CUH
#define GRID_BUFFER_CUH
#include <MnSystem/Cuda/HostUtils.hpp>

#include "mgmpm_kernels.cuh"
#include "settings.h"

namespace mn {
using namespace placeholder;//NOLINT(google-build-using-namespace) Allow placeholders to be included generally for simplification

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reporst variable errors fro template arguments
using BlockDomain	   = CompactDomain<char, config::G_BLOCKSIZE, config::G_BLOCKSIZE, config::G_BLOCKSIZE>;
using GridDomain	   = CompactDomain<int, config::G_GRID_SIZE, config::G_GRID_SIZE, config::G_GRID_SIZE>;
using GridBufferDomain = CompactDomain<int, config::G_MAX_ACTIVE_BLOCK>;

using grid_block_  = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;
using grid_		   = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridDomain, attrib_layout::AOS, grid_block_>;
using grid_buffer_ = Structural<StructuralType::DYNAMIC, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridBufferDomain, attrib_layout::AOS, grid_block_>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

struct GridBuffer : Instance<grid_buffer_> {
	using base_t = Instance<grid_buffer_>;

	template<typename Allocator>
	explicit GridBuffer(Allocator allocator)
		: base_t {spawn<grid_buffer_, orphan_signature>(allocator)} {}
	template<typename Allocator>
	void check_capacity(Allocator allocator, std::size_t capacity) {
		if(capacity > _capacity)
			resize(capacity, capacity);
	}
	template<typename CudaContext>
	void reset(int block_count, CudaContext& cu_dev) {
		//check_cuda_errors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0, grid_block_::size * block_count, cu_dev.stream_compute()));
		cu_dev.compute_launch({block_count, config::G_BLOCKVOLUME}, clear_grid, *this);
	}
};

}// namespace mn

#endif