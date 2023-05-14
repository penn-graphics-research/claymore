#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH
#include <MnBase/Object/Structural.h>

#include <MnSystem/Cuda/HostUtils.hpp>

#include "settings.h"

namespace mn {

template<int>
struct HaloPartition {
	template<typename Allocator>
	HaloPartition(Allocator allocator, int max_block_count) {
		(void) allocator;
		(void) max_block_count;
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {}
	void copy_to(HaloPartition& other, std::size_t block_count, cudaStream_t stream) {}
};
template<>
struct HaloPartition<1> {
	int* halo_count;
	int h_count;
	char* halo_marks;///< halo particle blocks
	int* overlap_marks;
	ivec3* halo_blocks;

	template<typename Allocator>
	HaloPartition(Allocator allocator, int max_block_count)
		: h_count(0) {
		halo_count	  = static_cast<int*>(allocator.allocate(sizeof(char) * max_block_count));
		halo_marks	  = static_cast<char*>(allocator.allocate(sizeof(char) * max_block_count));
		overlap_marks = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count));
		halo_blocks	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * max_block_count));
	}

	void copy_to(HaloPartition& other, std::size_t block_count, cudaStream_t stream) const {
		other.h_count = h_count;
		check_cuda_errors(cudaMemcpyAsync(other.halo_marks, halo_marks, sizeof(char) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.overlap_marks, overlap_marks, sizeof(int) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.halo_blocks, halo_blocks, sizeof(ivec3) * block_count, cudaMemcpyDefault, stream));
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {
		allocator.deallocate(halo_marks, sizeof(char) * prev_capacity);
		allocator.deallocate(overlap_marks, sizeof(int) * prev_capacity);
		allocator.deallocate(halo_blocks, sizeof(ivec3) * prev_capacity);
		halo_marks	  = static_cast<char*>(allocator.allocate(sizeof(char) * capacity));
		overlap_marks = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));
		halo_blocks	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * capacity));
	}

	void reset_halo_count(cudaStream_t stream) const {
		check_cuda_errors(cudaMemsetAsync(halo_count, 0, sizeof(int), stream));
	}

	void reset_overlap_marks(uint32_t neighbor_block_count, cudaStream_t stream) const {
		check_cuda_errors(cudaMemsetAsync(overlap_marks, 0, sizeof(int) * neighbor_block_count, stream));
	}

	void retrieve_halo_count(cudaStream_t stream) {
		check_cuda_errors(cudaMemcpyAsync(&h_count, halo_count, sizeof(int), cudaMemcpyDefault, stream));
	}
};

//NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming) Check is buggy and reporst variable errors fro template arguments
using block_partition_ = Structural<StructuralType::HASH, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::COMPACT>, GridDomain, attrib_layout::AOS, empty_>;
//NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, readability-identifier-naming)

//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Using pointer arithmetics cause library and allocators do so.
template<int Opt = 1>
struct Partition
	: Instance<block_partition_>
	, HaloPartition<Opt> {
	using base_t	  = Instance<block_partition_>;
	using halo_base_t = HaloPartition<Opt>;
	using block_partition_::key_t;
	using block_partition_::value_t;
	static_assert(sentinel_v == (value_t) (-1), "sentinel value not full 1s\n");

	template<typename Allocator>
	Partition(Allocator allocator, int max_block_count)
		: halo_base_t {allocator, max_block_count} {
		allocate_table(allocator, max_block_count);
		/// init
		reset();
	}

	~Partition() = default;

	Partition(const Partition& other)				 = default;
	Partition(Partition&& other) noexcept			 = default;
	Partition& operator=(const Partition& other)	 = default;
	Partition& operator=(Partition&& other) noexcept = default;

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t capacity) {
		halo_base_t::resize_partition(allocator, this->capacity, capacity);
		resize_table(allocator, capacity);
	}

	void reset() {
		check_cuda_errors(cudaMemset(this->Instance<block_partition_>::count, 0, sizeof(value_t)));
		check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
	}
	void reset_table(cudaStream_t stream) {
		check_cuda_errors(cudaMemsetAsync(this->index_table, 0xff, sizeof(value_t) * domain::extent, stream));
	}
	void copy_to(Partition& other, std::size_t block_count, cudaStream_t stream) {
		halo_base_t::copy_to(other, block_count, stream);
		check_cuda_errors(cudaMemcpyAsync(other.index_table, this->index_table, sizeof(value_t) * domain::extent, cudaMemcpyDefault, stream));
	}
	//FIXME: passing kjey_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t insert(key_t key) noexcept {
		value_t tag = atomicCAS(&this->index(key), sentinel_v, 0);
		if(tag == sentinel_v) {
			value_t idx			   = atomicAdd(this->Instance<block_partition_>::count, 1);
			this->index(key)	   = idx;
			this->active_keys[idx] = key;///< created a record
			return idx;
		}
		return -1;
	}
	//FIXME: passing kjey_t here might cause problems because cuda is buggy
	__forceinline__ __device__ value_t query(key_t key) const noexcept {
		return this->index(key);
	}
	__forceinline__ __device__ void reinsert(value_t index) {
		this->index(this->active_keys[index]) = index;
	}
};
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}// namespace mn

#endif