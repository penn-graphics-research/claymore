#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include <MnSystem/Cuda/HostUtils.hpp>

#include "mgmpm_kernels.cuh"
#include "settings.h"

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define PRINT_NEGATIVE_BLOGNOS 1

namespace mn {

template<int>
struct HaloPartition {
	template<typename Allocator>
	HaloPartition(Allocator allocator, int max_block_cnt) {
		(void) allocator;
		(void) max_block_cnt;
	}

	template<typename Allocator>
	void resize_partition(Allocator allocator, std::size_t prev_capacity, std::size_t capacity) {}
	void copy_to(HaloPartition& other, std::size_t block_cnt, cudaStream_t stream) {}
};
template<>
struct HaloPartition<1> {
	int* count;
	int h_count;
	char* halo_marks;///< halo particle blocks
	int* overlap_marks;
	ivec3* halo_blocks;

	template<typename Allocator>
	HaloPartition(Allocator allocator, int max_block_cnt)
		: h_count(0) {
		count		  = static_cast<int*>(allocator.allocate(sizeof(char) * max_block_cnt));
		halo_marks	  = static_cast<char*>(allocator.allocate(sizeof(char) * max_block_cnt));
		overlap_marks = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt));
		halo_blocks	  = static_cast<ivec3*>(allocator.allocate(sizeof(ivec3) * max_block_cnt));
	}

	void copy_to(HaloPartition& other, std::size_t block_cnt, cudaStream_t stream) const {
		other.h_count = h_count;
		check_cuda_errors(cudaMemcpyAsync(other.halo_marks, halo_marks, sizeof(char) * block_cnt, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.overlap_marks, overlap_marks, sizeof(int) * block_cnt, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.halo_blocks, halo_blocks, sizeof(ivec3) * block_cnt, cudaMemcpyDefault, stream));
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
		check_cuda_errors(cudaMemsetAsync(count, 0, sizeof(int), stream));
	}

	void reset_overlap_marks(uint32_t neighbor_block_count, cudaStream_t stream) const {
		check_cuda_errors(cudaMemsetAsync(overlap_marks, 0, sizeof(int) * neighbor_block_count, stream));
	}

	void retrieve_halo_count(cudaStream_t stream) {
		check_cuda_errors(cudaMemcpyAsync(&h_count, count, sizeof(int), cudaMemcpyDefault, stream));
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

	int* ppcs;
	int* ppbs;
	int* cellbuckets;
	int* blockbuckets;
	int* binsts;

	template<typename Allocator>
	Partition(Allocator allocator, int max_block_cnt)
		: halo_base_t {allocator, max_block_cnt} {
		allocate_table(allocator, max_block_cnt);
		ppcs		 = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt * config::G_BLOCKVOLUME));
		ppbs		 = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt));
		cellbuckets	 = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt * config::G_BLOCKVOLUME * config::G_MAX_PPC));
		blockbuckets = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt * config::G_PARTICLE_NUM_PER_BLOCK));
		binsts		 = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_cnt));

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

		allocator.deallocate(ppcs, sizeof(int) * this->capacity * config::G_BLOCKVOLUME);
		allocator.deallocate(ppbs, sizeof(int) * this->capacity);
		allocator.deallocate(cellbuckets, sizeof(int) * this->capacity * config::G_BLOCKVOLUME * config::G_MAX_PPC);
		allocator.deallocate(blockbuckets, sizeof(int) * this->capacity * config::G_BLOCKVOLUME);
		allocator.deallocate(binsts, sizeof(int) * this->capacity);

		ppcs		 = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_BLOCKVOLUME));
		ppbs		 = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));
		cellbuckets	 = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_BLOCKVOLUME * config::G_MAX_PPC));
		blockbuckets = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_PARTICLE_NUM_PER_BLOCK));
		binsts		 = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));

		resize_table(allocator, capacity);
	}

	void reset() {
		check_cuda_errors(cudaMemset(this->cnt, 0, sizeof(value_t)));
		check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
		check_cuda_errors(cudaMemset(this->ppcs, 0, sizeof(int) * this->capacity * config::G_BLOCKVOLUME));
	}
	void reset_table(cudaStream_t stream) {
		check_cuda_errors(cudaMemsetAsync(this->index_table, 0xff, sizeof(value_t) * domain::extent, stream));
	}
	template<typename CudaContext>
	void build_particle_buckets(CudaContext&& cu_dev, value_t cnt) {
		check_cuda_errors(cudaMemsetAsync(this->ppbs, 0, sizeof(int) * (cnt + 1), cu_dev.stream_compute()));
		cu_dev.compute_launch({cnt, config::G_BLOCKVOLUME}, cell_bucket_to_block, ppcs, cellbuckets, ppbs, blockbuckets);
	}
	void copy_to(Partition& other, std::size_t block_cnt, cudaStream_t stream) {
		halo_base_t::copy_to(other, block_cnt, stream);
		check_cuda_errors(cudaMemcpyAsync(other.index_table, this->index_table, sizeof(value_t) * domain::extent, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.ppbs, this->ppbs, sizeof(int) * block_cnt, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.binsts, this->binsts, sizeof(int) * block_cnt, cudaMemcpyDefault, stream));
	}
	__forceinline__ __device__ value_t insert(key_t key) noexcept {
		value_t tag = atomicCAS(&this->index(key), sentinel_v, 0);
		if(tag == sentinel_v) {
			value_t idx			   = atomicAdd(this->cnt, 1);
			this->index(key)	   = idx;
			this->active_keys[idx] = key;///< created a record
			return idx;
		}
		return -1;
	}
	__forceinline__ __device__ value_t query(key_t key) const noexcept {
		return this->index(key);
	}
	__forceinline__ __device__ void reinsert(value_t index) {
		this->index(this->active_keys[index]) = index;
	}
	__forceinline__ __device__ void add_advection(key_t cellid, int dirtag, int pidib) noexcept {
		key_t blockid	= cellid / config::G_BLOCKSIZE;
		value_t blockno = query(blockid);
#if PRINT_NEGATIVE_BLOGNOS
		if(blockno == -1) {
			ivec3 offset {};
			dir_components(dirtag, offset);
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg, readability-magic-numbers) Cuda has no other way to print; Numbers are array indices to be printed
			printf("The hell with this! loc(%d, %d, %d) dir(%d, %d, %d) pidib(%d)\n", cellid[0], cellid[1], cellid[2], offset[0], offset[1], offset[2], pidib);
			return;
		}
#endif
		//NOLINTNEXTLINE(readability-magic-numbers) Numbers are array indices to be printed
		value_t cellno																				 = ((cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (cellid[2] & config::G_BLOCKMASK);
		int pidic																					 = atomicAdd(ppcs + static_cast<ptrdiff_t>(blockno) * config::G_BLOCKVOLUME + cellno, 1);
		cellbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PPC + pidic] = (dirtag * config::G_PARTICLE_NUM_PER_BLOCK) | pidib;
	}
};
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}// namespace mn

#endif