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

	int* cell_particle_counts;
	int* particle_bucket_sizes;
	int* cellbuckets;
	int* blockbuckets;
	int* bin_offsets;

	template<typename Allocator>
	Partition(Allocator allocator, int max_block_count)
		: halo_base_t {allocator, max_block_count} {
		allocate_table(allocator, max_block_count);
		cell_particle_counts  = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count * config::G_BLOCKVOLUME));
		particle_bucket_sizes = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count));
		cellbuckets			  = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count * config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL));
		blockbuckets		  = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count * config::G_PARTICLE_NUM_PER_BLOCK));
		bin_offsets			  = static_cast<int*>(allocator.allocate(sizeof(int) * max_block_count));

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

		allocator.deallocate(cell_particle_counts, sizeof(int) * this->capacity * config::G_BLOCKVOLUME);
		allocator.deallocate(particle_bucket_sizes, sizeof(int) * this->capacity);
		allocator.deallocate(cellbuckets, sizeof(int) * this->capacity * config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL);
		allocator.deallocate(blockbuckets, sizeof(int) * this->capacity * config::G_BLOCKVOLUME);
		allocator.deallocate(bin_offsets, sizeof(int) * this->capacity);

		cell_particle_counts  = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_BLOCKVOLUME));
		particle_bucket_sizes = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));
		cellbuckets			  = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_BLOCKVOLUME * config::G_MAX_PARTICLES_IN_CELL));
		blockbuckets		  = static_cast<int*>(allocator.allocate(sizeof(int) * capacity * config::G_PARTICLE_NUM_PER_BLOCK));
		bin_offsets			  = static_cast<int*>(allocator.allocate(sizeof(int) * capacity));

		resize_table(allocator, capacity);
	}

	void reset() {
		check_cuda_errors(cudaMemset(this->Instance<block_partition_>::count, 0, sizeof(value_t)));
		check_cuda_errors(cudaMemset(this->index_table, 0xff, sizeof(value_t) * domain::extent));
		check_cuda_errors(cudaMemset(this->cell_particle_counts, 0, sizeof(int) * this->capacity * config::G_BLOCKVOLUME));
	}
	void reset_table(cudaStream_t stream) {
		check_cuda_errors(cudaMemsetAsync(this->index_table, 0xff, sizeof(value_t) * domain::extent, stream));
	}
	template<typename CudaContext>
	void build_particle_buckets(CudaContext&& cu_dev, value_t count) {
		check_cuda_errors(cudaMemsetAsync(this->particle_bucket_sizes, 0, sizeof(int) * (count + 1), cu_dev.stream_compute()));
		cu_dev.compute_launch({count, config::G_BLOCKVOLUME}, cell_bucket_to_block, cell_particle_counts, cellbuckets, particle_bucket_sizes, blockbuckets);
	}
	void copy_to(Partition& other, std::size_t block_count, cudaStream_t stream) {
		halo_base_t::copy_to(other, block_count, stream);
		check_cuda_errors(cudaMemcpyAsync(other.index_table, this->index_table, sizeof(value_t) * domain::extent, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.particle_bucket_sizes, this->particle_bucket_sizes, sizeof(int) * block_count, cudaMemcpyDefault, stream));
		check_cuda_errors(cudaMemcpyAsync(other.bin_offsets, this->bin_offsets, sizeof(int) * block_count, cudaMemcpyDefault, stream));
	}
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
	__forceinline__ __device__ value_t query(key_t key) const noexcept {
		return this->index(key);
	}
	__forceinline__ __device__ void reinsert(value_t index) {
		this->index(this->active_keys[index]) = index;
	}
	__forceinline__ __device__ void add_advection(key_t cellid, int dirtag, int particle_id_in_block) noexcept {
		key_t blockid	= cellid / config::G_BLOCKSIZE;
		value_t blockno = query(blockid);
#if PRINT_NEGATIVE_BLOGNOS
		if(blockno == -1) {
			ivec3 offset {};
			dir_components(dirtag, offset);
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg, readability-magic-numbers) Cuda has no other way to print; Numbers are array indices to be printed
			printf("The hell with this! loc(%d, %d, %d) dir(%d, %d, %d) particle_id_in_block(%d)\n", cellid[0], cellid[1], cellid[2], offset[0], offset[1], offset[2], particle_id_in_block);
			return;
		}
#endif
		//NOLINTNEXTLINE(readability-magic-numbers) Numbers are array indices to be printed
		value_t cellno																											 = ((cellid[0] & config::G_BLOCKMASK) << (config::G_BLOCKBITS << 1)) | ((cellid[1] & config::G_BLOCKMASK) << config::G_BLOCKBITS) | (cellid[2] & config::G_BLOCKMASK);
		int particle_id_in_cell																									 = atomicAdd(cell_particle_counts + static_cast<ptrdiff_t>(blockno) * config::G_BLOCKVOLUME + cellno, 1);
		cellbuckets[blockno * config::G_PARTICLE_NUM_PER_BLOCK + cellno * config::G_MAX_PARTICLES_IN_CELL + particle_id_in_cell] = (dirtag * config::G_PARTICLE_NUM_PER_BLOCK) | particle_id_in_block;
	}
};
//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}// namespace mn

#endif