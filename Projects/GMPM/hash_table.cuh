#ifndef __HASH_TABLE_CUH_
#define __HASH_TABLE_CUH_
#include "settings.h"
#include <MnBase/Object/Structural.h>
#include <MnSystem/Cuda/HostUtils.hpp>

namespace mn {

template <int> struct HaloPartition {
  template <typename Allocator> HaloPartition(Allocator, int) {}
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t prevCapacity,
                       std::size_t capacity) {}
  void copy_to(HaloPartition &other, std::size_t blockCnt,
               cudaStream_t stream) {}
};
template <> struct HaloPartition<1> {
  template <typename Allocator>
  HaloPartition(Allocator allocator, int maxBlockCnt) {
    _count = (int *)allocator.allocate(sizeof(char) * maxBlockCnt);
    _haloMarks = (char *)allocator.allocate(sizeof(char) * maxBlockCnt);
    _overlapMarks = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
    _haloBlocks = (ivec3 *)allocator.allocate(sizeof(ivec3) * maxBlockCnt);
  }
  void copy_to(HaloPartition &other, std::size_t blockCnt,
               cudaStream_t stream) {
    other.h_count = h_count;
    checkCudaErrors(cudaMemcpyAsync(other._haloMarks, _haloMarks,
                                    sizeof(char) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(other._overlapMarks, _overlapMarks,
                                    sizeof(int) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(other._haloBlocks, _haloBlocks,
                                    sizeof(ivec3) * blockCnt, cudaMemcpyDefault,
                                    stream));
  }
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t prevCapacity,
                       std::size_t capacity) {
    allocator.deallocate(_haloMarks, sizeof(char) * prevCapacity);
    allocator.deallocate(_overlapMarks, sizeof(int) * prevCapacity);
    allocator.deallocate(_haloBlocks, sizeof(ivec3) * prevCapacity);
    _haloMarks = (char *)allocator.allocate(sizeof(char) * capacity);
    _overlapMarks = (int *)allocator.allocate(sizeof(int) * capacity);
    _haloBlocks = (ivec3 *)allocator.allocate(sizeof(ivec3) * capacity);
  }
  void resetHaloCount(cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(_count, 0, sizeof(int), stream));
  }
  void resetOverlapMarks(uint32_t neighborBlockCount, cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(_overlapMarks, 0,
                                    sizeof(int) * neighborBlockCount, stream));
  }
  void retrieveHaloCount(cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(&h_count, _count, sizeof(int),
                                    cudaMemcpyDefault, stream));
  }
  int *_count, h_count;
  char *_haloMarks; ///< halo particle blocks
  int *_overlapMarks;
  ivec3 *_haloBlocks;
};

using block_partition_ =
    structural<structural_type::hash,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridDomain, attrib_layout::aos, empty_>;

template <int Opt = 1>
struct Partition : Instance<block_partition_>, HaloPartition<Opt> {
  using base_t = Instance<block_partition_>;
  using halo_base_t = HaloPartition<Opt>;
  using block_partition_::key_t;
  using block_partition_::value_t;
  static_assert(sentinel_v == (value_t)(-1), "sentinel value not full 1s\n");

  template <typename Allocator>
  Partition(Allocator allocator, int maxBlockCnt)
      : halo_base_t{allocator, maxBlockCnt} {
    allocate_table(allocator, maxBlockCnt);
    /// init
    reset();
  }
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t capacity) {
    halo_base_t::resizePartition(allocator, this->_capacity, capacity);
    resize_table(allocator, capacity);
  }
  ~Partition() {}
  void reset() {
    checkCudaErrors(cudaMemset(this->_cnt, 0, sizeof(value_t)));
    checkCudaErrors(
        cudaMemset(this->_indexTable, 0xff, sizeof(value_t) * domain::extent));
  }
  void resetTable(cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(this->_indexTable, 0xff,
                                    sizeof(value_t) * domain::extent, stream));
  }
  void copy_to(Partition &other, std::size_t blockCnt, cudaStream_t stream) {
    halo_base_t::copy_to(other, blockCnt, stream);
    checkCudaErrors(cudaMemcpyAsync(other._indexTable, this->_indexTable,
                                    sizeof(value_t) * domain::extent,
                                    cudaMemcpyDefault, stream));
  }
  __forceinline__ __device__ value_t insert(key_t key) noexcept {
    value_t tag = atomicCAS(&this->index(key), sentinel_v, 0);
    if (tag == sentinel_v) {
      value_t idx = atomicAdd(this->_cnt, 1);
      this->index(key) = idx;
      this->_activeKeys[idx] = key; ///< created a record
      return idx;
    }
    return -1;
  }
  __forceinline__ __device__ value_t query(key_t key) const noexcept {
    return this->index(key);
  }
  __forceinline__ __device__ void reinsert(value_t index) {
    this->index(this->_activeKeys[index]) = index;
  }
};

} // namespace mn

#endif