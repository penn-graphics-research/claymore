#ifndef __HASH_TABLE_CUH_
#define __HASH_TABLE_CUH_
#include "mgmpm_kernels.cuh"
#include "settings.h"
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
    h_count = other.h_count;
    checkCudaErrors(cudaMemcpyAsync(_haloMarks, other._haloMarks,
                                    sizeof(char) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(_overlapMarks, other._overlapMarks,
                                    sizeof(int) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(_haloBlocks, other._haloBlocks,
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
    // allocate_handle(allocator);
    _ppcs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                      config::g_blockvolume);
    _ppbs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
    _cellbuckets = (int *)allocator.allocate(
        sizeof(int) * maxBlockCnt * config::g_blockvolume * config::g_max_ppc);
    _blockbuckets = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                              config::g_particle_num_per_block);
    _binsts = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
    /// init
    reset();
  }
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t capacity) {
    halo_base_t::resizePartition(allocator, this->_capacity, capacity);
    allocator.deallocate(_ppcs,
                         sizeof(int) * this->_capacity * config::g_blockvolume);
    allocator.deallocate(_ppbs, sizeof(int) * this->_capacity);
    allocator.deallocate(_cellbuckets, sizeof(int) * this->_capacity *
                                           config::g_blockvolume *
                                           config::g_max_ppc);
    allocator.deallocate(_blockbuckets,
                         sizeof(int) * this->_capacity * config::g_blockvolume);
    allocator.deallocate(_binsts, sizeof(int) * this->_capacity);
    _ppcs = (int *)allocator.allocate(sizeof(int) * capacity *
                                      config::g_blockvolume);
    _ppbs = (int *)allocator.allocate(sizeof(int) * capacity);
    _cellbuckets = (int *)allocator.allocate(
        sizeof(int) * capacity * config::g_blockvolume * config::g_max_ppc);
    _blockbuckets = (int *)allocator.allocate(sizeof(int) * capacity *
                                              config::g_particle_num_per_block);
    _binsts = (int *)allocator.allocate(sizeof(int) * capacity);
    resize_table(allocator, capacity);
  }
  ~Partition() {
    // checkCudaErrors(cudaFree(_ppcs));
    // checkCudaErrors(cudaFree(_ppbs));
    // checkCudaErrors(cudaFree(_cellbuckets));
    // checkCudaErrors(cudaFree(_blockbuckets));
    // checkCudaErrors(cudaFree(_binsts));
  }
  void reset() {
    checkCudaErrors(cudaMemset(this->_cnt, 0, sizeof(value_t)));
    checkCudaErrors(
        cudaMemset(this->_indexTable, 0xff, sizeof(value_t) * domain::extent));
    checkCudaErrors(cudaMemset(
        this->_ppcs, 0, sizeof(int) * this->_capacity * config::g_blockvolume));
  }
  void resetTable(cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(this->_indexTable, 0xff,
                                    sizeof(value_t) * domain::extent, stream));
  }
  template <typename CudaContext>
  void buildParticleBuckets(CudaContext &&cuDev, value_t cnt) {
    checkCudaErrors(cudaMemsetAsync(this->_ppbs, 0, sizeof(int) * (cnt + 1),
                                    cuDev.stream_compute()));
    cuDev.compute_launch({cnt, config::g_blockvolume}, cell_bucket_to_block,
                         _ppcs, _cellbuckets, _ppbs, _blockbuckets);
  }
  void copy_to(Partition &other, std::size_t blockCnt, cudaStream_t stream) {
    halo_base_t::copy_to(other, blockCnt, stream);
    checkCudaErrors(cudaMemcpyAsync(other._indexTable, this->_indexTable,
                                    sizeof(value_t) * domain::extent,
                                    cudaMemcpyDefault, stream));
    checkCudaErrors(cudaMemcpyAsync(other._ppbs, this->_ppbs,
                                    sizeof(int) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(other._binsts, this->_binsts,
                                    sizeof(int) * blockCnt, cudaMemcpyDefault,
                                    stream));
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
  __forceinline__ __device__ void add_advection(key_t cellid, int dirtag,
                                                int pidib) noexcept {
    using namespace config;
    key_t blockid = cellid / g_blocksize;
    value_t blockno = query(blockid);
#if 1
    if (blockno == -1) {
      ivec3 offset{};
      dir_components(dirtag, offset);
      printf("The hell with this! loc(%d, %d, %d) dir(%d, %d, %d) pidib(%d)\n",
             cellid[0], cellid[1], cellid[2], offset[0], offset[1], offset[2],
             pidib);
      return;
    }
#endif
    value_t cellno = ((cellid[0] & g_blockmask) << (g_blockbits << 1)) |
                     ((cellid[1] & g_blockmask) << g_blockbits) |
                     (cellid[2] & g_blockmask);
    int pidic = atomicAdd(_ppcs + blockno * g_blockvolume + cellno, 1);
    _cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                 pidic] = (dirtag * g_particle_num_per_block) | pidib;
  }

  int *_ppcs, *_ppbs;
  int *_cellbuckets, *_blockbuckets;
  int *_binsts;
};

} // namespace mn

#endif