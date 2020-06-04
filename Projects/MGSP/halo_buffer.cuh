#ifndef __HALO_BUFFER_CUH_
#define __HALO_BUFFER_CUH_
#include "grid_buffer.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include <MnBase/Meta/Polymorphism.h>
//#include <cub/device/device_scan.cuh>

namespace mn {

using HaloGridBlocksDomain = compact_domain<int, config::g_max_halo_block>;
using halo_grid_blocks_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               HaloGridBlocksDomain, attrib_layout::soa, grid_block_>;

using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, f32_, f32_, f32_, f32_>;

/// halo grid block
struct HaloGridBlocks {
  struct HaloBuffer {
    Instance<halo_grid_blocks_> _grid;
    ivec3 *_blockids;
  };

  HaloGridBlocks(int numNeighbors)
      : numTargets{numNeighbors}, h_counts(numNeighbors, 0) {
    checkCudaErrors(cudaMalloc(&_counts, sizeof(uint32_t) * numTargets));
    _buffers.resize(numTargets);
  }
  template <typename Allocator>
  void initBlocks(Allocator allocator, uint32_t blockCount) {
    for (int did = 0; did < numTargets; ++did)
      _buffers[did]._blockids =
          (ivec3 *)allocator.allocate(sizeof(ivec3) * blockCount);
  }
  template <typename Allocator>
  void initBuffer(Allocator allocator, std::vector<uint32_t> counts) {
    for (int did = 0; did < numTargets; ++did)
      _buffers[did]._grid.allocate_handle(allocator, counts[did]);
  }
  void resetCounts(cudaStream_t stream) {
    checkCudaErrors(
        cudaMemsetAsync(_counts, 0, sizeof(uint32_t) * numTargets, stream));
  }
  void retrieveCounts(cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(h_counts.data(), _counts,
                                    sizeof(uint32_t) * numTargets,
                                    cudaMemcpyDefault, stream));
  }
  void send(HaloGridBlocks &other, int src, int dst, cudaStream_t stream) {
    auto cnt = other.h_counts[src] = h_counts[dst];
#if 0
    checkCudaErrors(cudaMemcpyAsync(
        &other._buffers[src].val(_1, 0), &_buffers[dst].val(_1, 0),
        sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
#else
    checkCudaErrors(
        cudaMemcpyAsync(other._buffers[src]._blockids, _buffers[dst]._blockids,
                        sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
#endif
#if 0
    checkCudaErrors(
        cudaMemcpyAsync(&other._buffers[src]._grid.ch(_0, 0).val_1d(_0, 0),
                        &_buffers[dst]._grid.ch(_0, 0).val_1d(_0, 0),
                        grid_block_::size * cnt, cudaMemcpyDefault, stream));
#else
    checkCudaErrors(
        cudaMemcpyPeerAsync(&other._buffers[src]._grid.ch(_0, 0).val_1d(_0, 0),
                            dst, &_buffers[dst]._grid.ch(_0, 0).val_1d(_0, 0),
                            src, grid_block_::size * cnt, stream));
#endif
    // printf("sending from %d to %d at %llu\n", src, dst,
    //       (unsigned long long)&other._buffers[src]._grid.ch(_0, 0).val_1d(_0,
    //       0));
  }
  const int numTargets;
  uint32_t *_counts;
  std::vector<uint32_t> h_counts;
  std::vector<HaloBuffer> _buffers;
};

} // namespace mn

#endif