#ifndef __HALO_KERNELS_CUH_
#define __HALO_KERNELS_CUH_

#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnSystem/Cuda/DeviceUtils.cuh>

namespace mn {

using namespace config;
using namespace placeholder;

template <typename Partition, typename HaloGridBlocks>
__global__ void
mark_overlapping_blocks(uint32_t blockCount, int otherdid,
                        const ivec3 *__restrict__ incomingBlockIds,
                        Partition partition, uint32_t *count,
                        HaloGridBlocks haloGridBlocks) {
  uint32_t inc_blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (inc_blockno >= blockCount)
    return;
  auto inc_blockid = incomingBlockIds[inc_blockno];
  auto blockno = partition.query(inc_blockid);
  if (blockno >= 0) {
    atomicOr(partition._overlapMarks + blockno, 1 << otherdid);
    auto halono = atomicAdd(count, 1);
    // haloGridBlocks.val(_1, halono) = inc_blockid;
    haloGridBlocks._blockids[halono] = inc_blockid;
  }
}

template <typename Partition>
__global__ void collect_blockids_for_halo_reduction(uint32_t particleBlockCount,
                                                    int did,
                                                    Partition partition) {
  std::size_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= particleBlockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  partition._haloMarks[blockno] = 0;
  for (char i = 0; i < 2; ++i)
    for (char j = 0; j < 2; ++j)
      for (char k = 0; k < 2; ++k) {
        ivec3 neighborid{blockid[0] + i, blockid[1] + j, blockid[2] + k};
        int neighborno = partition.query(neighborid);
        // if (partition._overlapMarks[neighborno] ^ ((HaloIndex)1 << did)) {
        if (partition._overlapMarks[neighborno]) {
          partition._haloMarks[blockno] = 1;
          auto halono = atomicAdd(partition._count, 1);
          partition._haloBlocks[halono] = blockid;
          return;
        }
      }
}

template <typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void collect_grid_blocks(Grid grid, Partition partition,
                                    HaloGridBlocks haloGridBlocks) {
  uint32_t halo_blockno = blockIdx.x;
  // auto halo_blockid = haloGridBlocks._grid.val(_1, halo_blockno);
  auto halo_blockid = haloGridBlocks._blockids[halo_blockno];

  auto blockno = partition.query(halo_blockid);
  auto halo_gridblock = haloGridBlocks._grid.ch(_0, halo_blockno);
  auto gridblock = grid.ch(_0, blockno);

  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    halo_gridblock.val_1d(_0, cidib) = gridblock.val_1d(_0, cidib);
    halo_gridblock.val_1d(_1, cidib) = gridblock.val_1d(_1, cidib);
    halo_gridblock.val_1d(_2, cidib) = gridblock.val_1d(_2, cidib);
    halo_gridblock.val_1d(_3, cidib) = gridblock.val_1d(_3, cidib);
  }
}

template <typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void reduce_grid_blocks(Grid grid, Partition partition,
                                   HaloGridBlocks haloGridBlocks) {
  uint32_t halo_blockno = blockIdx.x;
  // auto halo_blockid = haloGridBlocks._grid.val(_1, halo_blockno);
  auto halo_blockid = haloGridBlocks._blockids[halo_blockno];
  auto blockno = partition.query(halo_blockid);
  auto halo_gridblock = haloGridBlocks._grid.ch(_0, halo_blockno);
  auto gridblock = grid.ch(_0, blockno);

  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    atomicAdd(&gridblock.val_1d(_0, cidib), halo_gridblock.val_1d(_0, cidib));
    atomicAdd(&gridblock.val_1d(_1, cidib), halo_gridblock.val_1d(_1, cidib));
    atomicAdd(&gridblock.val_1d(_2, cidib), halo_gridblock.val_1d(_2, cidib));
    atomicAdd(&gridblock.val_1d(_3, cidib), halo_gridblock.val_1d(_3, cidib));
  }
}

template <typename Domain, typename Partition, typename HaloParticleBlocks>
__global__ void
mark_migration_grid_blocks(uint32_t blockCount, Domain const domain,
                           Partition const partition, uint32_t *count,
                           HaloParticleBlocks haloParticleBlocks,
                           int const *activeGridBlockMarks) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  if (activeGridBlockMarks[blockno]) {
    auto blockid = partition._activeKeys[blockno];
    if (domain.within(blockid, ivec3{0, 0, 0}, ivec3{1, 1, 1})) {
      // haloParticleBlocks._binpbs[halono] = 0;
      auto halono = atomicAdd(count, 1);
      haloParticleBlocks._gblockids[halono] = blockid;
    }
  }
}

template <typename Grid, typename Partition, typename HaloGridBlocks>
__global__ void collect_migration_grid_blocks(Grid grid, Partition partition,
                                              HaloGridBlocks haloGridBlocks) {
  uint32_t halo_blockno = blockIdx.x;
  auto halo_blockid = haloGridBlocks._gblockids[halo_blockno];
  auto halo_gridblock = haloGridBlocks._grid.ch(_0, halo_blockno);

  auto blockno = partition.query(halo_blockid);
  auto gridblock = grid.ch(_0, blockno);

  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    halo_gridblock.val_1d(_0, cidib) = gridblock.val_1d(_0, cidib);
    halo_gridblock.val_1d(_1, cidib) = gridblock.val_1d(_1, cidib);
    halo_gridblock.val_1d(_2, cidib) = gridblock.val_1d(_2, cidib);
    halo_gridblock.val_1d(_3, cidib) = gridblock.val_1d(_3, cidib);
  }
}

} // namespace mn

#endif