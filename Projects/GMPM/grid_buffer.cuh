#ifndef __GRID_BUFFER_CUH_
#define __GRID_BUFFER_CUH_
#include "mgmpm_kernels.cuh"
#include "settings.h"
#include <MnSystem/Cuda/HostUtils.hpp>

namespace mn {

using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, f32_, f32_, f32_, f32_>;
using grid_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridDomain, attrib_layout::aos, grid_block_>;
using grid_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridBufferDomain, attrib_layout::aos, grid_block_>;

struct GridBuffer : Instance<grid_buffer_> {
  using base_t = Instance<grid_buffer_>;

  template <typename Allocator>
  GridBuffer(Allocator allocator)
      : base_t{spawn<grid_buffer_, orphan_signature>(allocator)} {}
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > _capacity)
      resize(capacity, capacity);
  }
  template <typename CudaContext> void reset(int blockCnt, CudaContext &cuDev) {
    using namespace placeholder;
#if 0
    checkCudaErrors(cudaMemsetAsync((void *)&this->val_1d(_0, 0), 0,
                                    grid_block_::size * blockCnt, cuDev.stream_compute()));
#else
    cuDev.compute_launch({blockCnt, config::g_blockvolume}, clear_grid, *this);
#endif
  }
};

} // namespace mn

#endif