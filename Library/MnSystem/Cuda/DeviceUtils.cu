#include "DeviceUtils.cuh"

namespace mn {

template<int NumPageBits>
__device__ int retrieve_block_local_offset(int level, uint64_t block_offset) {///< the level number starts from 0
	return (block_offset >> (NumPageBits + level * 3)) & 7;
}

}// namespace mn
