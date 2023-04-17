#ifndef SORT_KERNELS_CUH
#define SORT_KERNELS_CUH
#include <stdint.h>

namespace mn {

template<typename ElemType, unsigned int Size, typename IndexType>
__global__ void gather_entry(int num, const AttribPort<ElemType, Size> from, AttribPort<ElemType, Size> to, const IndexType* prev) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	for(int i = 0; i < Size; ++i) {
		to[i][idx] = from[i][prev[idx]];
	}
}

template<typename ElemType, unsigned int Size, typename IndexType>
__global__ void scatter_entry(int num, const AttribPort<ElemType, Size> from, AttribPort<ElemType, Size> to, const IndexType* _next) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	for(int i = 0; i < Size; ++i) {
		to[i][_next[idx]] = from[i][idx];
	}
}

}// namespace mn

#endif