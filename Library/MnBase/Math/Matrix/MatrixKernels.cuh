#ifndef __MATRIX_KERNELS_CUH__
#define __MATRIX_KERNELS_CUH__

#include <device_types.h>
#include <MnBase/AggregatedAttribs.cuh>

namespace mn {

	template <typename T, int dim>
	__global__ std::enable_if_t<dim == 3> set_identity(int num, AttribPort<T, 9> mat) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= num) return;
		for (int c = 0; c < 9; ++c)
			mat[c][idx] = c & 0x3 ? 0 : 1;
	}

	template <typename T, int dim>
	__global__ std::enable_if_t<dim == 2> set_identity(int num, AttribPort<T, 4> mat) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= num) return;
		mat[0][idx] = 1;
		mat[1][idx] = 0;
		mat[2][idx] = 0;
		mat[3][idx] = 1;
	}

}

#endif