#ifndef QR_CUH
#define QR_CUH
#include "Givens.cuh"

namespace mn {

namespace math {

	/**
	 \brief 2x2 polar decomposition.
	 \param[in] A matrix.
	 \param[out] R Robustly a rotation matrix in givens form
	 \param[out] S_Sym Symmetric. Whole matrix is stored

	 Whole matrix S is stored since its faster to calculate due to simd vectorization
	 Polar guarantees negative sign is on the small magnitude singular value.
	 S is guaranteed to be the closest one to identity.
	 R is guaranteed to be the closest rotation to A.
	 */
	template<typename T>
	__forceinline__ __host__ __device__ void polar_decomposition(const std::array<T, 4>& a, GivensRotation<T>& r, std::array<T, 4>& s) {
		double x[2]		   = {a[0] + a[3], a[1] - a[2]};
		double denominator = sqrt(x[0] * x[0] + x[1] * x[1]);
		r.c				   = (T) 1;
		r.s				   = (T) 0;
		if(denominator != 0) {
			/*
          No need to use a tolerance here because x(0) and x(1) always have
          smaller magnitude then denominator, therefore overflow never happens.
        */
			r.c = x[0] / denominator;
			r.s = -x[1] / denominator;
		}
		for(int i = 0; i < 4; ++i) {
			s[i] = a[i];
		}
		r.template mat_rotation<2, T>(s);
	}

	/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
	template<typename T>
	__forceinline__ __host__ __device__ void polar_decomposition(const std::array<T, 4>& a, const std::array<T, 4>& r, const std::array<T, 4>& s) {
		GivensRotation<T> rotation(0, 1);
		polar_decomposition(a, rotation, s);
		rotation.fill<2>(R);
	}

}// namespace math

}// namespace mn

#endif