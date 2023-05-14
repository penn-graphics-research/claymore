#ifndef EIGEN_DECOMPOSITION_CUH
#define EIGEN_DECOMPOSITION_CUH

#include <device_types.h>

#include "MatrixUtils.h"

namespace mn {

namespace math {

	template<typename T>
	__forceinline__ __device__ void fast_eigenvalues(const std::array<T, 9>& mat, std::array<T, 3>& lambda) {
		// 24 mults, 20 adds, 1 atan2, 1 sincos, 2 sqrts
		double m	   = (1.0 / 3) * (mat[0] + mat[4] + mat[8]);
		double a00	   = mat[0] - m;
		double a11	   = mat[4] - m;
		double a22	   = mat[8] - m;
		double a12_sqr = mat[3] * mat[3];
		double a13_sqr = mat[6] * mat[6];
		double a23_sqr = mat[7] * mat[7];
		double p	   = (1.0 / 6) * (a00 * a00 + a11 * a11 + a22 * a22 + 2 * (a12_sqr + a13_sqr + a23_sqr));
		double q	   = 0.5 * (a00 * (a11 * a22 - a23_sqr) - a11 * a13_sqr - a22 * a12_sqr) + mat[3] * mat[6] * mat[7];
		double sqrt_p  = sqrt(p);
		double disc	   = p * p * p - q * q;
		double phi	   = (1.0 / 3) * atan2(sqrt(fmax(0.0, disc)), q);
		double c = cos(phi), s = sin(phi);
		double sqrt_p_cos			 = sqrt_p * c;
		double root_three_sqrt_p_sin = sqrt(3.0) * sqrt_p * s;
		lambda[0]					 = m + 2 * sqrt_p_cos;
		lambda[1]					 = m - sqrt_p_cos - root_three_sqrt_p_sin;
		lambda[2]					 = m - sqrt_p_cos + root_three_sqrt_p_sin;
		if(lambda[0] < lambda[1]) {
			swap(lambda + 0, lambda + 1);
		}
		if(lambda[1] < lambda[2]) {
			swap(lambda + 1, lambda + 2);
		}
		if(lambda[0] < lambda[1]) {
			swap(lambda + 0, lambda + 1);
		}
	}

	template<typename T>
	__forceinline__ __device__ void fast_eigenvectors(const std::array<T, 9>& mat, std::array<T, 3>& lambda, std::array<T, 9> vecs) {///< column major assumed
		// 71 mults, 44 adds, 3 divs, 3 sqrts
		// flip if necessary so that first eigenvalue is the most different
		bool flipped		  = false;
		double lambda_flip[3] = {lambda[0], lambda[1], lambda[2]};
		if(lambda[0] - lambda[1] < lambda[1] - lambda[2]) {// 2a
			swap(lambda_flip + 0, lambda_flip + 2);
			flipped = true;
		}

		// get first eigenvector
		std::array<double, 9> c1;
		std::array<double, 9> tmp;
		std::array<double, 9> dtmp;
		for(int i = 0; i < 9; ++i) {
			tmp[i] = mat[i] - (((i & 3) == 0) ? lambda_flip[0] : 0);
		}
		matrixCofactor3d(tmp, c1);
		// Eigen::Matrix3d::Index i;
		// T norm2 = c1.colwise().squaredNorm().maxCoeff(&i); // 3a + 12m+6a +
		// 9m+6a+1d+1s = 21m+15a+1d+1s
		int i;
		dtmp[0]		 = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2];
		dtmp[1]		 = c1[3] * c1[3] + c1[4] * c1[4] + c1[5] * c1[5];
		dtmp[2]		 = c1[6] * c1[6] + c1[7] * c1[7] + c1[8] * c1[8];
		double norm2 = dtmp[0];
		i			 = 0;
		if(dtmp[1] > norm2) {
			norm2 = dtmp[1];
			i	  = 1;
		}
		if(dtmp[2] > norm2) {
			norm2 = dtmp[2];
			i	  = 2;
		}

		// Eigen::Vector3d v1;
		std::array<double, 3> v1;
		if(sgn(norm2) > 0) {
			double one_over_sqrt = 1.0 / sqrt(norm2);
			for(int c = 0; c < 3; ++c) {
				v1[c] = c1[i * 3 + c] * one_over_sqrt;
			}
		} else {
			v1[0] = 1;
			v1[1] = 0;
			v1[2] = 0;
		}

		// form basis for orthogonal complement to v1, and reduce A to this space
		orthogonalVector(v1, tmp);
		dtmp[3] = tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2];
		dtmp[3] = 1.0 / sqrt(dtmp[3]);
		for(int i = 0; i < 3; ++i) {
			tmp[i] *= dtmp[3];
		}
		std::array<double, 3> v1_orthogonal = {tmp[0], tmp[1], tmp[2]};// 6m+2a+1d+1s (tweak: 5m+1a+1d+1s)
		// Eigen::Matrix<T, 3, 2> other_v;
		std::array<double, 6> other_v;///< 3x2
		other_v[0] = v1_orthogonal[0];
		other_v[1] = v1_orthogonal[1];
		other_v[2] = v1_orthogonal[2];
		// other_v.col(1) = v1.cross(v1_orthogonal); // 6m+3a (tweak: 4m+1a)
		cross(v1, v1_orthogonal, tmp);
		other_v[3] = tmp[0];
		other_v[4] = tmp[1];
		other_v[5] = tmp[2];

		// 2x3
		tmp[0] = other_v[0] * mat[0] + other_v[1] * mat[1] + other_v[2] * mat[2];
		tmp[1] = other_v[3] * mat[0] + other_v[4] * mat[1] + other_v[5] * mat[2];
		tmp[2] = other_v[0] * mat[3] + other_v[1] * mat[4] + other_v[2] * mat[5];
		tmp[3] = other_v[3] * mat[3] + other_v[4] * mat[4] + other_v[5] * mat[5];
		tmp[4] = other_v[0] * mat[6] + other_v[1] * mat[7] + other_v[2] * mat[8];
		tmp[5] = other_v[3] * mat[6] + other_v[4] * mat[7] + other_v[5] * mat[8];

		std::array<double, 4> A_reduced = {tmp[0] * other_v[0] + tmp[2] * other_v[1] + tmp[4] * other_v[2], tmp[1] * other_v[0] + tmp[3] * other_v[1] + tmp[5] * other_v[2], tmp[0] * other_v[3] + tmp[2] * other_v[4] + tmp[4] * other_v[5], tmp[1] * other_v[3] + tmp[3] * other_v[4] + tmp[5] * other_v[5]};
		// Eigen::Matrix2d A_reduced = other_v.transpose() * A_Sym * other_v; //
		// 21m+12a (tweak: 18m+9a)

		// find third eigenvector from A_reduced, and fill in second via cross product
		// Eigen::Matrix2d c3;
		std::array<double, 4> c3;
		// computeCofactorMtr<2>(A_reduced - lambda_flip(2) *
		// Eigen::Matrix2d::Identity(), c3);
		for(int i = 0; i < 4; ++i) {
			tmp[i] = A_reduced[i] - (((i % 3) == 0) ? lambda_flip[2] : 0);
		}
		matrixCofactor2d(tmp, c3);

		int j;
		// norm2 = c3.colwise().squaredNorm().maxCoeff(&j); // 3a + 12m+6a +
		// 9m+6a+1d+1s = 21m+15a+1d+1s
		dtmp[0] = c3[0] * c3[0] + c3[1] * c3[1];
		dtmp[1] = c3[2] * c3[2] + c3[3] * c3[3];
		norm2	= dtmp[0];
		j		= 0;
		if(dtmp[1] > norm2) {
			norm2 = dtmp[1];
			j	  = 1;
		}

		std::array<double, 3> v3;
		if(sgn(norm2) > 0) {
			double one_over_sqrt = (T) 1 / sqrt(norm2);
			v3[0]				 = (other_v[0] * c3[j * 2 + 0] + other_v[3] * c3[j * 2 + 1]) * one_over_sqrt;
			v3[1]				 = (other_v[1] * c3[j * 2 + 0] + other_v[4] * c3[j * 2 + 1]) * one_over_sqrt;
			v3[2]				 = (other_v[2] * c3[j * 2 + 0] + other_v[5] * c3[j * 2 + 1]) * one_over_sqrt;
		} else {
			v3[0] = other_v[0];
			v3[1] = other_v[1];
			v3[2] = other_v[2];
		}

		std::array<double, 3> v2;
		cross(v3, v1, v2);
		// v2 = v3.cross(v1); // 6m+3a
		// finish
		if(flipped) {
			vector_copy(1., v3, vecs + 0);
			vector_copy(1., v2, vecs + 3);
			vector_copy(-1., v1, vecs + 6);
		} else {
			vector_copy(1., v1, vecs + 0);
			vector_copy(1., v2, vecs + 3);
			vector_copy(1., v3, vecs + 6);
		}
	}

	template<typename T>
	__forceinline__ __device__ void make_pd_3d(const std::array<T, 9>& mat, T eps = 1e-6) {
		std::array<T, 3> vals;
		std::array<T, 9> vecs;
		std::array<T, 9> VT;
		std::array<T, 9> tmp;

		fast_eigenvalues(mat, vals);
		fast_eigenvectors(mat, vals, vecs);
		for(int i = 0; i < 3; ++i) {
			vals[i] = vals[i] > eps ? vals[i] : eps;///< clamp
		}
		matrix_transpose_3d(vecs, VT);
		matrix_diagonal_matrix_multiplication_3d(vecs, vals, tmp);
		matrix_matrix_multiplication_3d(tmp, VT, mat);
	}

	template<typename T>
	__forceinline__ __device__ void make_pd_2d(const std::array<T, 9>& mat, T eps = 1e-6) {
		// based on
		// http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
		const double a = mat[0];
		const double b = (mat[2] + mat[1]) / 2.0;
		const double d = mat[3];

		double b2				= b * b;
		const double d_big		= a * d - b2;
		const double t_div_2	= (a + d) / 2.0;
		const double sqrt_tt_4d = sqrt(t_div_2 * t_div_2 - d_big);
		const double l2			= t_div_2 - sqrt_tt_4d;
		if(l2 < 0.0) {
			const double l1 = t_div_2 + sqrt_tt_4d;
			if(l1 <= 0.0) {
				mat[0] = mat[1] = mat[2] = mat[3] = 0.0;
			} else {
				if(b2 == 0.0) {
					mat[0] = l1;
					mat[1] = mat[2] = mat[3] = 0.0f;
				} else {
					const double L1md		 = l1 - d;
					const double L1md_div_L1 = L1md / l1;
					mat[0]					 = L1md_div_L1 * L1md;
					mat[1] = mat[2] = b * L1md_div_L1;
					mat[3]			= b2 / l1;
				}
			}
		}
	}
}// namespace math

}// namespace mn

#endif