/**
Givens rotation
*/
#ifndef GIVENS_CUH
#define GIVENS_CUH
#include <type_traits>
#include <cmath>

#define ENABLE_COLUMN_ROTATION 0
#define ENABLE_ADDITIONAL_GIVENS_FUNCTION 0

namespace mn {

//Stable calculation from https://en.wikipedia.org/wiki/Givens_rotation
namespace math {
	/**
			Class for givens rotation.
			Row rotation G*A corresponds to something like
			c -s  0
			( s  c  0 ) A
			0  0  1
			Column rotation A G' corresponds to something like
			c -s  0
			A ( s  c  0 )
			0  0  1

			c and s are always computed so that
			( c -s ) ( a )  =  ( * )
			s  c     b       ( 0 )

			Assume rowi<rowk.
			*/
	template<typename T>
	struct GivensRotation {
	   public:
		int rowi;
		int rowk;
		T c;
		T s;
		T r;

		__forceinline__ __host__ __device__ GivensRotation(int rowi_in, int rowk_in)
			: rowi(rowi_in)
			, rowk(rowk_in)
			, c(1)
			, s(0) {}

		__forceinline__ __host__ __device__ GivensRotation(T a, T b, int rowi_in, int rowk_in)
			: rowi(rowi_in)
			, rowk(rowk_in) {
			compute(a, b);
		}

		__forceinline__ __host__ __device__ ~GivensRotation() {}

		__forceinline__ __host__ __device__ void set_identity() {
			c = 1;
			s = 0;
		}

		__forceinline__ __host__ __device__ void transpose_in_place() {
			s = -s;
		}

		/**
				Compute c and s from a and b so that
				( c -s ) ( a )  =  ( * )
				s  c     b       ( 0 )
				*/
		template<typename TT>
		__forceinline__ __host__ __device__ std::enable_if_t<std::disjunction<std::is_same<TT, float>, std::is_same<TT, double>>::value, void> compute(const TT a, const TT b) {
			if(b == 0.0){
				if(a == 0.0){
					c = static_cast<TT>(1.0);
				}else{
					c = std::copysign(static_cast<TT>(1.0), a);
				}
				s = 0.0;
				r = std::abs(a);
			}else if(a == 0.0){
				c = 0.0;
				s = std::copysign(static_cast<TT>(1.0), b);
				r = std::abs(b);
			}else if (std::abs(a) > std::abs(b)){
				const TT t = b / a;
				const TT u = std::copysign(sqrt(static_cast<TT>(1.0) + t * t), a);
				c = static_cast<TT>(1.0) / u;
				s = -c * t;
				r = a * u;
			}else{
				const TT t = a / b;
				const TT u = std::copysign(sqrt(static_cast<TT>(1.0) + t * t), b);
				s = -static_cast<TT>(1.0) / u;
				c = t / u;
				r = b * u;
			}
		}

		/**
			  Fill the R with the entries of this rotation
				*/
		template<int Dim, typename T1>
		__forceinline__ __host__ __device__ void fill(std::array<T1, Dim * Dim>& a) const {
			for(int i = 0; i < Dim * Dim; ++i) {
				a[i] = 0;
			}
			for(int i = 0; i < Dim * Dim; i += Dim + 1) {
				a[i] = 1;
			}
			a[rowi + rowi * Dim] = c;
			a[rowk + rowi * Dim] = s;
			a[rowi + rowk * Dim] = -s;
			a[rowk + rowk * Dim] = c;
		}

		/**
				This function does something like Q^T A -> A
				[ c -s  0 ]
				[ s  c  0 ] A -> A
				[ 0  0  1 ]
				It only affects row i and row k of A.
				*/
		template<int Dim, typename T1>
		__forceinline__ __host__ __device__ void mat_rotation(std::array<T1, Dim * Dim>& a) const {
			for(int d = 0; d < Dim; d++) {
				T1 tau1			  = a[rowi + d * Dim];
				T1 tau2			  = a[rowk + d * Dim];
				a[rowi + d * Dim] = c * tau1 - s * tau2;
				a[rowk + d * Dim] = s * tau1 + c * tau2;
			}
		}

		template<int Dim, typename T1>
		__forceinline__ void vec_rotation(std::array<T1, Dim>& a) const {
			T1 tau1 = a[rowi];
			T1 tau2 = a[rowk];
			a[rowi] = c * tau1 - s * tau2;
			a[rowk] = s * tau1 + c * tau2;
		}

#if ENABLE_COLUMN_ROTATION
		/**
				This function does something like A Q -> A
				   [ c  s  0 ]
				A  [-s  c  0 ]  -> A
				   [ 0  0  1 ]
				It only affects column i and column k of A.
				*/
		template<class MatrixType>
		ZIRAN_FORCE_INLINE void column_rotation(MatrixType& a) const {
			for(int j = 0; j < a.rows(); j++) {
				T tau1	   = a(j, rowi);
				T tau2	   = a(j, rowk);
				a(j, rowi) = c * tau1 - s * tau2;
				a(j, rowk) = s * tau1 + c * tau2;
			}
			//not type safe :/
		}
#endif

		/**
			  Multiply givens must be for same row and column
			  **/
		__forceinline__ __host__ __device__ void operator*=(const GivensRotation<T>& a) {
			T new_c = c * a.c - s * a.s;
			T new_s = s * a.c + c * a.s;
			c		= new_c;
			s		= new_s;
		}

		/**
			  Multiply givens must be for same row and column
			  **/
		__forceinline__ __host__ __device__ GivensRotation<T> operator*(const GivensRotation<T>& a) const {
			GivensRotation<T> r(*this);
			r *= a;
			return r;
		}
	};

#if ENABLE_ADDITIONAL_GIVENS_FUNCTION
	/**
			\brief zero chasing the 3X3 matrix to bidiagonal form
			original form of H:
			x x 0
			x x x
			0 0 x
			after zero chase:
			x x 0
			0 x x
			0 0 x
			*/
	template<class T>
	inline ZIRAN_FORCE_INLINE void zero_chase(Matrix<T, 3, 3>& h, Matrix<T, 3, 3>& u, Matrix<T, 3, 3>& v) {
		/**
				Reduce H to of form
				x x +
				0 x x
				0 0 x
				*/
		GivensRotation<T> r1(h(0, 0), h(1, 0), 0, 1);
		/**
				Reduce H to of form
				x x 0
				0 x x
				0 + x
				Can calculate r2 without multiplying by r1 since both entries are in first two
				rows thus no need to divide by sqrt(a^2+b^2)
				*/
		GivensRotation<T> r2(1, 2);
		if(h(1, 0) != 0) {
			r2.compute(h(0, 0) * h(0, 1) + h(1, 0) * h(1, 1), h(0, 0) * h(0, 2) + h(1, 0) * h(1, 2));
		} else {
			r2.compute(h(0, 1), h(0, 2));
		}

		r1.row_rotation(h);

		/* GivensRotation<T> r2(h(0, 1), h(0, 2), 1, 2); */
		r2.column_rotation(h);
		r2.column_rotation(v);

		/**
				Reduce H to of form
				x x 0
				0 x x
				0 0 x
				*/
		GivensRotation<T> r3(h(1, 1), h(2, 1), 1, 2);
		r3.row_rotation(h);

		// Save this till end for better cache coherency
		// r1.row_rotation(u_transpose);
		// r3.row_rotation(u_transpose);
		r1.column_rotation(u);
		r3.column_rotation(u);
	}

	/**
			 \brief make a 3X3 matrix to upper bidiagonal form
			 original form of H:   x x x
								   x x x
								   x x x
			 after zero chase:
								   x x 0
								   0 x x
								   0 0 x
		  */
	template<class T>
	inline ZIRAN_FORCE_INLINE void make_upper_bidiag(Matrix<T, 3, 3>& h, Matrix<T, 3, 3>& u, Matrix<T, 3, 3>& v) {
		u = Matrix<T, 3, 3>::Identity();
		v = Matrix<T, 3, 3>::Identity();

		/**
			  Reduce H to of form
								  x x x
								  x x x
								  0 x x
			*/

		GivensRotation<T> r(h(1, 0), h(2, 0), 1, 2);
		r.row_rotation(h);
		// r.row_rotation(u_transpose);
		r.column_rotation(u);
		// zero_chase(h, u_transpose, v);
		zero_chase(h, u, v);
	}

	/**
			 \brief make a 3X3 matrix to lambda shape
			 original form of H:   x x x
			 *                     x x x
			 *                     x x x
			 after :
			 *                     x 0 0
			 *                     x x 0
			 *                     x 0 x
		  */
	template<class T>
	inline ZIRAN_FORCE_INLINE void make_lambda_shape(Matrix<T, 3, 3>& h, Matrix<T, 3, 3>& u, Matrix<T, 3, 3>& v) {
		u = Matrix<T, 3, 3>::Identity();
		v = Matrix<T, 3, 3>::Identity();

		/**
			  Reduce H to of form
			  *                    x x 0
			  *                    x x x
			  *                    x x x
			  */

		GivensRotation<T> r1(h(0, 1), h(0, 2), 1, 2);
		r1.column_rotation(h);
		r1.column_rotation(v);

		/**
			  Reduce H to of form
			  *                    x x 0
			  *                    x x 0
			  *                    x x x
			  */

		r1.compute_unconventional(h(1, 2), h(2, 2));
		r1.row_rotation(h);
		r1.column_rotation(u);

		/**
			  Reduce H to of form
			  *                    x x 0
			  *                    x x 0
			  *                    x 0 x
			  */

		GivensRotation<T> r2(h(2, 0), h(2, 1), 0, 1);
		r2.column_rotation(h);
		r2.column_rotation(v);

		/**
			  Reduce H to of form
			  *                    x 0 0
			  *                    x x 0
			  *                    x 0 x
			  */
		r2.compute_unconventional(h(0, 1), h(1, 1));
		r2.row_rotation(h);
		r2.column_rotation(u);
	}
#endif
}// namespace math

}// namespace mn

#endif
