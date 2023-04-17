#ifndef CONSTITUTIVE_MODELS_CUH
#define CONSTITUTIVE_MODELS_CUH
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnBase/Math/Vec.h>

#include <MnBase/Math/Matrix/svd.cuh>

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define USE_JOSH_FRACTURE_PAPER 1//Selects which solve is used

namespace mn {

//TODO: But maybe use names instead for better understanding
//NOLINTBEGIN(readability-magic-numbers, readability-identifier-naming) Magic numbers are formula specific; Common naming for this physical formulas
template<typename T = float>
__forceinline__ __device__ void compute_stress_fixed_corotated(T volume, T mu, T lambda, const vec<T, 9>& F, vec<T, 9>& PF) {
	std::array<T, 9> U = {};
	std::array<T, 3> S = {};
	std::array<T, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	T J				= S[0] * S[1] * S[2];
	T scaled_mu		= 2.0f * mu;
	T scaled_lambda = lambda * (J - 10.f);
	vec<T, 3> P_hat;
	P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
	P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
	P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

	vec<T, 9> P;
	P[0] = P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
	P[1] = P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
	P[2] = P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
	P[3] = P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
	P[4] = P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
	P[5] = P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
	P[6] = P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
	P[7] = P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
	P[8] = P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

	/// PF'
	PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
	PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
	PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
	PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
	PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
	PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
	PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
	PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
	PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}
//NOLINTEND(readability-magic-numbers, readability-identifier-naming)

//TODO: But maybe use names instead for better understanding
//NOLINTBEGIN(readability-magic-numbers, readability-identifier-naming) Magic numbers are formula specific; Common naming for this physical formulas
template<typename T = float>
__forceinline__ __device__ void compute_stress_nacc(T volume, T mu, T lambda, T bm, T xi, T beta, T msqr, bool hardening_on, T& log_jp, vec<T, 9>& F, vec<T, 9>& PF) {
	(void) lambda;

	std::array<T, 9> U = {};
	std::array<T, 3> S = {};
	std::array<T, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	T p0	= bm * (static_cast<T>(0.00001) + sinh(xi * (-log_jp > 0 ? -log_jp : 0)));
	T p_min = -beta * p0;

	T Je_trial = S[0] * S[1] * S[2];

	///< 0). calculate YS
	std::array<T, 3> B_hat_trial = {S[0] * S[0], S[1] * S[1], S[2] * S[2]};
	T trace_B_hat_trial_divdim	 = (B_hat_trial[0] + B_hat_trial[1] + B_hat_trial[2]) / 3.f;
	T J_power_neg_2_d_mulmu		 = mu * powf(Je_trial, -2.f / 3.f);///< J^(-2/dim) * mu
	std::array<T, 3> s_hat_trial = {J_power_neg_2_d_mulmu * (B_hat_trial[0] - trace_B_hat_trial_divdim), J_power_neg_2_d_mulmu * (B_hat_trial[1] - trace_B_hat_trial_divdim), J_power_neg_2_d_mulmu * (B_hat_trial[2] - trace_B_hat_trial_divdim)};
	T psi_kappa_partial_J		 = bm * 0.5f * (Je_trial - 1.f / Je_trial);
	T p_trial					 = -psi_kappa_partial_J * Je_trial;

	T y_s_half_coeff	  = 3.f / 2.f * (1 + 2.f * beta);///< a
	T y_p_half			  = (msqr * (p_trial - p_min) * (p_trial - p0));
	T s_hat_trial_sqrnorm = s_hat_trial[0] * s_hat_trial[0] + s_hat_trial[1] * s_hat_trial[1] + s_hat_trial[2] * s_hat_trial[2];
	T y					  = (y_s_half_coeff * s_hat_trial_sqrnorm) + y_p_half;

	//< 1). update strain and hardening alpha(in log_jp)

	///< case 1, project to max tip of YS
	if(p_trial > p0) {
		T Je_new = sqrtf(-2.f * p0 / bm + 1.f);
		S[0] = S[1] = S[2]	   = powf(Je_new, 1.f / 3.f);
		std::array<T, 9> New_F = {};
		matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(hardening_on) {
			log_jp += logf(Je_trial / Je_new);
		}
	}///< case 1 -- end

	/// case 2, project to min tip of YS
	else if(p_trial < p_min) {
		T Je_new = sqrtf(-2.f * p_min / bm + 1.f);
		S[0] = S[1] = S[2]	   = powf(Je_new, 1.f / 3.f);
		std::array<T, 9> New_F = {};
		matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(hardening_on) {
			log_jp += logf(Je_trial / Je_new);
		}
	}///< case 2 -- end

	/// case 3, keep or project to YS by hardening
	else {
		///< outside YS
		if(y >= 1e-4) {
			////< yield surface projection
			T B_s_coeff = powf(Je_trial, 2.f / 3.f) / mu * sqrtf(-y_p_half / y_s_half_coeff) / sqrtf(s_hat_trial_sqrnorm);
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				S[i] = sqrtf(s_hat_trial[i] * B_s_coeff + trace_B_hat_trial_divdim);
			}
			std::array<T, 9> New_F = {};
			matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
			for(int i = 0; i < 9; i++) {
				F[i] = New_F[i];
			}

			////< hardening
			if(hardening_on && p0 > 1e-4 && p_trial < p0 - 1e-4 && p_trial > 1e-4 + p_min) {
				T p_center = (static_cast<T>(1.0) - beta) * p0 / 2;
#if USE_JOSH_FRACTURE_PAPER/// solve in 19 Josh Fracture paper
				T q_trial				   = sqrtf(3.f / 2.f * s_hat_trial_sqrnorm);
				std::array<T, 2> direction = {p_center - p_trial, -q_trial};
				T direction_norm		   = sqrtf(direction[0] * direction[0] + direction[1] * direction[1]);
				direction[0] /= direction_norm;
				direction[1] /= direction_norm;

				T C = msqr * (p_center - p_min) * (p_center - p0);
				T B = msqr * direction[0] * (2 * p_center - p0 - p_min);
				T A = msqr * direction[0] * direction[0] + (1 + 2 * beta) * direction[1] * direction[1];

				T l1 = (-B + sqrtf(B * B - 4 * A * C)) / (2 * A);
				T l2 = (-B - sqrtf(B * B - 4 * A * C)) / (2 * A);

				T p1 = p_center + l1 * direction[0];
				T p2 = p_center + l2 * direction[0];
#else/// solve in ziran - Compare_With_Physbam
				T aa = msqr * powf(p_trial - p_center, 2) / (y_s_half_coeff * s_hat_trial_sqrnorm);
				T dd = 1 + aa;
				T ff = aa * beta * p0 - aa * p0 - 2 * p_center;
				T gg = (p_center * p_center) - aa * beta * (p0 * p0);
				T zz = sqrtf(fabsf(ff * ff - 4 * dd * gg));
				T p1 = (-ff + zz) / (2 * dd);
				T p2 = (-ff - zz) / (2 * dd);
#endif

				T p_fake	  = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;
				T tmp_Je_sqr  = (-2 * p_fake / bm + 1);
				T Je_new_fake = sqrtf(tmp_Je_sqr > 0 ? tmp_Je_sqr : -tmp_Je_sqr);
				if(Je_new_fake > 1e-4) {
					log_jp += logf(Je_trial / Je_new_fake);
				}
			}
		}///< outside YS -- end
	}	 ///< case 3 --end

	//< 2). elasticity
	///< known: F(renewed), U, V, S(renewed)
	///< unknown: J, dev(FF^T)
	T J					   = S[0] * S[1] * S[2];
	std::array<T, 9> b_dev = {};
	std::array<T, 9> b	   = {};
	matrix_matrix_tranpose_multiplication_3d(F.data_arr(), b);
	matrix_deviatoric_3d(b, b_dev);

	///< |f| = P * F^T * Volume
	T dev_b_coeff = mu * powf(J, -2.f / 3.f);
	T i_coeff	  = bm * .5f * (J * J - 1.f);
	PF[0]		  = (dev_b_coeff * b_dev[0] + i_coeff) * volume;
	PF[1]		  = (dev_b_coeff * b_dev[1]) * volume;
	PF[2]		  = (dev_b_coeff * b_dev[2]) * volume;
	PF[3]		  = (dev_b_coeff * b_dev[3]) * volume;
	PF[4]		  = (dev_b_coeff * b_dev[4] + i_coeff) * volume;
	PF[5]		  = (dev_b_coeff * b_dev[5]) * volume;
	PF[6]		  = (dev_b_coeff * b_dev[6]) * volume;
	PF[7]		  = (dev_b_coeff * b_dev[7]) * volume;
	PF[8]		  = (dev_b_coeff * b_dev[8] + i_coeff) * volume;
}
//NOLINTEND(readability-magic-numbers, readability-identifier-naming)

//TODO: But maybe use names instead for better understanding
//NOLINTBEGIN(readability-magic-numbers, readability-identifier-naming) Magic numbers are formula specific; Common naming for this physical formulas
template<typename T = float>
__forceinline__ __device__ void compute_stress_sand(T volume, T mu, T lambda, T cohesion, T beta, T yield_surface, bool vol_correction, T& log_jp, vec<T, 9>& F, vec<T, 9>& PF) {
	std::array<T, 9> U = {};
	std::array<T, 3> S = {};
	std::array<T, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	T scaled_mu = static_cast<T>(2.0) * mu;

	std::array<T, 3> epsilon = {};
	std::array<T, 3> New_S	 = {};///< helper
	std::array<T, 9> New_F	 = {};

#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		T abs_S	   = S[i] > 0 ? S[i] : -S[i];
		abs_S	   = abs_S > static_cast<T>(1e-4) ? abs_S : static_cast<T>(1e-4);
		epsilon[i] = logf(abs_S) - cohesion;
	}
	T sum_epsilon	= epsilon[0] + epsilon[1] + epsilon[2];
	T trace_epsilon = sum_epsilon + log_jp;

	std::array<T, 3> epsilon_hat = {};
#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		epsilon_hat[i] = epsilon[i] - (trace_epsilon / static_cast<T>(3.0));
	}

	T epsilon_hat_norm = sqrtf(epsilon_hat[0] * epsilon_hat[0] + epsilon_hat[1] * epsilon_hat[1] + epsilon_hat[2] * epsilon_hat[2]);

	/* Calculate Plasticiy */
	if(trace_epsilon >= static_cast<T>(0.0)) {///< case II: project to the cone tip
		New_S[0] = New_S[1] = New_S[2] = expf(cohesion);
		matmul_mat_diag_mat_t_3d(New_F, U, New_S, V);// new F_e
													 /* Update F */
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(vol_correction) {
			log_jp = beta * sum_epsilon + log_jp;
		}
	} else if(mu != 0) {
		log_jp			   = 0;
		T delta_gamma	   = epsilon_hat_norm + (static_cast<T>(3.0) * lambda + scaled_mu) / scaled_mu * trace_epsilon * yield_surface;
		std::array<T, 3> H = {};
		if(delta_gamma <= 0) {///< case I: inside the yield surface cone
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				H[i] = epsilon[i] + cohesion;
			}
		} else {///< case III: project to the cone surface
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				H[i] = epsilon[i] - (delta_gamma / epsilon_hat_norm) * epsilon_hat[i] + cohesion;
			}
		}
#pragma unroll 3
		for(int i = 0; i < 3; i++) {
			New_S[i] = expf(H[i]);
		}
		matmul_mat_diag_mat_t_3d(New_F, U, New_S, V);// new F_e
													 /* Update F */
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
	} else {
		//TODO: What to do here? Just don't change values?
	}

	/* Elasticity -- Calculate Coefficient */
	std::array<T, 3> New_S_log = {logf(New_S[0]), logf(New_S[1]), logf(New_S[2])};
	std::array<T, 3> P_hat	   = {};

	// T S_inverse[3] = {1.f/S[0], 1.f/S[1], 1.f/S[2]};  // TO CHECK
	// T S_inverse[3] = {1.f / New_S[0], 1.f / New_S[1], 1.f / New_S[2]}; // TO
	// CHECK
	T trace_log_S = New_S_log[0] + New_S_log[1] + New_S_log[2];
#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		P_hat[i] = (scaled_mu * New_S_log[i] + lambda * trace_log_S) / New_S[i];
	}

	std::array<T, 9> P = {};
	matmul_mat_diag_mat_t_3d(P, U, P_hat, V);
	///< |f| = P * F^T * Volume
	PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
	PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
	PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
	PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
	PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
	PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
	PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
	PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
	PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}
//NOLINTEND(readability-magic-numbers, readability-identifier-naming)

}// namespace mn

#endif