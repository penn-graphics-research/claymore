#ifndef CONSTITUTIVE_MODELS_CUH
#define CONSTITUTIVE_MODELS_CUH
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnBase/Math/Vec.h>

#include <MnBase/Math/Matrix/svd.cuh>

#include "settings.h"

//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define USE_JOSH_FRACTURE_PAPER 1//Selects which solve is used

namespace mn {
	
//Need this, cause we cannot partially instantiate function templates in current c++ version
template<typename T = float>
struct ComputeStressIntermediate{
	T bm;
	T xi;
	T beta;
	T msqr;
	T log_jp;
	T cohesion;
	T yield_surface;
	bool hardening_on;
	bool volume_correction;
};

template<typename T = float, MaterialE MaterialType>
__forceinline__ __device__ void compute_stress(const T volume, const T mu, const T lambda, std::array<T, 9>& F, std::array<T, 9>& PF, ComputeStressIntermediate<T>& data);

//TODO: But maybe use names instead for better understanding
//NOLINTBEGIN(readability-magic-numbers, readability-identifier-naming) Magic numbers are formula specific; Common naming for this physical formulas
template<>
__forceinline__ __device__ void compute_stress<float, MaterialE::FIXED_COROTATED>(const float volume, const float mu, const float lambda, std::array<float, 9>& F, std::array<float, 9>& PF, ComputeStressIntermediate<float>& data) {
	std::array<float, 9> U = {};
	std::array<float, 3> S = {};
	std::array<float, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	float J				= S[0] * S[1] * S[2];
	float scaled_mu		= 2.0f * mu;
	float scaled_lambda = lambda * (J - 1.0f);
	vec<float, 3> P_hat;
	P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
	P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
	P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

	vec<float, 9> P;
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
template<>
__forceinline__ __device__ void compute_stress<float, MaterialE::NACC>(const float volume, const float mu, const float lambda, std::array<float, 9>& F, std::array<float, 9>& PF, ComputeStressIntermediate<float>& data) {
	(void) lambda;

	std::array<float, 9> U = {};
	std::array<float, 3> S = {};
	std::array<float, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	float p0	= data.bm * (static_cast<float>(0.00001) + sinh(data.xi * (-data.log_jp > 0 ? -data.log_jp : 0)));
	float p_min = -data.beta * p0;

	float Je_trial = S[0] * S[1] * S[2];

	///< 0). calculate YS
	std::array<float, 3> B_hat_trial = {S[0] * S[0], S[1] * S[1], S[2] * S[2]};
	float trace_B_hat_trial_divdim	 = (B_hat_trial[0] + B_hat_trial[1] + B_hat_trial[2]) / 3.f;
	float J_power_neg_2_d_mulmu		 = mu * powf(Je_trial, -2.f / 3.f);///< J^(-2/dim) * mu
	std::array<float, 3> s_hat_trial = {J_power_neg_2_d_mulmu * (B_hat_trial[0] - trace_B_hat_trial_divdim), J_power_neg_2_d_mulmu * (B_hat_trial[1] - trace_B_hat_trial_divdim), J_power_neg_2_d_mulmu * (B_hat_trial[2] - trace_B_hat_trial_divdim)};
	float psi_kappa_partial_J		 = data.bm * 0.5f * (Je_trial - 1.f / Je_trial);
	float p_trial					 = -psi_kappa_partial_J * Je_trial;

	float y_s_half_coeff	  = 3.f / 2.f * (1 + 2.f * data.beta);///< a
	float y_p_half			  = (data.msqr * (p_trial - p_min) * (p_trial - p0));
	float s_hat_trial_sqrnorm = s_hat_trial[0] * s_hat_trial[0] + s_hat_trial[1] * s_hat_trial[1] + s_hat_trial[2] * s_hat_trial[2];
	float y					  = (y_s_half_coeff * s_hat_trial_sqrnorm) + y_p_half;

	//< 1). update strain and hardening alpha(in log_jp)

	///< case 1, project to max tip of YS
	if(p_trial > p0) {
		float Je_new = sqrtf(-2.f * p0 / data.bm + 1.f);
		S[0] = S[1] = S[2]	   = powf(Je_new, 1.f / 3.f);
		std::array<float, 9> New_F = {};
		matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(data.hardening_on) {
			data.log_jp += logf(Je_trial / Je_new);
		}
	}///< case 1 -- end

	/// case 2, project to min tip of YS
	else if(p_trial < p_min) {
		float Je_new = sqrtf(-2.f * p_min / data.bm + 1.f);
		S[0] = S[1] = S[2]	   = powf(Je_new, 1.f / 3.f);
		std::array<float, 9> New_F = {};
		matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(data.hardening_on) {
			data.log_jp += logf(Je_trial / Je_new);
		}
	}///< case 2 -- end

	/// case 3, keep or project to YS by hardening
	else {
		///< outside YS
		if(y >= 1e-4) {
			////< yield surface projection
			float B_s_coeff = powf(Je_trial, 2.f / 3.f) / mu * sqrtf(-y_p_half / y_s_half_coeff) / sqrtf(s_hat_trial_sqrnorm);
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				S[i] = sqrtf(s_hat_trial[i] * B_s_coeff + trace_B_hat_trial_divdim);
			}
			std::array<float, 9> New_F = {};
			matmul_mat_diag_mat_t_3d(New_F, U, S, V);
#pragma unroll 9
			for(int i = 0; i < 9; i++) {
				F[i] = New_F[i];
			}

			////< hardening
			if(data.hardening_on && p0 > 1e-4 && p_trial < p0 - 1e-4 && p_trial > 1e-4 + p_min) {
				float p_center = (static_cast<float>(1.0) - data.beta) * p0 / 2;
#if USE_JOSH_FRACTURE_PAPER/// solve in 19 Josh Fracture paper
				float q_trial				   = sqrtf(3.f / 2.f * s_hat_trial_sqrnorm);
				std::array<float, 2> direction = {p_center - p_trial, -q_trial};
				float direction_norm		   = sqrtf(direction[0] * direction[0] + direction[1] * direction[1]);
				direction[0] /= direction_norm;
				direction[1] /= direction_norm;

				float C = data.msqr * (p_center - p_min) * (p_center - p0);
				float B = data.msqr * direction[0] * (2 * p_center - p0 - p_min);
				float A = data.msqr * direction[0] * direction[0] + (1 + 2 * data.beta) * direction[1] * direction[1];

				float l1 = (-B + sqrtf(B * B - 4 * A * C)) / (2 * A);
				float l2 = (-B - sqrtf(B * B - 4 * A * C)) / (2 * A);

				float p1 = p_center + l1 * direction[0];
				float p2 = p_center + l2 * direction[0];
#else/// solve in ziran - Compare_With_Physbam
				float aa = data.msqr * powf(p_trial - p_center, 2) / (y_s_half_coeff * s_hat_trial_sqrnorm);
				float dd = 1 + aa;
				float ff = aa * data.beta * p0 - aa * p0 - 2 * p_center;
				float gg = (p_center * p_center) - aa * data.beta * (p0 * p0);
				float zz = sqrtf(fabsf(ff * ff - 4 * dd * gg));
				float p1 = (-ff + zz) / (2 * dd);
				float p2 = (-ff - zz) / (2 * dd);
#endif

				float p_fake	  = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;
				float tmp_Je_sqr  = (-2 * p_fake / data.bm + 1);
				float Je_new_fake = sqrtf(tmp_Je_sqr > 0 ? tmp_Je_sqr : -tmp_Je_sqr);
				if(Je_new_fake > 1e-4) {
					data.log_jp += logf(Je_trial / Je_new_fake);
				}
			}
		}///< outside YS -- end
	}	 ///< case 3 --end

	//< 2). elasticity
	///< known: F(renewed), U, V, S(renewed)
	///< unknown: J, dev(FF^T)
	float J					   = S[0] * S[1] * S[2];
	std::array<float, 9> b_dev = {};
	std::array<float, 9> b	   = {};
	matrix_matrix_tranpose_multiplication_3d(F, b);
	matrix_deviatoric_3d(b, b_dev);
	
	//FIXME: Fail-safe test  for precision error caused by compiler. Not placed in matrix_deviatoric_3d cause error does not seem to be detectable in there
	if(
		   b[0] == 1.0f
		&& b[1] == 1.0f
		&& b[2] == 1.0f
		&& (
			b_dev[0] != 0.0f
			||b_dev[4] != 0.0f
			||b_dev[8] != 0.0f
		)
	){
		printf("matrix_deviatoric_3d failed to get things write (probably compiler issue)");
	}

	///< |f| = P * F^T * Volume
	float dev_b_coeff = mu * powf(J, -2.f / 3.f);
	float i_coeff	  = data.bm * .5f * (J * J - 1.f);
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
template<>
__forceinline__ __device__ void compute_stress<float, MaterialE::SAND>(const float volume, const float mu, const float lambda, std::array<float, 9>& F, std::array<float, 9>& PF, ComputeStressIntermediate<float>& data) {
	std::array<float, 9> U = {};
	std::array<float, 3> S = {};
	std::array<float, 9> V = {};
	math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
	float scaled_mu = static_cast<float>(2.0) * mu;

	std::array<float, 3> epsilon = {};
	std::array<float, 3> New_S	 = {};///< helper
	std::array<float, 9> New_F	 = {};

#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		float abs_S	   = S[i] > 0 ? S[i] : -S[i];
		abs_S	   = abs_S > static_cast<float>(1e-4) ? abs_S : static_cast<float>(1e-4);
		epsilon[i] = logf(abs_S) - data.cohesion;
	}
	float sum_epsilon	= epsilon[0] + epsilon[1] + epsilon[2];
	float trace_epsilon = sum_epsilon + data.log_jp;

	std::array<float, 3> epsilon_hat = {};
#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		epsilon_hat[i] = epsilon[i] - (trace_epsilon / static_cast<float>(3.0));
	}

	float epsilon_hat_norm = sqrtf(epsilon_hat[0] * epsilon_hat[0] + epsilon_hat[1] * epsilon_hat[1] + epsilon_hat[2] * epsilon_hat[2]);

	/* Calculate Plasticiy */
	if(trace_epsilon >= static_cast<float>(0.0)) {///< case II: project to the cone tip
		New_S[0] = New_S[1] = New_S[2] = expf(data.cohesion);
		matmul_mat_diag_mat_t_3d(New_F, U, New_S, V);// new F_e
													 /* Update F */
#pragma unroll 9
		for(int i = 0; i < 9; i++) {
			F[i] = New_F[i];
		}
		if(data.volume_correction) {
			data.log_jp = data.beta * sum_epsilon + data.log_jp;
		}
	} else if(mu != 0) {
		data.log_jp			   = 0;
		float delta_gamma	   = epsilon_hat_norm + (static_cast<float>(3.0) * lambda + scaled_mu) / scaled_mu * trace_epsilon * data.yield_surface;
		std::array<float, 3> H = {};
		if(delta_gamma <= 0) {///< case I: inside the yield surface cone
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				H[i] = epsilon[i] + data.cohesion;
			}
		} else {///< case III: project to the cone surface
#pragma unroll 3
			for(int i = 0; i < 3; i++) {
				H[i] = epsilon[i] - (delta_gamma / epsilon_hat_norm) * epsilon_hat[i] + data.cohesion;
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
	std::array<float, 3> New_S_log = {logf(New_S[0]), logf(New_S[1]), logf(New_S[2])};
	std::array<float, 3> P_hat	   = {};

	// float S_inverse[3] = {1.f/S[0], 1.f/S[1], 1.f/S[2]};  // TO CHECK
	// float S_inverse[3] = {1.f / New_S[0], 1.f / New_S[1], 1.f / New_S[2]}; // TO
	// CHECK
	float trace_log_S = New_S_log[0] + New_S_log[1] + New_S_log[2];
	
#pragma unroll 3
	for(int i = 0; i < 3; i++) {
		P_hat[i] = (scaled_mu * New_S_log[i] + lambda * trace_log_S) / New_S[i];
	}

	std::array<float, 9> P = {};
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