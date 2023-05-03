#ifndef UTILITY_FUNCS_HPP
#define UTILITY_FUNCS_HPP
#include "settings.h"

namespace mn {

//TODO: Maybe create parameters for some of this magic numbers
//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
/// assume p is already within kernel range [-1.5, 1.5]
constexpr vec3 bspline_weight(float p) {
	vec3 dw {0.0f, 0.0f, 0.0f};
	float d = p * config::G_DX_INV;///< normalized offset
	dw[0]	= 0.5f * (1.5f - d) * (1.5f - d);
	d -= 1.0f;
	dw[1] = 0.75f - d * d;
	d	  = 0.5f + d;
	dw[2] = 0.5f * d * d;
	return dw;
}

constexpr ivec3 get_block_id(const std::array<float, 3>& position) {
	return ivec3(static_cast<int>(std::lround(position[0] * config::G_DX_INV)), static_cast<int>(std::lround(position[1] * config::G_DX_INV)), static_cast<int>(std::lround(position[2] * config::G_DX_INV)));
}

constexpr int dir_offset(const std::array<int, 3>& d) {
	return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}
constexpr void dir_components(int dir, std::array<int, 3>& d) {
	d[2] = (dir % 3) - 1;
	d[1] = ((dir / 3) % 3) - 1;
	d[0] = ((dir / 9) % 3) - 1;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
constexpr Duration compute_dt(float max_vel, const Duration cur_time, const Duration next_time, const Duration dt_default) noexcept {
	//Choose dt such that particles with maximum velocity cannot move more than G_DX * CFL
	//This ensures CFL condition is satisfied
	Duration dt = dt_default;
	if(max_vel > 0.0f) {
		const Duration new_dt(config::G_DX * config::CFL / max_vel);
		dt = std::min(new_dt, dt);
	}

	//If next_time - cur_time is smaller as current dt, use this.
	dt = std::min(dt, next_time - cur_time);

	return dt;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

}// namespace mn

#endif