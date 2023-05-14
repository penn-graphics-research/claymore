#ifndef UTILITY_FUNCS_HPP
#define UTILITY_FUNCS_HPP
#include "settings.h"

namespace mn {

//TODO: Maybe create parameters fopr some of this magic numbers
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

constexpr int dir_offset(ivec3 d) {
	return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}
constexpr void dir_components(int dir, ivec3& d) {
	d[2] = (dir % 3) - 1;
	d[1] = ((dir / 3) % 3) - 1;
	d[0] = ((dir / 9) % 3) - 1;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
constexpr float compute_dt(float max_vel, const float cur, const float next, const float dt_default) noexcept {
	if(next < cur) {
		return 0.0f;
	}

	float dt = dt_default;
	if(max_vel > 0.0f) {
		max_vel = config::G_DX * 0.3f / max_vel;
		if(max_vel < dt_default) {
			dt = max_vel;
		}
	}

	if(cur + dt >= next) {
		dt = next - cur;
	} else {
		max_vel = (next - cur) * 0.51f;
		if(max_vel < dt) {
			dt = max_vel;
		}
	}

	return dt;
}
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

}// namespace mn

#endif