#ifndef __UTILITY_FUNCS_HPP_
#define __UTILITY_FUNCS_HPP_
#include "settings.h"

namespace mn {

/// assume p is already within kernel range [-1.5, 1.5]
constexpr vec3 bspline_weight(float p) {
  vec3 dw{0.f, 0.f, 0.f};
  float d = p * config::g_dx_inv; ///< normalized offset
  dw[0] = 0.5f * (1.5 - d) * (1.5 - d);
  d -= 1.0f;
  dw[1] = 0.75 - d * d;
  d = 0.5f + d;
  dw[2] = 0.5 * d * d;
  return dw;
}

constexpr int dir_offset(ivec3 d) {
  return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}
constexpr void dir_components(int dir, ivec3 &d) {
  d[2] = (dir % 3) - 1;
  d[1] = ((dir / 3) % 3) - 1;
  d[0] = ((dir / 9) % 3) - 1;
}

constexpr float compute_dt(float maxVel, const float cur, const float next,
                           const float dtDefault) noexcept {
  if (next < cur)
    return 0.f;
  float dt = dtDefault;
  if (maxVel > 0. && (maxVel = config::g_dx * .3 / maxVel) < dtDefault)
    dt = maxVel;

  if (cur + dt >= next)
    dt = next - cur;
  else if ((maxVel = (next - cur) * 0.51) < dt)
    dt = maxVel;

  return dt;
}

} // namespace mn

#endif