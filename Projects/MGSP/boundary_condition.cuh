#ifndef __BOUNDARY_CONDITION_CUH_
#define __BOUNDARY_CONDITION_CUH_
#include "grid_buffer.cuh"
#include "settings.h"
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnBase/Memory/Allocator.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>

namespace mn {

using block_signed_distance_field_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, f32_, f32_, f32_,
               f32_>; ///< sdis, gradx, grady, gradz
using signed_distance_field_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               GridDomain, attrib_layout::aos, block_signed_distance_field_>;
enum class boundary_t { Sticky, Slip, Separate };

struct SignedDistanceGrid : Instance<signed_distance_field_> {
  using base_t = Instance<signed_distance_field_>;

  constexpr auto &self() noexcept { return static_cast<base_t &>(*this); }

  template <typename Allocator>
  SignedDistanceGrid(Allocator allocator)
      : base_t{spawn<signed_distance_field_, orphan_signature>(allocator)} {
    _rotMat.set(0.f);
    _rotMat(0, 0) = _rotMat(1, 1) = _rotMat(2, 2) = 1.f;
    _trans.set(0.f);
    _transVel.set(0.f);
    _omega.set(0.f);
    _dsdt = 0.f;
    _scale = 1.f;
    _friction = 0.3f;
    _type = boundary_t::Sticky;
  }

  void init(base_t &hostData, cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(&this->ch(_0, 0, 0, 0).val_1d(_0, 0),
                                    &hostData.ch(_0, 0, 0, 0).val_1d(_0, 0),
                                    signed_distance_field_::base_t::size,
                                    cudaMemcpyDefault, stream));
  }
  constexpr vec3 get_material_velocity(const vec3 &X) {
    vec3 radius = X - _trans;
    vec3 vel{};
    vec_crossMul_vec_3D(vel.data(), _omega.data(), radius.data());
    vel += _transVel;
    return vel;
  }

  __forceinline__ __device__ auto rot_angle_to_matrix(const float omega,
                                                      const int dim) -> vec3x3 {
    vec3x3 res;
    res.set(0.f);
    if (dim == 0) {
      res(0, 0) = 1;
      res(1, 1) = res(2, 2) = cosf(omega);
      res(2, 1) = res(1, 2) = sinf(omega);
      res(1, 2) = -res(1, 2);
    } else if (dim == 1) {
      res(1, 1) = 1;
      res(0, 0) = res(2, 2) = cosf(omega);
      res(2, 0) = res(0, 2) = sinf(omega);
      res(2, 0) = -res(2, 0);
    } else if (dim == 2) {
      res(2, 2) = 1;
      res(0, 0) = res(1, 1) = cosf(omega);
      res(1, 0) = res(0, 1) = sinf(omega);
      res(0, 1) = -res(0, 1);
    }
    return res;
  }
  constexpr auto vec3_cross_vec3(vec3 v1, vec3 v2) {
    vec3 res{v1[1] * v2[2] + v1[2] * v2[1], v1[2] * v2[0] + v1[0] * v2[2],
             v1[0] * v2[1] + v1[1] * v2[0]};
    return res;
  }

  //< return signed distance value + set normal
  __forceinline__ __device__ float
  get_signed_distance_and_normal(const vec3 &X, vec3 &normal) {

    //< g_cid <=> global cell ID
    ivec3 g_cid = (X / config::g_dx).cast<int>();

    // 1. init
    float sdis_res{0.f};
    normal.set(0.f);

    // 1.1 prepare
    float W[2][2][2]; //< linear interpolation weight
    {
      vec3 dis_lb = X - (g_cid.cast<float>() *
                         config::g_dx); //< distance to the left-corner node
      float W_1d[3][2];                 //< 1d weight, [dim][node]
      for (int d = 0; d < 3; ++d) {
        W_1d[d][0] = 1 - dis_lb[d] / config::g_dx;
        W_1d[d][1] = dis_lb[d] / config::g_dx;
      }
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          for (int k = 0; k < 2; ++k)
            W[i][j][k] = W_1d[0][i] * W_1d[1][j] * W_1d[2][k];
    }

    // 2. compute signed distance and normal
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k) {
          sdis_res +=
              W[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::g_blocksize),
                                     ((g_cid[1] + j) / config::g_blocksize),
                                     ((g_cid[2] + k) / config::g_blocksize))
                                .val(_0, (g_cid[0] + i) % config::g_blocksize,
                                     (g_cid[1] + j) % config::g_blocksize,
                                     (g_cid[2] + k) % config::g_blocksize));
          normal[0] +=
              W[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::g_blocksize),
                                     ((g_cid[1] + j) / config::g_blocksize),
                                     ((g_cid[2] + k) / config::g_blocksize))
                                .val(_1, (g_cid[0] + i) % config::g_blocksize,
                                     (g_cid[1] + j) % config::g_blocksize,
                                     (g_cid[2] + k) % config::g_blocksize));
          normal[1] +=
              W[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::g_blocksize),
                                     ((g_cid[1] + j) / config::g_blocksize),
                                     ((g_cid[2] + k) / config::g_blocksize))
                                .val(_2, (g_cid[0] + i) % config::g_blocksize,
                                     (g_cid[1] + j) % config::g_blocksize,
                                     (g_cid[2] + k) % config::g_blocksize));
          normal[2] +=
              W[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::g_blocksize),
                                     ((g_cid[1] + j) / config::g_blocksize),
                                     ((g_cid[2] + k) / config::g_blocksize))
                                .val(_3, (g_cid[0] + i) % config::g_blocksize,
                                     (g_cid[1] + j) % config::g_blocksize,
                                     (g_cid[2] + k) % config::g_blocksize));
        }
    normal /= sqrtf(normal.l2NormSqr());
    return sdis_res;
  }
  __forceinline__ __device__ bool query_sdf(vec3 &normal, const vec3 &X) {
    if (X[0] < g_bc * config::g_dx * config::g_blocksize ||
        X[0] >= (GridDomain::range(_0) - g_bc) * config::g_blocksize *
                    config::g_dx ||
        X[1] < g_bc * config::g_dx * config::g_blocksize ||
        X[1] >= (GridDomain::range(_1) - g_bc) * config::g_blocksize *
                    config::g_dx ||
        X[2] < g_bc * config::g_dx * config::g_blocksize ||
        X[2] >=
            (GridDomain::range(_2) - g_bc) * config::g_blocksize * config::g_dx)
      return false;
    return get_signed_distance_and_normal(X, normal) <= 0.f;
  }
  //< detect if there is collision with the object, if there is, reset grid
  // velocity < call this inside grid update kernel
  /*  Takes a position and its velocity,
   *  project the grid velocity, //? [to check] and returns a normal if the
   * collision happened as a SLIP collsion.
   *
   * derivation:
   *
   *     x = \phi(X,t) = R(t)s(t)X+b(t)
   *     X = \phi^{-1}(x,t) = (1/s) R^{-1} (x-b)
   *     V(X,t) = \frac{\partial \phi}{\partial t}
   *            = R'sX + Rs'X + RsX' + b'
   *     v(x,t) = V(\phi^{-1}(x,t),t)
   *            = R'R^{-1}(x-b) + (s'/s)(x-b) + RsX' + b'
   *            = omega \cross (x-b) + (s'/s)(x-b) + RsV + b'*/
  __forceinline__ __device__ void
  detect_and_resolve_collision(const ivec3 block_id, const ivec3 cell_id,
                               float currentTime, vec3 &vel) {
    vec3 x_minus_trans =
        (block_id * config::g_blocksize + cell_id).cast<float>() *
            config::g_dx -
        (_trans + _transVel * currentTime);
    // material space
    vec3 X;
    vec3x3 rotMat = _rotMat;
    {
      vec3 X0 = x_minus_trans * (1.f / (1.f + _dsdt * currentTime));

      vec3x3 rot_tmp = rot_angle_to_matrix(_omega[0] * currentTime, 0);
      vec3x3 prevRot = rotMat;
      matrixMatrixMultiplication3d(prevRot.data(), rot_tmp.data(),
                                   rotMat.data());

      rot_tmp = rot_angle_to_matrix(_omega[1] * currentTime, 1);
      prevRot = rotMat;
      matrixMatrixMultiplication3d(prevRot.data(), rot_tmp.data(),
                                   rotMat.data());

      rot_tmp = rot_angle_to_matrix(_omega[2] * currentTime, 2);
      prevRot = rotMat;
      matrixMatrixMultiplication3d(prevRot.data(), rot_tmp.data(),
                                   rotMat.data());
      matT_mul_vec_3D(X.data(), rotMat.data(), X0.data());
    }

    X = X * _scale + _trans;

    //< enforce BC if inside LS
    vec3 obj_normal;
    bool hit = query_sdf(obj_normal, X); //< material space normal
    if (hit) {
      ///< calculate object velocity in deformation space
      vec3 v_object = vec3_cross_vec3(_omega, x_minus_trans);
      v_object += x_minus_trans * (_dsdt / _scale);
      {
        vec3 rot_V;
        matrixVectorMultiplication3d(
            rotMat.data(), get_material_velocity(X).data(), rot_V.data());
        v_object += rot_V * _scale + _transVel;
      }
      vel -= v_object;

      /// sticky
      if (_type == boundary_t::Sticky)
        vel.set(0.f);
      /// slip
      else if (_type == boundary_t::Slip) {
        {
          vec3 n;
          matrixVectorMultiplication3d(rotMat.data(), obj_normal.data(),
                                       n.data());
          obj_normal = n;
        }
        float v_dot_n = obj_normal.dot(vel);
        vel -= (obj_normal * v_dot_n);
        if (_friction > 0.f) {
          if (v_dot_n < 0) {
            float velNorm = sqrtf(vel.l2NormSqr());
            if (-v_dot_n * _friction < velNorm)
              vel += (vel / velNorm * (v_dot_n * _friction));
            else
              vel.set(0.f);
          }
        }
      }
      /// sticky
      else if (_type == boundary_t::Separate) {
        if (obj_normal[0] == 0.f && obj_normal[1] == 0.f &&
            obj_normal[2] == 0.f) {
          vel.set(0.f);
          return;
        }
        {
          vec3 n;
          matrixVectorMultiplication3d(rotMat.data(), obj_normal.data(),
                                       n.data());
          obj_normal = n;
        }
        float v_dot_n = obj_normal.dot(vel);
        if (v_dot_n < 0) {
          vel -= (obj_normal * v_dot_n);
          if (_friction != 0) {
            float velNorm = sqrtf(vel.l2NormSqr());
            if (-v_dot_n * _friction < velNorm)
              vel += (vel / velNorm * (v_dot_n * _friction));
            else
              vel.set(0.f);
          }
        }
      } else
        printf("[ERROR] Wrong Boundary Type!\n");
      vel += v_object;
    }
  }
  vec3x3 _rotMat;
  vec3 _trans, _transVel, _omega;
  float _dsdt, _scale;
  float _friction;
  boundary_t _type;
};

template <place_id Chn>
void fillSignedDistanceField(std::integral_constant<place_id, Chn> chn,
                             const std::vector<float> &s_dis,
                             Instance<signed_distance_field_> &hostData) {
  int insideNodeNum = 0;
  for (auto bx = 0; bx < GridDomain::range(_0); ++bx)
    for (auto by = 0; by < GridDomain::range(_1); ++by)
      for (auto bz = 0; bz < GridDomain::range(_2); ++bz) {
        auto sdis_block = hostData.ch(_0, bx, by, bz);
        for (auto cx = 0; cx < config::g_blocksize; ++cx) {
          auto i = bx * config::g_blocksize + cx;
          for (auto cy = 0; cy < config::g_blocksize; ++cy) {
            auto j = by * config::g_blocksize + cy;
            for (auto cz = 0; cz < config::g_blocksize; ++cz) {
              auto k = bz * config::g_blocksize + cz;
              auto idx = (i * GridDomain::range(_1) * config::g_blocksize *
                          GridDomain::range(_2) * config::g_blocksize) +
                         (j * GridDomain::range(_2) * config::g_blocksize) + k;
              sdis_block.val(chn, cx, cy, cz) = s_dis[idx];
              if (Chn == 0)
                if (s_dis[idx] <= 0.f)
                  insideNodeNum++;
            }
          }
        }
      }
  if (Chn == 0)
    fmt::print("[Collision Object]\n\t[from saved signed_distance_field] "
               "Finish init signed distance buffer, inside node num = {}.\n",
               insideNodeNum);
  else
    fmt::print("[Collision Object]\n\t[from saved signed_distance_field] "
               "Finish init signed distance gradient [{}].\n",
               Chn);
}

void initFromSignedDistanceFile(std::string filename,
                                vec<std::size_t, 3> resolution,
                                Instance<signed_distance_field_> &hostData) {
  std::string fileAddr = std::string(AssetDirPath) + "vdbSDF/";
  std::vector<float> sdisf(resolution.prod());
  auto readFile = [&](std::string suffix) {
    auto fn = fopen((fileAddr + filename + suffix).c_str(), "rb");
    std::size_t readNum =
        std::fread((float *)sdisf.data(), sizeof(float), sdisf.size(), fn);
    if (readNum != (std::size_t)resolution.prod()) {
      printf("Error in loading file [%s]: read in %d entries, should be %d\n",
             filename.c_str(), (int)readNum, (int)resolution.prod());
      exit(0);
    }
    std::fclose(fn);
  };
  readFile("_sdf.bin");
  fillSignedDistanceField(_0, sdisf, hostData);
  readFile("_grad_0.bin");
  fillSignedDistanceField(_1, sdisf, hostData);
  readFile("_grad_1.bin");
  fillSignedDistanceField(_2, sdisf, hostData);
  readFile("_grad_2.bin");
  fillSignedDistanceField(_3, sdisf, hostData);
}

} // namespace mn

#endif