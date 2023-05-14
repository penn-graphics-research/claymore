#ifndef BOUNDARY_CONDITION_CUH
#define BOUNDARY_CONDITION_CUH
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnBase/Memory/Allocator.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <vector>

#include "grid_buffer.cuh"
#include "settings.h"

namespace mn {

using block_signed_distance_field_ = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, BlockDomain, attrib_layout::SOA, f32_, f32_, f32_, f32_>;///< sdis, gradx, grady, gradz
using signed_distance_field_	   = Structural<StructuralType::DENSE, Decorator<StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy::SUM_POW2_ALIGN>, GridDomain, attrib_layout::AOS, block_signed_distance_field_>;

enum class BoundaryT {
	STICKY,
	SLIP,
	SEPARATE
};

//NOLINTBEGIN(readability-magic-numbers) Parameter definitions and formulas
struct SignedDistanceGrid : Instance<signed_distance_field_> {
	using base_t = Instance<signed_distance_field_>;

	vec3x3 rot_mat;
	vec3 trans;
	vec3 trans_vel;
	vec3 omega;
	float dsdt;
	float scale;
	float friction;
	BoundaryT type;

	template<typename Allocator>
	explicit SignedDistanceGrid(Allocator allocator)
		: base_t {spawn<signed_distance_field_, orphan_signature>(allocator)}
		, dsdt(0.0f)
		, scale(1.0f)
		, friction(0.3f)
		, type(BoundaryT::STICKY) {
		rot_mat.set(0.0f);
		rot_mat(0, 0) = rot_mat(1, 1) = rot_mat(2, 2) = 1.f;
		trans.set(0.f);
		trans_vel.set(0.f);
		omega.set(0.f);
	}

	void init(base_t& host_data, cudaStream_t stream) {
		check_cuda_errors(cudaMemcpyAsync(&this->ch(_0, 0, 0, 0).val_1d(_0, 0), &host_data.ch(_0, 0, 0, 0).val_1d(_0, 0), signed_distance_field_::base_t::size, cudaMemcpyDefault, stream));
	}

	constexpr auto& self() noexcept {
		return static_cast<base_t&>(*this);
	}

	constexpr vec3 get_material_velocity(const vec3& x) {
		vec3 radius = x - trans;
		vec3 vel {};
		vec_cross_mul_vec_3d(vel.data_arr(), omega.data_arr(), radius.data_arr());
		vel += trans_vel;
		return vel;
	}

	static __forceinline__ __device__ auto rot_angle_to_matrix(const float omega, const int dim) -> vec3x3 {
		vec3x3 res;
		res.set(0.0f);
		if(dim == 0) {
			res(0, 0) = 1;
			res(1, 1) = res(2, 2) = cosf(omega);
			res(2, 1) = res(1, 2) = sinf(omega);
			res(1, 2)			  = -res(1, 2);
		} else if(dim == 1) {
			res(1, 1) = 1;
			res(0, 0) = res(2, 2) = cosf(omega);
			res(2, 0) = res(0, 2) = sinf(omega);
			res(2, 0)			  = -res(2, 0);
		} else if(dim == 2) {
			res(2, 2) = 1;
			res(0, 0) = res(1, 1) = cosf(omega);
			res(1, 0) = res(0, 1) = sinf(omega);
			res(0, 1)			  = -res(0, 1);
		} else {
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print
			printf("Unsupported dimension");
		}
		return res;
	}

	static constexpr auto vec3_cross_vec3(vec3 v1, vec3 v2) {
		vec3 res {v1[1] * v2[2] + v1[2] * v2[1], v1[2] * v2[0] + v1[0] * v2[2], v1[0] * v2[1] + v1[1] * v2[0]};
		return res;
	}

	//< return signed distance value + set normal
	__forceinline__ __device__ float get_signed_distance_and_normal(const vec3& x, vec3& normal) {
		//< g_cid <=> global cell ID
		ivec3 g_cid = (x / config::G_DX).cast<int>();

		// 1. init
		float sdis_res {0.f};
		normal.set(0.f);

		// 1.1 prepare
		std::array<std::array<std::array<float, 2>, 2>, 2> w = {};//< linear interpolation weight
		{
			vec3 dis_lb								 = x - (g_cid.cast<float>() * config::G_DX);//< distance to the left-corner node
			std::array<std::array<float, 2>, 3> w_1d = {};										//< 1d weight, [dim][node]
			for(int d = 0; d < 3; ++d) {
				w_1d[d][0] = 1 - dis_lb[d] / config::G_DX;
				w_1d[d][1] = dis_lb[d] / config::G_DX;
			}
			for(int i = 0; i < 2; ++i) {
				for(int j = 0; j < 2; ++j) {
					for(int k = 0; k < 2; ++k) {
						w[i][j][k] = w_1d[0][i] * w_1d[1][j] * w_1d[2][k];
					}
				}
			}
		}

		// 2. compute signed distance and normal
		for(int i = 0; i < 2; ++i) {
			for(int j = 0; j < 2; ++j) {
				for(int k = 0; k < 2; ++k) {
					sdis_res += w[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::G_BLOCKSIZE), ((g_cid[1] + j) / config::G_BLOCKSIZE), ((g_cid[2] + k) / config::G_BLOCKSIZE)).val(_0, (g_cid[0] + i) % config::G_BLOCKSIZE, (g_cid[1] + j) % config::G_BLOCKSIZE, (g_cid[2] + k) % config::G_BLOCKSIZE));
					normal[0] += w[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::G_BLOCKSIZE), ((g_cid[1] + j) / config::G_BLOCKSIZE), ((g_cid[2] + k) / config::G_BLOCKSIZE)).val(_1, (g_cid[0] + i) % config::G_BLOCKSIZE, (g_cid[1] + j) % config::G_BLOCKSIZE, (g_cid[2] + k) % config::G_BLOCKSIZE));
					normal[1] += w[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::G_BLOCKSIZE), ((g_cid[1] + j) / config::G_BLOCKSIZE), ((g_cid[2] + k) / config::G_BLOCKSIZE)).val(_2, (g_cid[0] + i) % config::G_BLOCKSIZE, (g_cid[1] + j) % config::G_BLOCKSIZE, (g_cid[2] + k) % config::G_BLOCKSIZE));
					normal[2] += w[i][j][k] * (this->ch(_0, ((g_cid[0] + i) / config::G_BLOCKSIZE), ((g_cid[1] + j) / config::G_BLOCKSIZE), ((g_cid[2] + k) / config::G_BLOCKSIZE)).val(_3, (g_cid[0] + i) % config::G_BLOCKSIZE, (g_cid[1] + j) % config::G_BLOCKSIZE, (g_cid[2] + k) % config::G_BLOCKSIZE));
				}
			}
		}
		normal /= sqrtf(normal.l2NormSqr());
		return sdis_res;
	}
	__forceinline__ __device__ bool query_sdf(vec3& normal, const vec3& x) {
		if(x[0] < config::G_BOUNDARY_CONDITION * config::G_DX * config::G_BLOCKSIZE || x[0] >= (GridDomain::range(_0) - config::G_BOUNDARY_CONDITION) * config::G_BLOCKSIZE * config::G_DX || x[1] < config::G_BOUNDARY_CONDITION * config::G_DX * config::G_BLOCKSIZE || x[1] >= (GridDomain::range(_1) - config::G_BOUNDARY_CONDITION) * config::G_BLOCKSIZE * config::G_DX || x[2] < config::G_BOUNDARY_CONDITION * config::G_DX * config::G_BLOCKSIZE || x[2] >= (GridDomain::range(_2) - config::G_BOUNDARY_CONDITION) * config::G_BLOCKSIZE * config::G_DX) {
			return false;
		}
		return get_signed_distance_and_normal(x, normal) <= 0.f;
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
	__forceinline__ __device__ void detect_and_resolve_collision(const ivec3 block_id, const ivec3 cell_id, float current_time, vec3& vel) {
		vec3 x_minustrans = (block_id * config::G_BLOCKSIZE + cell_id).cast<float>() * config::G_DX - (trans + trans_vel * current_time);
		// material space
		vec3 x;
		vec3x3 current_rot_mat = rot_mat;
		{
			vec3 x0 = x_minustrans * (1.f / (1.f + dsdt * current_time));

			vec3x3 rot_tmp	= SignedDistanceGrid::rot_angle_to_matrix(omega[0] * current_time, 0);
			vec3x3 prev_rot = current_rot_mat;
			matrix_matrix_multiplication_3d(prev_rot.data_arr(), rot_tmp.data_arr(), current_rot_mat.data_arr());

			rot_tmp	 = SignedDistanceGrid::rot_angle_to_matrix(omega[1] * current_time, 1);
			prev_rot = current_rot_mat;
			matrix_matrix_multiplication_3d(prev_rot.data_arr(), rot_tmp.data_arr(), current_rot_mat.data_arr());

			rot_tmp	 = SignedDistanceGrid::rot_angle_to_matrix(omega[2] * current_time, 2);
			prev_rot = current_rot_mat;
			matrix_matrix_multiplication_3d(prev_rot.data_arr(), rot_tmp.data_arr(), current_rot_mat.data_arr());
			mat_t_mul_vec_3d(x.data_arr(), current_rot_mat.data_arr(), x0.data_arr());
		}

		x = x * scale + trans;

		//< enforce BC if inside LS
		vec3 obj_normal;
		bool hit = query_sdf(obj_normal, x);//< material space normal
		if(hit) {
			///< calculate object velocity in deformation space
			vec3 v_object = SignedDistanceGrid::vec3_cross_vec3(omega, x_minustrans);
			v_object += x_minustrans * (dsdt / scale);
			{
				vec3 rot_v;
				matrix_vector_multiplication_3d(current_rot_mat.data_arr(), get_material_velocity(x).data_arr(), rot_v.data_arr());
				v_object += rot_v * scale + trans_vel;
			}
			vel -= v_object;

			/// STICKY
			if(type == BoundaryT::STICKY) {
				vel.set(0.0f);
				/// SLIP
			} else if(type == BoundaryT::SLIP) {
				{
					vec3 n;
					matrix_vector_multiplication_3d(current_rot_mat.data_arr(), obj_normal.data_arr(), n.data_arr());
					obj_normal = n;
				}
				float v_dot_n = obj_normal.dot(vel);
				vel -= (obj_normal * v_dot_n);
				if(friction > 0.0f) {
					if(v_dot_n < 0) {
						float vel_norm = sqrtf(vel.l2NormSqr());
						if(-v_dot_n * friction < vel_norm) {
							vel += (vel / vel_norm * (v_dot_n * friction));
						} else {
							vel.set(0.0f);
						}
					}
				}
			}
			/// STICKY
			else if(type == BoundaryT::SEPARATE) {
				if(obj_normal[0] == 0.0f && obj_normal[1] == 0.0f && obj_normal[2] == 0.0f) {
					vel.set(0.0f);
					return;
				}
				{
					vec3 n;
					matrix_vector_multiplication_3d(current_rot_mat.data_arr(), obj_normal.data_arr(), n.data_arr());
					obj_normal = n;
				}
				float v_dot_n = obj_normal.dot(vel);
				if(v_dot_n < 0) {
					vel -= (obj_normal * v_dot_n);
					if(friction != 0) {
						float vel_norm = sqrtf(vel.l2NormSqr());
						if(-v_dot_n * friction < vel_norm) {
							vel += (vel / vel_norm * (v_dot_n * friction));
						} else {
							vel.set(0.0f);
						}
					}
				}
			} else {
				//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print
				printf("[ERROR] Wrong Boundary Type!\n");
			}
			vel += v_object;
		}
	}
};
//NOLINTEND(readability-magic-numbers)

template<place_id Chn>
inline void fill_signed_distance_field(std::integral_constant<place_id, Chn> chn, const std::vector<float>& s_dis, Instance<signed_distance_field_>& host_data) {
	int inside_node_num = 0;
	for(auto bx = 0; bx < GridDomain::range(_0); ++bx) {
		for(auto by = 0; by < GridDomain::range(_1); ++by) {
			for(auto bz = 0; bz < GridDomain::range(_2); ++bz) {
				auto sdis_block = host_data.ch(_0, bx, by, bz);

				for(auto cx = 0; cx < config::G_BLOCKSIZE; ++cx) {
					auto i = bx * config::G_BLOCKSIZE + cx;

					for(auto cy = 0; cy < config::G_BLOCKSIZE; ++cy) {
						auto j = by * config::G_BLOCKSIZE + cy;

						for(auto cz = 0; cz < config::G_BLOCKSIZE; ++cz) {
							auto k							= bz * config::G_BLOCKSIZE + cz;
							auto idx						= (i * GridDomain::range(_1) * config::G_BLOCKSIZE * GridDomain::range(_2) * config::G_BLOCKSIZE) + (j * GridDomain::range(_2) * config::G_BLOCKSIZE) + k;
							sdis_block.val(chn, cx, cy, cz) = s_dis[idx];
							if(Chn == 0) {
								if(s_dis[idx] <= 0.0f) {
									inside_node_num++;
								}
							}
						}
					}
				}
			}
		}
	}
	if(Chn == 0) {
		fmt::print(
			"[Collision Object]\n\t[from saved signed_distance_field] "
			"Finish init signed distance buffer, inside node num = {}.\n",
			inside_node_num
		);
	} else {
		fmt::print(
			"[Collision Object]\n\t[from saved signed_distance_field] "
			"Finish init signed distance gradient [{}].\n",
			Chn
		);
	}
}

inline void init_from_signed_distance_file(const std::string& filename, vec<std::size_t, config::NUM_DIMENSIONS> resolution, Instance<signed_distance_field_>& host_data) {
	std::string file_addr = std::string(AssetDirPath) + "vdbSDF/";
	std::vector<float> sdisf(resolution.prod());
	auto read_file = [&](const std::string& suffix) {
		std::string file_path;
		file_path.append(file_addr).append(filename).append(suffix);
		FILE* fn;
		fopen_s(&fn, file_path.c_str(), "rb");
		std::size_t read_num = std::fread(static_cast<float*>(sdisf.data()), sizeof(float), sdisf.size(), fn);
		if(read_num != static_cast<size_t>(resolution.prod())) {
			std::cout << "Error in loading file [" << filename.c_str() << "]: read in " << read_num << " entries, should be " << resolution.prod() << std::endl;
			throw std::runtime_error("Error occured while loading file");
		}
		std::fclose(fn);
	};
	read_file("_sdf.bin");
	fill_signed_distance_field(_0, sdisf, host_data);
	read_file("_grad_0.bin");
	fill_signed_distance_field(_1, sdisf, host_data);
	read_file("_grad_1.bin");
	fill_signed_distance_field(_2, sdisf, host_data);
	read_file("_grad_2.bin");
	fill_signed_distance_field(_3, sdisf, host_data);
}

}// namespace mn

#endif