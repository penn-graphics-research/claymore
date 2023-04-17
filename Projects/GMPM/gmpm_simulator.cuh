#ifndef GMPM_SIMULATOR_CUH
#define GMPM_SIMULATOR_CUH
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/IO/ParticleIO.hpp>
#include <array>
#include <vector>

#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"

namespace mn {

struct GmpmSimulator {
	static constexpr float DEFAULT_DT	= 1e-4;
	static constexpr int DEFAULT_FPS	= 24;
	static constexpr int DEFAULT_FRAMES = 60;

	static constexpr size_t BIN_COUNT = 2;

	using streamIdx		 = Cuda::StreamIndex;
	using eventIdx		 = Cuda::EventIndex;
	using host_allocator = HeapAllocator;

	static_assert(std::is_same_v<GridBufferDomain::index_type, int>, "block index type is not int");

	struct DeviceAllocator {			   // hide the global one
		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			void* ret = nullptr;
			check_cuda_errors(cudaMalloc(&ret, bytes));
			return ret;
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) size;
			check_cuda_errors(cudaFree(p));
		}
	};

	struct Intermediates {
		void* base;

		int* d_tmp;
		int* active_block_marks;
		int* destinations;
		int* sources;
		int* binpbs;
		float* d_max_vel;
		//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
		void alloc(size_t max_block_cnt) {
			//NOLINTBEGIN(readability-magic-numbers) Magic numbers are variable count
			check_cuda_errors(cudaMalloc(&base, sizeof(int) * (max_block_cnt * 5 + 1)));

			d_tmp			   = static_cast<int*>(base);
			active_block_marks = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_cnt));
			destinations	   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_cnt * 2));
			sources			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_cnt * 3));
			binpbs			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_cnt * 4));
			d_max_vel		   = static_cast<float*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_cnt * 5));
			//NOLINTEND(readability-magic-numbers)
		}
		//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		void dealloc() const {
			cudaDeviceSynchronize();
			check_cuda_errors(cudaFree(base));
		}
		void resize(size_t max_block_cnt) {
			dealloc();
			alloc(max_block_cnt);
		}
	};

	///
	int gpuid;
	int nframes;
	int fps;
	/// animation runtime settings
	float dt;
	float next_dt;
	float dt_default;
	float cur_time;
	float max_vel;
	uint64_t cur_frame;
	uint64_t cur_step;
	/// data on device, double buffering
	std::vector<GridBuffer> grid_blocks									= {};
	std::array<std::vector<particle_buffer_t>, BIN_COUNT> particle_bins = {};
	std::vector<Partition<1>> partitions								= {};///< with halo info
	std::vector<ParticleArray> particles								= {};

	Intermediates tmps;

	/// data on host
	char rollid;
	std::size_t cur_num_active_blocks;
	std::vector<std::size_t> cur_num_active_bins	= {};
	std::array<std::size_t, BIN_COUNT> checked_cnts = {};
	std::vector<std::size_t> checked_bin_cnts		= {};
	float max_vels;
	int pbcnt;
	int nbcnt;
	int ebcnt;///< num blocks
	std::vector<int> bincnt										 = {};
	std::vector<uint32_t> pcnt									 = {};///< num particles
	std::vector<std::array<float, config::NUM_DIMENSIONS>> model = {};
	std::vector<vec3> vel0										 = {};

	explicit GmpmSimulator(int gpu = 0, float dt = DEFAULT_DT, int fps = DEFAULT_FPS, int frames = DEFAULT_FRAMES)
		: gpuid(gpu)
		, dt_default(dt)
		, cur_time(0.0f)
		, rollid(0)
		, cur_frame(0)
		, cur_step(0)
		, fps(fps)
		, nframes(frames)
		, dt()
		, next_dt()
		, max_vel()
		, tmps()
		, cur_num_active_blocks()
		, max_vels()
		, pbcnt()
		, nbcnt()
		, ebcnt() {
		// data
		initialize();
	}
	~GmpmSimulator() = default;

	//TODO: Maybe implement?
	GmpmSimulator(GmpmSimulator& other)				= delete;
	GmpmSimulator(GmpmSimulator&& other)			= delete;
	GmpmSimulator& operator=(GmpmSimulator& other)	= delete;
	GmpmSimulator& operator=(GmpmSimulator&& other) = delete;

	void initialize() {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		cu_dev.set_context();
		tmps.alloc(config::G_MAX_ACTIVE_BLOCK);
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			grid_blocks.emplace_back(DeviceAllocator {});
			partitions.emplace_back(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			checked_cnts[copyid] = 0;
		}
		cu_dev.syncStream<streamIdx::COMPUTE>();
		cur_num_active_blocks = config::G_MAX_ACTIVE_BLOCK;
	}

	template<MaterialE M>
	void init_model(const std::vector<std::array<float, config::NUM_DIMENSIONS>>& model, const mn::vec<float, config::NUM_DIMENSIONS>& v0) {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
			particle_bins[copyid].emplace_back(ParticleBuffer<M>(DeviceAllocator {}, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));
			match(particle_bins[copyid].back())([&](auto& pb) {
				pb.reserve_buckets(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			});
		}
		vel0.emplace_back();
		for(int i = 0; i < config::NUM_DIMENSIONS; ++i) {
			vel0.back()[i] = v0[i];
		}

		particles.emplace_back(ParticleArray {spawn<particle_array_, orphan_signature>(DeviceAllocator {}, sizeof(float) * config::NUM_DIMENSIONS * model.size())});
		cur_num_active_bins.emplace_back(config::G_MAX_PARTICLE_BIN);
		bincnt.emplace_back(0);
		checked_bin_cnts.emplace_back(0);

		pcnt.emplace_back(static_cast<unsigned int>(model.size()));//NOTE: Explicic narrowing cast
		fmt::print("init {}-th model with {} particles\n", particle_bins[0].size() - 1, pcnt.back());
		cudaMemcpyAsync(static_cast<void*>(&particles.back().val_1d(_0, 0)), model.data(), sizeof(std::array<float, config::NUM_DIMENSIONS>) * model.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cu_dev.syncStream<streamIdx::COMPUTE>();

		std::string fn = std::string {"model"} + "_id[" + std::to_string(particle_bins[0].size() - 1) + "]_frame[0].bgeo";
		IO::insert_job([fn, model]() {
			write_partio<float, config::NUM_DIMENSIONS>(fn, model);
		});
		IO::flush();
	}

	void update_fr_parameters(float rho, float vol, float ym, float pr) {
		match(particle_bins[0].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& pb) {
				pb.update_parameters(rho, vol, ym, pr);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& pb) {
				pb.update_parameters(rho, vol, ym, pr);
			}
		);
	}

	void update_j_fluid_parameters(float rho, float vol, float bulk, float gamma, float visco) {
		match(particle_bins[0].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& pb) {
				pb.update_parameters(rho, vol, bulk, gamma, visco);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& pb) {
				pb.update_parameters(rho, vol, bulk, gamma, visco);
			}
		);
	}

	void update_nacc_parameters(float rho, float vol, float ym, float pr, float beta, float xi) {
		match(particle_bins[0].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::NACC>& pb) {
				pb.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& pb) {},
			[&](ParticleBuffer<MaterialE::NACC>& pb) {
				pb.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
	}

	template<typename CudaContext>
	void excl_scan(int cnt, int const* const in, int* out, CudaContext& cu_dev) {
//TODO:Not sure, what it does. Maybe remove or create names control macro for activation
#if 1//NOLINT(readability-avoid-unconditional-preprocessor-if)
		auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
		thrust::exclusive_scan(policy, get_device_ptr(in), get_device_ptr(in) + cnt, get_device_ptr(out));
#else
		std::size_t temp_storage_bytes = 0;
		auto plus_op				   = [] __device__(const int& a, const int& b) {
			  return a + b;
		};
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, in, out, plus_op, 0, cnt, cu_dev.stream_compute()));
		void* d_tmp = tmps[cu_dev.get_dev_id()].d_tmp;
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, in, out, plus_op, 0, cnt, cu_dev.stream_compute()));
#endif
	}

	float get_mass(int id = 0) {
		return match(particle_bins[rollid][id])([&](const auto& particle_buffer) {
			return particle_buffer.mass;
		});
	}

	[[nodiscard]] int get_model_cnt() const noexcept {
		return static_cast<int>(particle_bins[0].size());//NOTE: Explicit narrowing cast (But we should not have that much models anyway.)
	}

	void check_capacity() {
		//TODO: Is that right? Maybe create extra parameter for this?
		//NOLINTBEGIN(readability-magic-numbers) Magic numbers are resize thresholds?
		if(ebcnt > cur_num_active_blocks * config::NUM_DIMENSIONS / 4 && checked_cnts[0] == 0) {
			cur_num_active_blocks = cur_num_active_blocks * config::NUM_DIMENSIONS / 2;
			checked_cnts[0]		  = 2;
			fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", ebcnt, cur_num_active_blocks);
		}
		for(int i = 0; i < get_model_cnt(); ++i) {
			if(bincnt[i] > cur_num_active_bins[i] * config::NUM_DIMENSIONS / 4 && checked_bin_cnts[i] == 0) {
				cur_num_active_bins[i] = cur_num_active_bins[i] * config::NUM_DIMENSIONS / 2;
				checked_bin_cnts[i]	   = 2;
				fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincnt[i], cur_num_active_bins[i]);
			}
		}
		//NOLINTEND(readability-magic-numbers)
	}

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers) Current c++ version does not yet support std::span
	void main_loop() {
		/// initial
		const float initial_next_time = 1.0f / static_cast<float>(fps);
		{
			float max_vel = 0.f;
			for(int i = 0; i < get_model_cnt(); ++i) {
				const float vel_norm = std::sqrt(vel0[i].l2NormSqr());
				if(vel_norm > max_vel) {
					max_vel = vel_norm;
				}
			}

			dt = compute_dt(max_vel, cur_time, initial_next_time, dt_default);
		}
		fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", cur_time, dt, initial_next_time, dt_default);
		initial_setup();
		cur_time = dt;
		for(cur_frame = 1; cur_frame <= nframes; ++cur_frame) {
			//FIXME: Count of time steps might decrease with further framecount, cause floating point precision decreases
			const float next_time = static_cast<float>(cur_frame) / static_cast<float>(fps);
			for(; cur_time < next_time; cur_time += dt, cur_step++) {
				/// max grid vel
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					/// check capacity
					check_capacity();
					float* d_max_vel = tmps.d_max_vel;
					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();
					check_cuda_errors(cudaMemsetAsync(d_max_vel, 0, sizeof(float), cu_dev.stream_compute()));
					cu_dev.compute_launch({(nbcnt + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * 32, config::G_NUM_WARPS_PER_CUDA_BLOCK}, update_grid_velocity_query_max, static_cast<uint32_t>(nbcnt), grid_blocks[0], partitions[rollid], dt, d_max_vel);
					check_cuda_errors(cudaMemcpyAsync(&max_vels, d_max_vel, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute()));
					timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query", gpuid, cur_frame, cur_step));
				}

				/// host: compute maxvel & next dt
				float max_vel = max_vels;
				// if (max_vels > max_vel)
				//  max_vel = max_vels[id];
				max_vel = std::sqrt(max_vel);// this is a bug, should insert this line
				next_dt = compute_dt(max_vel, cur_time, next_time, dt_default);
				fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}, max_vel: {}\n", cur_time, next_dt, next_time, dt_default, max_vel);

				/// g2p2g
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					/// check capacity
					for(int i = 0; i < get_model_cnt(); ++i) {
						if(checked_bin_cnts[i] > 0) {
							match(particle_bins[rollid ^ 1][i])([this, &i](auto& pb) {
								pb.resize(DeviceAllocator {}, cur_num_active_bins[i]);
							});
							checked_bin_cnts[i]--;
						}
					}

					timer.tick();
					// grid
					grid_blocks[1].reset(nbcnt, cu_dev);
					// adv map
					for(int i = 0; i < get_model_cnt(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& pb) {
							check_cuda_errors(cudaMemsetAsync(pb.ppcs, 0, sizeof(int) * ebcnt * config::G_BLOCKVOLUME, cu_dev.stream_compute()));
						});
						// g2p2g
						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& pb) {
							cu_dev.compute_launch({pbcnt, 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt, next_dt, pb, get<typename std::decay_t<decltype(pb)>>(particle_bins[rollid ^ 1][i]), partitions[rollid ^ 1], partitions[rollid], grid_blocks[0], grid_blocks[1]);
						});
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", gpuid, cur_frame, cur_step));
					if(checked_cnts[0] > 0) {
						partitions[rollid ^ 1].resize_partition(DeviceAllocator {}, cur_num_active_blocks);
						for(int i = 0; i < get_model_cnt(); ++i) {
							match(particle_bins[rollid][i])([this, &cu_dev](auto& pb) {
								pb.reserve_buckets(DeviceAllocator {}, cur_num_active_blocks);
							});
						}
						checked_cnts[0]--;
					}
				}

				/// update partition
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();
					/// mark particle blocks
					for(int i = 0; i < get_model_cnt(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& pb) {
							check_cuda_errors(cudaMemsetAsync(pb.ppbs, 0, sizeof(int) * (ebcnt + 1), cu_dev.stream_compute()));
							cu_dev.compute_launch({ebcnt, config::G_BLOCKVOLUME}, cell_bucket_to_block, pb.ppcs, pb.cellbuckets, pb.ppbs, pb.blockbuckets);
							// partitions[rollid].buildParticleBuckets(cu_dev, ebcnt);
						});
					}

					int* active_block_marks = tmps.active_block_marks;
					int* destinations		= tmps.destinations;
					int* sources			= tmps.sources;
					check_cuda_errors(cudaMemsetAsync(active_block_marks, 0, sizeof(int) * nbcnt, cu_dev.stream_compute()));
					/// mark grid blocks
					cu_dev.compute_launch({(nbcnt * config::G_BLOCKVOLUME + 127) / 128, 128}, mark_active_grid_blocks, static_cast<uint32_t>(nbcnt), grid_blocks[1], active_block_marks);
					/// mark particle blocks
					check_cuda_errors(cudaMemsetAsync(sources, 0, sizeof(int) * (ebcnt + 1), cu_dev.stream_compute()));
					for(int i = 0; i < get_model_cnt(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev, &sources](auto& pb) {
							cu_dev.compute_launch({(ebcnt + 1 + 127) / 128, 128}, mark_active_particle_blocks, ebcnt + 1, pb.ppbs, sources);
						});
					}
					excl_scan(ebcnt + 1, sources, destinations, cu_dev);
					/// building new partition
					// block count
					check_cuda_errors(cudaMemcpyAsync(partitions[rollid ^ 1].cnt, destinations + ebcnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					check_cuda_errors(cudaMemcpyAsync(&pbcnt, destinations + ebcnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.compute_launch({(ebcnt + 255) / 256, 256}, exclusive_scan_inverse, ebcnt, static_cast<const int*>(destinations), sources);
					// indextable, activeKeys, ppb, buckets
					partitions[rollid ^ 1].reset_table(cu_dev.stream_compute());
					cu_dev.syncStream<streamIdx::COMPUTE>();
					cu_dev.compute_launch({(pbcnt + 127) / 128, 128}, update_partition, static_cast<uint32_t>(pbcnt), static_cast<const int*>(sources), partitions[rollid], partitions[rollid ^ 1]);
					for(int i = 0; i < get_model_cnt(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev, &sources, &i](auto& pb) {
							auto& next_pb = get<typename std::decay_t<decltype(pb)>>(particle_bins[rollid][i]);
							cu_dev.compute_launch({pbcnt, 128}, update_buckets, static_cast<uint32_t>(pbcnt), static_cast<const int*>(sources), pb, next_pb);
						});
					}
					// binsts
					int* binpbs = tmps.binpbs;
					for(int i = 0; i < get_model_cnt(); ++i) {
						match(particle_bins[rollid][i])([this, &cu_dev, &binpbs, &i](auto& pb) {
							cu_dev.compute_launch({(pbcnt + 1 + 127) / 128, 128}, compute_bin_capacity, pbcnt + 1, static_cast<const int*>(pb.ppbs), binpbs);
							excl_scan(pbcnt + 1, binpbs, pb.binsts, cu_dev);
							check_cuda_errors(cudaMemcpyAsync(&bincnt[i], pb.binsts + pbcnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
							cu_dev.syncStream<streamIdx::COMPUTE>();
						});
					}
					timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition", gpuid, cur_frame, cur_step));

					/// neighboring blocks
					timer.tick();
					cu_dev.compute_launch({(pbcnt + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(pbcnt), partitions[rollid ^ 1]);
					auto prev_nbcnt = nbcnt;
					check_cuda_errors(cudaMemcpyAsync(&nbcnt, partitions[rollid ^ 1].cnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_grid", gpuid, cur_frame, cur_step));

					/// check capacity
					if(checked_cnts[0] > 0) {
						grid_blocks[0].resize(DeviceAllocator {}, cur_num_active_blocks);
					}
					/// rearrange grid blocks
					timer.tick();
					grid_blocks[0].reset(ebcnt, cu_dev);
					cu_dev.compute_launch({prev_nbcnt, config::G_BLOCKVOLUME}, copy_selected_grid_blocks, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[rollid ^ 1], static_cast<const int*>(active_block_marks), grid_blocks[1], grid_blocks[0]);
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks", gpuid, cur_frame, cur_step));
					/// check capacity
					if(checked_cnts[0] > 0) {
						grid_blocks[1].resize(DeviceAllocator {}, cur_num_active_blocks);
						tmps.resize(cur_num_active_blocks);
					}
				}

				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();
					/// exterior blocks
					cu_dev.compute_launch({(pbcnt + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(pbcnt), partitions[rollid ^ 1]);
					check_cuda_errors(cudaMemcpyAsync(&ebcnt, partitions[rollid ^ 1].cnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();

					fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, pbcnt, nbcnt, ebcnt, cur_num_active_blocks);
					for(int i = 0; i < get_model_cnt(); ++i) {
						fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincnt[i], cur_num_active_bins[i]);
					}
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_particles", gpuid, cur_frame, cur_step));
				}
				rollid ^= 1;
				dt = next_dt;
			}
			IO::flush();
			output_model();
			fmt::print(
				fmt::emphasis::bold | fg(fmt::color::red),
				"-----------------------------------------------------------"
				"-----\n"
			);
		}
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(readability-magic-numbers)
	void output_model() {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		CudaTimer timer {cu_dev.stream_compute()};
		timer.tick();
		for(int i = 0; i < get_model_cnt(); ++i) {
			int parcnt	  = 0;
			int* d_parcnt = static_cast<int*>(cu_dev.borrow(sizeof(int)));
			check_cuda_errors(cudaMemsetAsync(d_parcnt, 0, sizeof(int), cu_dev.stream_compute()));
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_parcnt](const auto& pb) {
				cu_dev.compute_launch({pbcnt, 128}, retrieve_particle_buffer, partitions[rollid], partitions[rollid ^ 1], pb, get<typename std::decay_t<decltype(pb)>>(particle_bins[rollid ^ 1][i]), particles[i], d_parcnt);
			});
			check_cuda_errors(cudaMemcpyAsync(&parcnt, d_parcnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			fmt::print(fg(fmt::color::red), "total number of particles {}\n", parcnt);
			model.resize(parcnt);
			check_cuda_errors(cudaMemcpyAsync(model.data(), static_cast<void*>(&particles[i].val_1d(_0, 0)), sizeof(std::array<float, 3>) * (parcnt), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			std::string fn = std::string {"model"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";
			IO::insert_job([fn, m = model]() {
				write_partio<float, config::NUM_DIMENSIONS>(fn, m);
			});
		}
		timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", gpuid, cur_frame, cur_step));
	}
	//NOLINTEND(readability-magic-numbers)

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)Current c++ version does not yet support std::span
	void initial_setup() {
		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};

			timer.tick();
			for(int i = 0; i < get_model_cnt(); ++i) {
				cu_dev.compute_launch({(pcnt[i] + 255) / 256, 256}, activate_blocks, pcnt[i], particles[i], partitions[rollid ^ 1]);
			}

			check_cuda_errors(cudaMemcpyAsync(&pbcnt, partitions[rollid ^ 1].cnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			timer.tock(fmt::format("GPU[{}] step {} init_table", gpuid, cur_step));

			timer.tick();
			cu_dev.reset_mem();
			// particle block
			for(int i = 0; i < get_model_cnt(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& pb) {
					cu_dev.compute_launch({(pcnt[i] + 255) / 256, 256}, build_particle_cell_buckets, pcnt[i], particles[i], pb, partitions[rollid ^ 1]);
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();

			// bucket, binsts
			for(int i = 0; i < get_model_cnt(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev](auto& pb) {
					check_cuda_errors(cudaMemsetAsync(pb.ppbs, 0, sizeof(int) * (pbcnt + 1), cu_dev.stream_compute()));
					cu_dev.compute_launch({pbcnt, config::G_BLOCKVOLUME}, cell_bucket_to_block, pb.ppcs, pb.cellbuckets, pb.ppbs, pb.blockbuckets);
					// partitions[rollid ^ 1].buildParticleBuckets(cu_dev, pbcnt);
				});
			}
			int* binpbs = tmps.binpbs;
			for(int i = 0; i < get_model_cnt(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &binpbs, &i](auto& pb) {
					cu_dev.compute_launch({(pbcnt + 1 + 127) / 128, 128}, compute_bin_capacity, pbcnt + 1, static_cast<const int*>(pb.ppbs), binpbs);
					excl_scan(pbcnt + 1, binpbs, pb.binsts, cu_dev);
					check_cuda_errors(cudaMemcpyAsync(&bincnt[i], pb.binsts + pbcnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					cu_dev.compute_launch({pbcnt, 128}, array_to_buffer, particles[i], pb);
				});
			}
			// grid block
			cu_dev.compute_launch({(pbcnt + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(pbcnt), partitions[rollid ^ 1]);
			check_cuda_errors(cudaMemcpyAsync(&nbcnt, partitions[rollid ^ 1].cnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			cu_dev.compute_launch({(pbcnt + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(pbcnt), partitions[rollid ^ 1]);
			check_cuda_errors(cudaMemcpyAsync(&ebcnt, partitions[rollid ^ 1].cnt, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_partition", gpuid, cur_step));

			fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, pbcnt, nbcnt, ebcnt, cur_num_active_blocks);
			for(int i = 0; i < get_model_cnt(); ++i) {
				fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincnt[i], cur_num_active_bins[i]);
			}
		}

		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};

			partitions[rollid ^ 1].copy_to(partitions[rollid], ebcnt, cu_dev.stream_compute());
			check_cuda_errors(cudaMemcpyAsync(partitions[rollid].active_keys, partitions[rollid ^ 1].active_keys, sizeof(ivec3) * ebcnt, cudaMemcpyDefault, cu_dev.stream_compute()));
			for(int i = 0; i < get_model_cnt(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& pb) {
					// binsts, ppbs
					pb.copy_to(get<typename std::decay_t<decltype(pb)>>(particle_bins[rollid ^ 1][i]), pbcnt, cu_dev.stream_compute());
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();

			timer.tick();
			grid_blocks[0].reset(nbcnt, cu_dev);
			for(int i = 0; i < get_model_cnt(); ++i) {
				cu_dev.compute_launch({(pcnt[i] + 255) / 256, 256}, rasterize, pcnt[i], particles[i], grid_blocks[0], partitions[rollid], dt, get_mass(i), vel0[i]);
				match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& pb) {
					cu_dev.compute_launch({pbcnt, 128}, init_adv_bucket, static_cast<const int*>(pb.ppbs), pb.blockbuckets);
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_grid", gpuid, cur_step));
		}
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)
};

}// namespace mn

#endif