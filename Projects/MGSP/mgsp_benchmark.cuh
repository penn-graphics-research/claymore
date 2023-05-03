#ifndef MGSP_BENCHMARK_CUH
#define MGSP_BENCHMARK_CUH
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

#include "boundary_condition.cuh"
#include "grid_buffer.cuh"
#include "halo_buffer.cuh"
#include "halo_kernels.cuh"
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"

namespace mn {

struct MgspBenchmark {
	static constexpr float DEFAULT_DT = 1e-4;
	static constexpr int DEFAULT_FPS  = 24;

	static constexpr size_t BIN_COUNT = 2;

	using streamIdx		 = Cuda::StreamIndex;
	using eventIdx		 = Cuda::EventIndex;
	using host_allocator = HeapAllocator;

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

	struct TempAllocator {
		int did;

		explicit TempAllocator(int did)
			: did {did} {}

		void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			return Cuda::ref_cuda_context(did).borrow(bytes);
		}

		void deallocate(void* p, std::size_t size) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
			(void) p;
			(void) size;
		}
	};

	struct Intermediates {
		void* base;

		int* d_tmp;
		int* active_block_marks;
		int* destinations;
		int* sources;
		int* bin_sizes;
		float* d_max_vel;
		//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) Current c++ version does not yet support std::span
		void alloc(size_t max_block_count) {
			//NOLINTBEGIN(readability-magic-numbers) Magic numbers are variable count
			check_cuda_errors(cudaMalloc(&base, sizeof(int) * (max_block_count * 5 + 1)));

			d_tmp			   = static_cast<int*>(base);
			active_block_marks = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count));
			destinations	   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 2));
			sources			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 3));
			bin_sizes		   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 4));
			d_max_vel		   = static_cast<float*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 5));
			//NOLINTEND(readability-magic-numbers)
		}
		//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		void dealloc() const {
			cudaDeviceSynchronize();
			check_cuda_errors(cudaFree(base));
		}
		void resize(size_t max_block_count) {
			dealloc();
			alloc(max_block_count);
		}
	};

	///
	/// animation runtime settings
	float dt;
	float next_dt;
	float dt_default;
	float cur_time;
	float max_vel;
	uint64_t cur_frame;
	uint64_t cur_step;
	uint64_t fps;
	/// data on device, double buffering
	std::vector<optional<SignedDistanceGrid>> collision_objs;
	std::array<std::vector<GridBuffer>, BIN_COUNT> grid_blocks;
	std::array<std::vector<particle_buffer_t>, BIN_COUNT> particle_bins;
	std::array<std::vector<Partition<1>>, BIN_COUNT> partitions;///< with halo info
	std::vector<HaloGridBlocks> input_halo_grid_blocks;
	std::vector<HaloGridBlocks> output_halo_grid_blocks;
	// std::vector<HaloParticleBlocks> inputHaloParticleBlocks,
	// outputHaloParticleBlocks;
	vec<ParticleArray, config::G_DEVICE_COUNT> particles;

	std::array<Intermediates, config::G_DEVICE_COUNT> tmps = {};

	// halo data
	vec<ivec3*, config::G_DEVICE_COUNT, config::G_DEVICE_COUNT> halo_block_ids;

	/// data on host
	static_assert(std::is_same_v<GridBufferDomain::index_type, int>, "block index type is not int");
	char rollid;
	std::array<std::size_t, config::G_DEVICE_COUNT> cur_num_active_blocks				  = {};
	std::array<std::size_t, config::G_DEVICE_COUNT> cur_num_active_bins					  = {};
	std::array<std::array<std::size_t, BIN_COUNT>, config::G_DEVICE_COUNT> checked_counts = {};
	vec<float, config::G_DEVICE_COUNT> max_vels;
	vec<int, config::G_DEVICE_COUNT> partition_block_count;
	vec<int, config::G_DEVICE_COUNT> nbcount;
	vec<int, config::G_DEVICE_COUNT> exterior_block_count;
	vec<int, config::G_DEVICE_COUNT> bincount;			  ///< num blocks
	vec<uint32_t, config::G_DEVICE_COUNT> particle_counts;///< num particles
	std::array<std::vector<float>, config::G_DEVICE_COUNT + 1> durations;
	std::array<std::vector<std::array<float, config::NUM_DIMENSIONS>>, config::G_DEVICE_COUNT> models;
	Instance<signed_distance_field_> host_data;

	/// control
	bool b_running;
	std::array<ThreadsafeQueue<std::function<void(int)>>, config::G_DEVICE_COUNT> jobs;
	std::array<std::thread, config::G_DEVICE_COUNT> ths;///< thread is not trivial
	std::mutex mut_slave;
	std::mutex mut_ctrl;
	std::condition_variable cv_slave;
	std::condition_variable cv_ctrl;
	std::atomic_uint idle_count {0};

	/// computations per substep
	std::vector<std::function<void(int)>> init_tasks;
	std::vector<std::function<void(int)>> loop_tasks;

	MgspBenchmark()
		: dt()
		, next_dt()
		, dt_default(DEFAULT_DT)
		, cur_time(0.f)
		, max_vel()
		, rollid(0)
		, cur_frame(0)
		, cur_step(0)
		, fps(DEFAULT_FPS)
		, host_data()
		, b_running(true) {
		// data
		host_data = spawn<signed_distance_field_, orphan_signature>(host_allocator {});
		collision_objs.resize(config::G_DEVICE_COUNT);
		init_particles<0>();
		fmt::print(
			"{} -vs- {}\n",
			match(particle_bins[0][0])([&](auto& particle_buffer) {
				return particle_buffer.size;
			}),
			match(particle_bins[0][1])([&](auto& particle_buffer) {
				return particle_buffer.size;
			})
		);
		// tasks
		for(int did = 0; did < config::G_DEVICE_COUNT; ++did) {
			ths[did] = std::thread(
				[this](int did) {
					this->gpu_worker(did);
				},
				did
			);
		}
	}

	~MgspBenchmark() {
		auto is_empty = [this]() {
			for(int did = 0; did < config::G_DEVICE_COUNT; ++did) {
				if(!jobs[did].empty()) {
					return false;
				}
			}
			return true;
		};
		do {
			cv_slave.notify_all();
		} while(!is_empty());
		b_running = false;
		for(auto& th: ths) {
			th.join();
		}
	}

	//TODO: Maybe implement?
	MgspBenchmark(MgspBenchmark& other)				= delete;
	MgspBenchmark(MgspBenchmark&& other)			= delete;
	MgspBenchmark& operator=(MgspBenchmark& other)	= delete;
	MgspBenchmark& operator=(MgspBenchmark&& other) = delete;

	template<std::size_t I>
	void init_particles() {
		auto& cu_dev = Cuda::ref_cuda_context(I);
		cu_dev.set_context();
		tmps[I].alloc(config::G_MAX_ACTIVE_BLOCK);
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			grid_blocks[copyid].emplace_back(DeviceAllocator {});
			particle_bins[copyid].emplace_back(ParticleBuffer<config::get_material_type(I)> {DeviceAllocator {}});
			partitions[copyid].emplace_back(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
		}
		cu_dev.syncStream<streamIdx::COMPUTE>();
		input_halo_grid_blocks.emplace_back(config::G_DEVICE_COUNT);
		output_halo_grid_blocks.emplace_back(config::G_DEVICE_COUNT);
		particles[I]			 = static_cast<ParticleArray>(spawn<particle_array_, orphan_signature>(DeviceAllocator {}));
		checked_counts[I][0]	 = 0;
		checked_counts[I][1]	 = 0;
		cur_num_active_blocks[I] = config::G_MAX_ACTIVE_BLOCK;
		cur_num_active_bins[I]	 = config::G_MAX_PARTICLE_BIN;
		/// tail-recursion optimization
		if constexpr(I + 1 < config::G_DEVICE_COUNT) {
			init_particles<I + 1>();
		}
	}

	void init_model(int devid, const std::vector<std::array<float, config::NUM_DIMENSIONS>>& model) {
		auto& cu_dev = Cuda::ref_cuda_context(devid);
		cu_dev.set_context();
		particle_counts[devid] = model.size();
		fmt::print("init model[{}] with {} particles\n", devid, particle_counts[devid]);
		cudaMemcpyAsync(static_cast<void*>(&particles[devid].val_1d(_0, 0)), model.data(), sizeof(std::array<float, config::NUM_DIMENSIONS>) * model.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cu_dev.syncStream<streamIdx::COMPUTE>();

		std::string fn = std::string {"model"} + "_dev[" + std::to_string(devid) + "]_frame[0].bgeo";
		IO::insert_job([fn, model]() {
			write_partio<float, config::NUM_DIMENSIONS>(fn, model);
		});
		IO::flush();
	}

	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(readability-magic-numbers)
	void init_boundary(const std::string& fn) {
		init_from_signed_distance_file(fn, vec<std::size_t, config::NUM_DIMENSIONS> {static_cast<size_t>(1024), static_cast<size_t>(1024), static_cast<size_t>(512)}, host_data);
		for(int did = 0; did < config::G_DEVICE_COUNT; ++did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			cu_dev.set_context();
			collision_objs[did] = SignedDistanceGrid {DeviceAllocator {}};
			collision_objs[did]->init(host_data, cu_dev.stream_compute());
			cu_dev.syncStream<streamIdx::COMPUTE>();
		}
	}
	//NOLINTEND(readability-magic-numbers)

	template<typename CudaContext>
	void exclusive_scan(int count, int const* const in, int* out, CudaContext& cu_dev) {
//TODO:Not sure, what it does. Maybe remove or create names control macro for activation
#if 1//NOLINT(readability-avoid-unconditional-preprocessor-if)
		auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(cu_dev.stream_compute()));
		thrust::exclusive_scan(policy, get_device_ptr(in), get_device_ptr(in) + count, get_device_ptr(out));
#else
		std::size_t temp_storage_bytes = 0;
		auto plus_op				   = [] __device__(const int& a, const int& b) {
			  return a + b;
		};
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
		void* d_tmp = tmps[cu_dev.get_dev_id()].d_tmp;
		check_cuda_errors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, in, out, plus_op, 0, count, cu_dev.stream_compute()));
#endif
	}

	float get_mass(int did) {
		return match(particle_bins[rollid][did])([&](const auto& particle_buffer) {
			return particle_buffer.mass;
		});
	}

	void check_capacity(int did) {
		//TODO: Is that right? Maybe create extra parameter for this?
		//NOLINTBEGIN(readability-magic-numbers) Magic numbers are resize thresholds?
		if(exterior_block_count[did] > cur_num_active_blocks[did] * config::NUM_DIMENSIONS / 4 && checked_counts[did][0] == 0) {
			cur_num_active_blocks[did] = cur_num_active_blocks[did] * config::NUM_DIMENSIONS / 2;
			checked_counts[did][0]	   = 2;
			fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", exterior_block_count[did], cur_num_active_blocks[did]);
		}
		if(bincount[did] > cur_num_active_bins[did] * config::NUM_DIMENSIONS / 4 && checked_counts[did][1] == 0) {
			cur_num_active_bins[did] = cur_num_active_bins[did] * config::NUM_DIMENSIONS / 2;
			checked_counts[did][1]	 = 2;
			fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincount[did], cur_num_active_bins[did]);
		}
		//NOLINTEND(readability-magic-numbers)
	}

	/// thread local ctrl flow
	void gpu_worker(int did) {
		auto wait = [did, this]() {
			std::unique_lock<std::mutex> lk {this->mut_slave};
			this->cv_slave.wait(lk, [did, this]() {
				return !this->b_running || !this->jobs[did].empty();
			});
		};
		auto signal = [this]() {
			std::unique_lock<std::mutex> lk {this->mut_ctrl};
			this->idle_count.fetch_add(1);
			lk.unlock();
			this->cv_ctrl.notify_one();
		};
		auto& cu_dev = Cuda::ref_cuda_context(did);
		cu_dev.set_context();
		fmt::print(fg(fmt::color::light_blue), "{}-th gpu worker operates on GPU {}\n", did, cu_dev.get_dev_id());
		while(this->b_running) {
			wait();
			auto job = this->jobs[did].try_pop();
			if(job) {
				(*job)(did);
			}
			signal();
		}
		fmt::print(fg(fmt::color::light_blue), "{}-th gpu worker exits\n", did);
	}

	void sync() {
		std::unique_lock<std::mutex> lk {mut_ctrl};
		cv_ctrl.wait(lk, [this]() {
			return this->idle_count == config::G_DEVICE_COUNT;
		});
		fmt::print(
			fmt::emphasis::bold,
			"-----------------------------------------------------------"
			"-----\n"
		);
	}

	void issue(const std::function<void(int)>& job) {
		std::unique_lock<std::mutex> lk {mut_slave};
		for(int did = 0; did < config::G_DEVICE_COUNT; ++did) {
			jobs[did].push(job);
		}
		idle_count = 0;
		lk.unlock();
		cv_slave.notify_all();
	}

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers) Current c++ version does not yet support std::span
	void main_loop() {
		/// initial
		const float initial_next_time = 1.0f / static_cast<float>(fps);
		dt							  = compute_dt(0.0f, cur_time, initial_next_time, dt_default);
		fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", cur_time, dt, initial_next_time, dt_default);
		initial_setup();
		cur_time = dt;
		for(cur_frame = 1; cur_frame <= config::G_TOTAL_FRAME_COUNT; ++cur_frame) {
			const float next_time = static_cast<float>(cur_frame) / static_cast<float>(fps);
			for(; cur_time < next_time; cur_time += dt, cur_step++) {
				/// max grid vel
				issue([this](int did) {
					auto& cu_dev = Cuda::ref_cuda_context(did);
					/// check capacity
					check_capacity(did);
					float* d_max_vel = tmps[did].d_max_vel;
					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();
					check_cuda_errors(cudaMemsetAsync(d_max_vel, 0, sizeof(float), cu_dev.stream_compute()));
					if(collision_objs[did]) {
						cu_dev.compute_launch(
							{(nbcount[did] + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * 32, config::G_NUM_WARPS_PER_CUDA_BLOCK},
							update_grid_velocity_query_max,
							static_cast<uint32_t>(nbcount[did]),
							// grid_blocks[0][did], partitions[rollid][did], dt, d_max_vel);
							grid_blocks[0][did],
							partitions[rollid][did],
							dt,
							static_cast<const SignedDistanceGrid>(*collision_objs[did]),
							d_max_vel
						);
					} else {
						cu_dev.compute_launch(
							{(nbcount[did] + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * 32, config::G_NUM_WARPS_PER_CUDA_BLOCK},
							update_grid_velocity_query_max,
							static_cast<uint32_t>(nbcount[did]),
							// grid_blocks[0][did], partitions[rollid][did], dt, d_max_vel);
							grid_blocks[0][did],
							partitions[rollid][did],
							dt,
							d_max_vel
						);
					}
					check_cuda_errors(cudaMemcpyAsync(&max_vels[did], d_max_vel, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute()));
					timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query", did, cur_frame, cur_step));
				});
				sync();

				/// host: compute maxvel & next dt
				float max_vel = 0.f;
				for(int did = 0; did < config::G_DEVICE_COUNT; ++did) {
					if(max_vels[did] > max_vel) {
						max_vel = max_vels[did];
					}
				}
				max_vel = std::sqrt(max_vel);
				next_dt = compute_dt(max_vel, cur_time, next_time, dt_default);
				fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}, max_vel: {}\n", cur_time, next_dt, next_time, dt_default, max_vel);

				/// g2p2g
				issue([this](int did) {
					auto& cu_dev = Cuda::ref_cuda_context(did);
					CudaTimer timer {cu_dev.stream_compute()};

					/// check capacity
					if(checked_counts[did][1] > 0) {
						match(particle_bins[rollid ^ 1][did])([this, &did](auto& particle_buffer) {
							particle_buffer.resize(DeviceAllocator {}, cur_num_active_bins[did]);
						});
						checked_counts[did][1]--;
					}

					timer.tick();
					// grid
					grid_blocks[1][did].reset(nbcount[did], cu_dev);
					// adv map
					check_cuda_errors(cudaMemsetAsync(partitions[rollid][did].cell_particle_counts, 0, sizeof(int) * exterior_block_count[did] * config::G_BLOCKVOLUME, cu_dev.stream_compute()));
					// g2p2g
					match(particle_bins[rollid][did])([this, &did, &cu_dev](const auto& particle_buffer) {
						if(partitions[rollid][did].h_count) {
							cu_dev.compute_launch({partitions[rollid][did].h_count, 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt, next_dt, static_cast<const ivec3*>(partitions[rollid][did].halo_blocks), particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid ^ 1][did]), partitions[rollid ^ 1][did], partitions[rollid][did], grid_blocks[0][did], grid_blocks[1][did]);
						}
					});
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} halo_g2p2g", did, cur_frame, cur_step));
				});
				sync();

				collect_halo_grid_blocks();

				issue([this](int did) {
					auto& cu_dev = Cuda::ref_cuda_context(did);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();
					match(particle_bins[rollid][did])([this, &did, &cu_dev](const auto& particle_buffer) {
						cu_dev.compute_launch({partition_block_count[did], 128, (512 * 3 * 4) + (512 * 4 * 4)}, g2p2g, dt, next_dt, static_cast<const ivec3*>(nullptr), particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid ^ 1][did]), partitions[rollid ^ 1][did], partitions[rollid][did], grid_blocks[0][did], grid_blocks[1][did]);
					});
					timer.tock(fmt::format("GPU[{}] frame {} step {} non_halo_g2p2g", did, cur_frame, cur_step));
					if(checked_counts[did][0] > 0) {
						partitions[rollid ^ 1][did].resize_partition(DeviceAllocator {}, cur_num_active_blocks[did]);
						checked_counts[did][0]--;
					}
				});
				sync();

				reduce_halo_grid_blocks();

				issue([this](int did) {
					auto& cu_dev = Cuda::ref_cuda_context(did);
					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();
					/// mark particle blocks
					partitions[rollid][did].build_particle_buckets(cu_dev, exterior_block_count[did]);

					int* active_block_marks = tmps[did].active_block_marks;
					int* destinations		= tmps[did].destinations;
					int* sources			= tmps[did].sources;
					check_cuda_errors(cudaMemsetAsync(active_block_marks, 0, sizeof(int) * nbcount[did], cu_dev.stream_compute()));
					/// mark grid blocks
					cu_dev.compute_launch({(nbcount[did] * config::G_BLOCKVOLUME + 127) / 128, 128}, mark_active_grid_blocks, static_cast<uint32_t>(nbcount[did]), grid_blocks[1][did], active_block_marks);
					cu_dev.compute_launch({(exterior_block_count[did] + 1 + 127) / 128, 128}, mark_active_particle_blocks, exterior_block_count[did] + 1, partitions[rollid][did].particle_bucket_sizes, sources);
					exclusive_scan(exterior_block_count[did] + 1, sources, destinations, cu_dev);
					/// building new partition
					// block count
					check_cuda_errors(cudaMemcpyAsync(partitions[rollid ^ 1][did].count, destinations + exterior_block_count[did], sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					check_cuda_errors(cudaMemcpyAsync(&partition_block_count[did], destinations + exterior_block_count[did], sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.compute_launch({(exterior_block_count[did] + 255) / 256, 256}, exclusive_scan_inverse, exterior_block_count[did], static_cast<const int*>(destinations), sources);
					// indextable, activeKeys, particle_bucket_size, buckets
					partitions[rollid ^ 1][did].reset_table(cu_dev.stream_compute());
					cu_dev.syncStream<streamIdx::COMPUTE>();
					cu_dev.compute_launch({partition_block_count[did], 128}, update_partition, static_cast<uint32_t>(partition_block_count[did]), static_cast<const int*>(sources), partitions[rollid][did], partitions[rollid ^ 1][did]);
					// bin_offsets
					{
						int* bin_sizes = tmps[did].bin_sizes;
						cu_dev.compute_launch({(partition_block_count[did] + 1 + 127) / 128, 128}, compute_bin_capacity, partition_block_count[did] + 1, static_cast<const int*>(partitions[rollid ^ 1][did].particle_bucket_sizes), bin_sizes);
						exclusive_scan(partition_block_count[did] + 1, bin_sizes, partitions[rollid ^ 1][did].bin_offsets, cu_dev);
						check_cuda_errors(cudaMemcpyAsync(&bincount[did], partitions[rollid ^ 1][did].bin_offsets + partition_block_count[did], sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
						cu_dev.syncStream<streamIdx::COMPUTE>();
					}
					timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition", did, cur_frame, cur_step));

					/// neighboring blocks
					timer.tick();
					cu_dev.compute_launch({(partition_block_count[did] + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count[did]), partitions[rollid ^ 1][did]);
					auto prev_neighbor_block_count = nbcount[did];
					check_cuda_errors(cudaMemcpyAsync(&nbcount[did], partitions[rollid ^ 1][did].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_grid", did, cur_frame, cur_step));

					/// check capacity
					if(checked_counts[did][0] > 0) {
						grid_blocks[0][did].resize(DeviceAllocator {}, cur_num_active_blocks[did]);
					}
					/// rearrange grid blocks
					timer.tick();
					grid_blocks[0][did].reset(exterior_block_count[did], cu_dev);
					cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks, static_cast<const ivec3*>(partitions[rollid][did].active_keys), partitions[rollid ^ 1][did], static_cast<const int*>(active_block_marks), grid_blocks[1][did], grid_blocks[0][did]);
					cu_dev.syncStream<streamIdx::COMPUTE>();
					timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks", did, cur_frame, cur_step));
					/// check capacity
					if(checked_counts[did][0] > 0) {
						grid_blocks[1][did].resize(DeviceAllocator {}, cur_num_active_blocks[did]);
						tmps[did].resize(cur_num_active_blocks[did]);
					}
				});
				sync();

				/// halo tag
				halo_tagging();

				issue([this](int did) {
					auto& cu_dev = Cuda::ref_cuda_context(did);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();
					/// exterior blocks
					cu_dev.compute_launch({(partition_block_count[did] + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count[did]), partitions[rollid ^ 1][did]);
					check_cuda_errors(cudaMemcpyAsync(&exterior_block_count[did], partitions[rollid ^ 1][did].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]; {} [{}]\n", did, partition_block_count[did], nbcount[did], exterior_block_count[did], cur_num_active_blocks[did], bincount[did], cur_num_active_bins[did]);
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_particles", did, cur_frame, cur_step));
				});
				sync();
				rollid ^= 1;
				dt = next_dt;
			}
			issue([this](int did) {
				IO::flush();
				output_model(did);
			});
			sync();
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
	void output_model(int did) {
		auto& cu_dev = Cuda::ref_cuda_context(did);
		cu_dev.set_context();
		CudaTimer timer {cu_dev.stream_compute()};
		timer.tick();
		int particle_count	  = 0;
		int* d_particle_count = static_cast<int*>(cu_dev.borrow(sizeof(int)));
		check_cuda_errors(cudaMemsetAsync(d_particle_count, 0, sizeof(int), cu_dev.stream_compute()));
		match(particle_bins[rollid][did])([this, &cu_dev, &did, &d_particle_count](const auto& particle_buffer) {
			cu_dev.compute_launch({partition_block_count[did], 128}, retrieve_particle_buffer, partitions[rollid][did], partitions[rollid ^ 1][did], particle_buffer, particles[did], d_particle_count);
		});
		check_cuda_errors(cudaMemcpyAsync(&particle_count, d_particle_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
		cu_dev.syncStream<streamIdx::COMPUTE>();
		fmt::print(fg(fmt::color::red), "total number of particles {}\n", particle_count);
		models[did].resize(particle_count);
		check_cuda_errors(cudaMemcpyAsync(models[did].data(), (void*) &particles[did].val_1d(_0, 0), sizeof(std::array<float, 3>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
		cu_dev.syncStream<streamIdx::COMPUTE>();
		std::string fn = std::string {"model"} + "_dev[" + std::to_string(did) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";
		IO::insert_job([fn, model = models[did]]() {
			write_partio<float, config::NUM_DIMENSIONS>(fn, model);
		});
		timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", did, cur_frame, cur_step));
	}
	//NOLINTEND(readability-magic-numbers)

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)Current c++ version does not yet support std::span
	void initial_setup() {
		issue([this](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			cu_dev.set_context();
			CudaTimer timer {cu_dev.stream_compute()};
			timer.tick();
			cu_dev.compute_launch({(particle_counts[did] + 255) / 256, 256}, activate_blocks, particle_counts[did], particles[did], partitions[rollid ^ 1][did]);
			check_cuda_errors(cudaMemcpyAsync(&partition_block_count[did], partitions[rollid ^ 1][did].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			timer.tock(fmt::format("GPU[{}] step {} init_table", did, cur_step));

			timer.tick();
			cu_dev.reset_mem();
			// particle block
			cu_dev.compute_launch({(particle_counts[did] + 255) / 256, 256}, build_particle_cell_buckets, particle_counts[did], particles[did], partitions[rollid ^ 1][did]);
			// bucket, bin_offsets
			cu_dev.syncStream<streamIdx::COMPUTE>();
			partitions[rollid ^ 1][did].build_particle_buckets(cu_dev, partition_block_count[did]);
			{
				int* bin_sizes = tmps[did].bin_sizes;
				cu_dev.compute_launch({(partition_block_count[did] + 1 + 127) / 128, 128}, compute_bin_capacity, partition_block_count[did] + 1, static_cast<const int*>(partitions[rollid ^ 1][did].particle_bucket_sizes), bin_sizes);
				exclusive_scan(partition_block_count[did] + 1, bin_sizes, partitions[rollid ^ 1][did].bin_offsets, cu_dev);
				check_cuda_errors(cudaMemcpyAsync(&bincount[did], partitions[rollid ^ 1][did].bin_offsets + partition_block_count[did], sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
				cu_dev.syncStream<streamIdx::COMPUTE>();
			}
			match(particle_bins[rollid][did])([this, &cu_dev, &did](const auto& particle_buffer) {
				cu_dev.compute_launch({partition_block_count[did], 128}, array_to_buffer, particles[did], particle_buffer, partitions[rollid ^ 1][did]);
			});
			// grid block
			cu_dev.compute_launch({(partition_block_count[did] + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count[did]), partitions[rollid ^ 1][did]);
			check_cuda_errors(cudaMemcpyAsync(&nbcount[did], partitions[rollid ^ 1][did].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			cu_dev.compute_launch({(partition_block_count[did] + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count[did]), partitions[rollid ^ 1][did]);
			check_cuda_errors(cudaMemcpyAsync(&exterior_block_count[did], partitions[rollid ^ 1][did].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_partition", did, cur_step));

			fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]; {} [{}]\n", did, partition_block_count[did], nbcount[did], exterior_block_count[did], cur_num_active_blocks[did], bincount[did], cur_num_active_bins[did]);
		});
		sync();

		halo_tagging();

		issue([this](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			CudaTimer timer {cu_dev.stream_compute()};

			/// need to copy halo tag info as well
			partitions[rollid ^ 1][did].copy_to(partitions[rollid][did], exterior_block_count[did], cu_dev.stream_compute());
			check_cuda_errors(cudaMemcpyAsync(partitions[rollid][did].active_keys, partitions[rollid ^ 1][did].active_keys, sizeof(ivec3) * exterior_block_count[did], cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();

			timer.tick();
			grid_blocks[0][did].reset(nbcount[did], cu_dev);
			cu_dev.compute_launch({(particle_counts[did] + 255) / 256, 256}, rasterize, particle_counts[did], particles[did], grid_blocks[0][did], partitions[rollid][did], dt, get_mass(did));
			cu_dev.compute_launch({partition_block_count[did], 128}, init_adv_bucket, static_cast<const int*>(partitions[rollid][did].particle_bucket_sizes), partitions[rollid][did].blockbuckets);
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} init_grid", did, cur_step));
		});
		sync();

		collect_halo_grid_blocks(0);
		reduce_halo_grid_blocks(0);
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)Current c++ version does not yet support std::span
	void halo_tagging() {
		issue([this](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			cu_dev.reset_mem();
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				if(otherdid != did) {
					halo_block_ids[did][otherdid] = static_cast<ivec3*>(cu_dev.borrow(sizeof(ivec3) * nbcount[otherdid]));
				}
			}
			/// init halo blockids
			output_halo_grid_blocks[did].init_blocks(TempAllocator {did}, nbcount[did]);
			input_halo_grid_blocks[did].init_blocks(TempAllocator {did}, nbcount[did]);
		});
		sync();
		issue([this](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			/// prepare counts
			output_halo_grid_blocks[did].reset_counts(cu_dev.stream_compute());
			cu_dev.syncStream<streamIdx::COMPUTE>();
			/// sharing local active blocks
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				if(otherdid != did) {
					check_cuda_errors(cudaMemcpyAsync(halo_block_ids[otherdid][did], partitions[rollid ^ 1][did].active_keys, sizeof(ivec3) * nbcount[did], cudaMemcpyDefault, cu_dev.stream_spare(otherdid)));
					cu_dev.spare_event_record(otherdid);
				}
			}
		});
		sync();
		issue([this](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			CudaTimer timer {cu_dev.stream_compute()};
			timer.tick();
			/// init overlap marks
			partitions[rollid ^ 1][did].reset_overlap_marks(nbcount[did], cu_dev.stream_compute());
			cu_dev.syncStream<streamIdx::COMPUTE>();
			/// receiving active blocks from other devices
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				if(otherdid != did) {
					cu_dev.spareStreamWaitForEvent(otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
					cu_dev.spare_launch(otherdid, {(nbcount[otherdid] + 127) / 128, 128}, mark_overlapping_blocks, static_cast<uint32_t>(nbcount[otherdid]), otherdid, static_cast<const ivec3*>(halo_block_ids[did][otherdid]), partitions[rollid ^ 1][did], output_halo_grid_blocks[did].counts + otherdid, output_halo_grid_blocks[did].buffers[otherdid]);
					cu_dev.spare_event_record(otherdid);
					cu_dev.computeStreamWaitForEvent(cu_dev.event_spare(otherdid));
				}
			}
			// self halo particle block
			partitions[rollid ^ 1][did].reset_halo_count(cu_dev.stream_compute());
			cu_dev.compute_launch({(partition_block_count[did] + 127) / 128, 128}, collect_blockids_for_halo_reduction, static_cast<uint32_t>(partition_block_count[did]), did, partitions[rollid ^ 1][did]);
			/// retrieve counts
			partitions[rollid ^ 1][did].retrieve_halo_count(cu_dev.stream_compute());
			output_halo_grid_blocks[did].retrieve_counts(cu_dev.stream_compute());
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} halo_tagging", did, cur_step));

			fmt::print(fg(fmt::color::green), "halo particle blocks[{}]: {}\n", did, partitions[rollid ^ 1][did].h_count);
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				fmt::print(fg(fmt::color::green), "halo grid blocks[{}][{}]: {}\n", did, otherdid, output_halo_grid_blocks[did].h_counts[otherdid]);
			}
		});
		sync();
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)

	void collect_halo_grid_blocks(int gid = 1) {
		/// init halo grid blocks
		issue([this](int did) {
			std::vector<uint32_t> counts(config::G_DEVICE_COUNT);
			output_halo_grid_blocks[did].init_buffer(TempAllocator {did}, output_halo_grid_blocks[did].h_counts);
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				counts[otherdid] = (otherdid != did) ? output_halo_grid_blocks[otherdid].h_counts[did] : 0;
			}
			input_halo_grid_blocks[did].init_buffer(TempAllocator {did}, counts);
		});
		sync();
		issue([this, gid](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			CppTimer timer {};
			timer.tick();
			/// sharing local active blocks
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				if(otherdid != did) {
					if(output_halo_grid_blocks[did].h_counts[otherdid] > 0) {
						auto& count = output_halo_grid_blocks[did].h_counts[otherdid];
						cu_dev.spare_launch(otherdid, {count, config::G_BLOCKVOLUME}, collect_grid_blocks, grid_blocks[gid][did], partitions[rollid][did], output_halo_grid_blocks[did].buffers[otherdid]);
						output_halo_grid_blocks[did].send(input_halo_grid_blocks[otherdid], did, otherdid, cu_dev.stream_spare(otherdid));
						cu_dev.spare_event_record(otherdid);
					} else {
						input_halo_grid_blocks[otherdid].h_counts[did] = 0;
					}
				}
			}
			timer.tock(fmt::format("GPU[{}] step {} collect_send_halo_grid", did, cur_step));
		});
		sync();
	}

	void reduce_halo_grid_blocks(int gid = 1) {
		issue([this, gid](int did) {
			auto& cu_dev = Cuda::ref_cuda_context(did);
			CppTimer timer {};
			timer.tick();
			/// receiving active blocks from other devices
			for(int otherdid = 0; otherdid < config::G_DEVICE_COUNT; otherdid++) {
				if(otherdid != did) {
					if(input_halo_grid_blocks[did].h_counts[otherdid] > 0) {
						cu_dev.spareStreamWaitForEvent(otherdid, Cuda::ref_cuda_context(otherdid).event_spare(did));
						cu_dev.spare_launch(otherdid, {input_halo_grid_blocks[did].h_counts[otherdid], config::G_BLOCKVOLUME}, reduce_grid_blocks, grid_blocks[gid][did], partitions[rollid][did], input_halo_grid_blocks[did].buffers[otherdid]);
						cu_dev.spare_event_record(otherdid);
						cu_dev.computeStreamWaitForEvent(cu_dev.event_spare(otherdid));
					}
				}
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			timer.tock(fmt::format("GPU[{}] step {} receive_reduce_halo_grid", did, cur_step));
		});
		sync();
	}
};

}// namespace mn

#endif