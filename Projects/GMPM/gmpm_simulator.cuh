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
			bin_sizes			   = static_cast<int*>(static_cast<void*>(static_cast<char*>(base) + sizeof(int) * max_block_count * 4));
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
	std::array<std::size_t, BIN_COUNT> checked_counts = {};
	std::vector<std::size_t> checked_bin_counts		= {};
	float max_vels;
	int partition_block_count;
	int neighbor_block_count;
	int exterior_block_count;///< num blocks
	std::vector<int> bincount										 = {};
	std::vector<uint32_t> particle_counts									 = {};///< num particles
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
		, partition_block_count()
		, neighbor_block_count()
		, exterior_block_count() {
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
		
		//Allocate intermediate data for all blocks
		tmps.alloc(config::G_MAX_ACTIVE_BLOCK);
		
		//Create partitions and grid blocks
		for(int copyid = 0; copyid < BIN_COUNT; copyid++) {
			grid_blocks.emplace_back(DeviceAllocator {});
			partitions.emplace_back(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			checked_counts[copyid] = 0;
		}
		
		cu_dev.syncStream<streamIdx::COMPUTE>();
		cur_num_active_blocks = config::G_MAX_ACTIVE_BLOCK;
	}

	template<MaterialE M>
	void init_model(const std::vector<std::array<float, config::NUM_DIMENSIONS>>& model, const mn::vec<float, config::NUM_DIMENSIONS>& v0) {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		
		//Create particle buffers and reserve buckets
		for(int copyid = 0; copyid < BIN_COUNT; ++copyid) {
			particle_bins[copyid].emplace_back(ParticleBuffer<M>(DeviceAllocator {}, model.size() / config::G_BIN_CAPACITY + config::G_MAX_ACTIVE_BLOCK));
			match(particle_bins[copyid].back())([&](auto& particle_buffer) {
				particle_buffer.reserve_buckets(DeviceAllocator {}, config::G_MAX_ACTIVE_BLOCK);
			});
		}
		
		//Set initial velocity
		vel0.emplace_back();
		for(int i = 0; i < config::NUM_DIMENSIONS; ++i) {
			vel0.back()[i] = v0[i];
		}

		//Create array for initial particles
		particles.emplace_back(ParticleArray {spawn<particle_array_, orphan_signature>(DeviceAllocator {}, sizeof(float) * config::NUM_DIMENSIONS * model.size())});
		
		//Init bin counts
		cur_num_active_bins.emplace_back(config::G_MAX_PARTICLE_BIN);
		bincount.emplace_back(0);
		checked_bin_counts.emplace_back(0);

		//Init particle counts
		particle_counts.emplace_back(static_cast<unsigned int>(model.size()));//NOTE: Explicic narrowing cast
		
		
		fmt::print("init {}-th model with {} particles\n", particle_bins[0].size() - 1, particle_counts.back());
		
		//Copy particle positions from host to device
		cudaMemcpyAsync(static_cast<void*>(&particles.back().val_1d(_0, 0)), model.data(), sizeof(std::array<float, config::NUM_DIMENSIONS>) * model.size(), cudaMemcpyDefault, cu_dev.stream_compute());
		cu_dev.syncStream<streamIdx::COMPUTE>();

		//Write out initial state to file
		std::string fn = std::string {"model"} + "_id[" + std::to_string(particle_bins[0].size() - 1) + "]_frame[0].bgeo";
		IO::insert_job([fn, model]() {
			write_partio<float, config::NUM_DIMENSIONS>(fn, model);
		});
		IO::flush();
	}

	void update_fr_parameters(float rho, float vol, float ym, float pr) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::FIXED_COROTATED>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr);
			}
		);
	}

	void update_j_fluid_parameters(float rho, float vol, float bulk, float gamma, float visco) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, bulk, gamma, visco);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::J_FLUID>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, bulk, gamma, visco);
			}
		);
	}

	void update_nacc_parameters(float rho, float vol, float ym, float pr, float beta, float xi) {
		match(particle_bins[0].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::NACC>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
		match(particle_bins[1].back())(
			[&](auto& particle_buffer) {},
			[&](ParticleBuffer<MaterialE::NACC>& particle_buffer) {
				particle_buffer.update_parameters(rho, vol, ym, pr, beta, xi);
			}
		);
	}

	//Sum up count values from in and store them in out
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

	float get_mass(int id = 0) {
		return match(particle_bins[rollid][id])([&](const auto& particle_buffer) {
			return particle_buffer.mass;
		});
	}

	[[nodiscard]] int get_model_count() const noexcept {
		return static_cast<int>(particle_bins[0].size());//NOTE: Explicit narrowing cast (But we should not have that much models anyway.)
	}

	//Increase bin and active block count if too low
	void check_capacity() {
		//TODO: Is that right? Maybe create extra parameter for this?
		//NOLINTBEGIN(readability-magic-numbers) Magic numbers are resize thresholds?
		if(exterior_block_count > cur_num_active_blocks * config::NUM_DIMENSIONS / 4 && checked_counts[0] == 0) {
			cur_num_active_blocks = cur_num_active_blocks * config::NUM_DIMENSIONS / 2;
			checked_counts[0]		  = 2;
			fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", exterior_block_count, cur_num_active_blocks);
		}
		
		for(int i = 0; i < get_model_count(); ++i) {
			if(bincount[i] > cur_num_active_bins[i] * config::NUM_DIMENSIONS / 4 && checked_bin_counts[i] == 0) {
				cur_num_active_bins[i] = cur_num_active_bins[i] * config::NUM_DIMENSIONS / 2;
				checked_bin_counts[i]	   = 2;
				fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincount[i], cur_num_active_bins[i]);
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
			for(int i = 0; i < get_model_count(); ++i) {
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
				//Calculate maximum grid velocity and update the grid velocity
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					
					/// check capacity
					check_capacity();
					
					CudaTimer timer {cu_dev.stream_compute()};
					timer.tick();
					
					float* d_max_vel = tmps.d_max_vel;
					//Initialize max_vel with 0
					check_cuda_errors(cudaMemsetAsync(d_max_vel, 0, sizeof(float), cu_dev.stream_compute()));
					
					//Update the grid velocity
					cu_dev.compute_launch({(neighbor_block_count + config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK - 1) / config::G_NUM_GRID_BLOCKS_PER_CUDA_BLOCK, config::G_NUM_WARPS_PER_CUDA_BLOCK * 32}, update_grid_velocity_query_max, static_cast<uint32_t>(neighbor_block_count), grid_blocks[0], partitions[rollid], dt, d_max_vel);
					
					//Copy maximum velocity to host site
					check_cuda_errors(cudaMemcpyAsync(&max_vels, d_max_vel, sizeof(float), cudaMemcpyDefault, cu_dev.stream_compute()));
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query", gpuid, cur_frame, cur_step));
				}

				/// host: compute maxvel & next dt
				float max_vel = max_vels;
				// if (max_vels > max_vel)
				//  max_vel = max_vels[id];
			
				//If our maximum velocity is infinity our computation will crash, so we stop here.
				if(std::isinf(max_vel)){
					std::cout << "Maximum velocity is infinity" << std::endl;
					goto outer_loop_end;
				}
			
				max_vel = std::sqrt(max_vel);// this is a bug, should insert this line
				next_dt = compute_dt(max_vel, cur_time, next_time, dt_default);
				fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}, max_vel: {}\n", cur_time, next_dt, next_time, dt_default, max_vel);

				/// g2p2g
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					//Resize particle buffers if we increased the size of active bins
					//This also removes all particle data of next particle buffer but does not clear it
					for(int i = 0; i < get_model_count(); ++i) {
						if(checked_bin_counts[i] > 0) {
							match(particle_bins[rollid ^ 1][i])([this, &i](auto& particle_buffer) {
								particle_buffer.resize(DeviceAllocator {}, cur_num_active_bins[i]);
							});
							checked_bin_counts[i]--;
						}
					}

					timer.tick();
					
					//Clear the grid
					grid_blocks[1].reset(neighbor_block_count, cu_dev);
					
					//Perform G2P2G step
					for(int i = 0; i < get_model_count(); ++i) {
						//First clear the count of particles per cell for next buffer
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& particle_buffer) {
							check_cuda_errors(cudaMemsetAsync(particle_buffer.cell_particle_counts, 0, sizeof(int) * exterior_block_count * config::G_BLOCKVOLUME, cu_dev.stream_compute()));
						});
						
						//Perform g2p2g
						match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
							cu_dev.compute_launch({partition_block_count, 128}, g2p2g, dt, next_dt, particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid ^ 1][i]), partitions[rollid ^ 1], partitions[rollid], grid_blocks[0], grid_blocks[1]);
						});
					}
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", gpuid, cur_frame, cur_step));
					
					//Resize partition if we increased the size of active blocks
					//This also deletes current particle buffer meta data.
					if(checked_counts[0] > 0) {
						partitions[rollid ^ 1].resize_partition(DeviceAllocator {}, cur_num_active_blocks);
						for(int i = 0; i < get_model_count(); ++i) {
							match(particle_bins[rollid][i])([this, &cu_dev](auto& particle_buffer) {
								particle_buffer.reserve_buckets(DeviceAllocator {}, cur_num_active_blocks);
							});
						}
						checked_counts[0]--;
					}
				}

				/// update partition
				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};
					
					timer.tick();
					
					//Copy cell buckets from partition to next particle buffer
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& particle_buffer) {
							//First init sizes with 0
							check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
							
							cu_dev.compute_launch({exterior_block_count, config::G_BLOCKVOLUME}, cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
							// partitions[rollid].buildParticleBuckets(cu_dev, exterior_block_count);
						});
					}

					int* active_block_marks = tmps.active_block_marks;
					int* destinations		= tmps.destinations;
					int* sources			= tmps.sources;
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(active_block_marks, 0, sizeof(int) * neighbor_block_count, cu_dev.stream_compute()));
					
					//Mark cells that have mass bigger 0.0
					cu_dev.compute_launch({(neighbor_block_count * config::G_BLOCKVOLUME + 127) / 128, 128}, mark_active_grid_blocks, static_cast<uint32_t>(neighbor_block_count), grid_blocks[1], active_block_marks);
					
					//Clear marks
					check_cuda_errors(cudaMemsetAsync(sources, 0, sizeof(int) * (exterior_block_count + 1), cu_dev.stream_compute()));
					
					//Mark particle buckets that have at least one particle
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev, &sources](auto& particle_buffer) {
							cu_dev.compute_launch({(exterior_block_count + 1 + 127) / 128, 128}, mark_active_particle_blocks, exterior_block_count + 1, particle_buffer.particle_bucket_sizes, sources);
						});
					}
					
					//Sum up number of active buckets and calculate offsets (empty buckets are collapsed
					exclusive_scan(exterior_block_count + 1, sources, destinations, cu_dev);
					
					/// building new partition
					
					//Store new bucket count
					check_cuda_errors(cudaMemcpyAsync(partitions[rollid ^ 1].count, destinations + exterior_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					check_cuda_errors(cudaMemcpyAsync(&partition_block_count, destinations + exterior_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					
					//Calculate indices of block by position
					cu_dev.compute_launch({(exterior_block_count + 255) / 256, 256}, exclusive_scan_inverse, exterior_block_count, static_cast<const int*>(destinations), sources);
					
					
					//Reset partitions
					partitions[rollid ^ 1].reset_table(cu_dev.stream_compute());
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					//Check size
					if(partition_block_count > config::G_MAX_ACTIVE_BLOCK){
						std::cerr << "Too much active blocks: " << partition_block_count << std::endl;
						std::abort();
					}
					
					//Reinsert buckets
					cu_dev.compute_launch({(partition_block_count + 127) / 128, 128}, update_partition, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), partitions[rollid], partitions[rollid ^ 1]);
					
					//Copy block buckets and sizes from next particle buffer to current particle buffer
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid ^ 1][i])([this, &cu_dev, &sources, &i](auto& particle_buffer) {
							auto& next_particle_buffer = get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid][i]);
							cu_dev.compute_launch({partition_block_count, 128}, update_buckets, static_cast<uint32_t>(partition_block_count), static_cast<const int*>(sources), particle_buffer, next_particle_buffer);
						});
					}
					
					//Compute bin capacities, bin offsets and the summed bin size for current particle buffer
					int* bin_sizes = tmps.bin_sizes;
					for(int i = 0; i < get_model_count(); ++i) {
						match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
							cu_dev.compute_launch({(partition_block_count + 1 + 127) / 128, 128}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);
							
							//Stores aggregated bin sizes in particle_buffer 
							exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);
							
							//Stores last aggregated size == whole size in bincount
							check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
							cu_dev.syncStream<streamIdx::COMPUTE>();
						});
					}
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition", gpuid, cur_frame, cur_step));

					timer.tick();
					
					//Activate blocks near active blocks
					cu_dev.compute_launch({(partition_block_count + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[rollid ^ 1]);
					
					//Retrieve total count
					auto prev_neighbor_block_count = neighbor_block_count;
					check_cuda_errors(cudaMemcpyAsync(&neighbor_block_count, partitions[rollid ^ 1].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					//Check size
					if(neighbor_block_count > config::G_MAX_ACTIVE_BLOCK){
						std::cerr << "Too much neighbour blocks: " << partition_block_count << std::endl;
						std::abort();
					}
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} build_partition_for_grid", gpuid, cur_frame, cur_step));

					//Resize grid if necessary
					if(checked_counts[0] > 0) {
						grid_blocks[0].resize(DeviceAllocator {}, cur_num_active_blocks);
					}
					
					timer.tick();
					
					//Clear the grid
					grid_blocks[0].reset(exterior_block_count, cu_dev);
					
					//Copy values from old grid for active blocks
					cu_dev.compute_launch({prev_neighbor_block_count, config::G_BLOCKVOLUME}, copy_selected_grid_blocks, static_cast<const ivec3*>(partitions[rollid].active_keys), partitions[rollid ^ 1], static_cast<const int*>(active_block_marks), grid_blocks[1], grid_blocks[0]);
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks", gpuid, cur_frame, cur_step));
					
					//Resize grid if necessary
					if(checked_counts[0] > 0) {
						grid_blocks[1].resize(DeviceAllocator {}, cur_num_active_blocks);
						tmps.resize(cur_num_active_blocks);
					}
				}

				{
					auto& cu_dev = Cuda::ref_cuda_context(gpuid);
					CudaTimer timer {cu_dev.stream_compute()};

					timer.tick();
					
					//Activate blocks near active blocks, including those before that block
					cu_dev.compute_launch({(partition_block_count + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count), partitions[rollid ^ 1]);
					
					//Retrieve total count
					check_cuda_errors(cudaMemcpyAsync(&exterior_block_count, partitions[rollid ^ 1].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					//Check size
					if(exterior_block_count > config::G_MAX_ACTIVE_BLOCK){
						std::cerr << "Too much exterior blocks: " << partition_block_count << std::endl;
						std::abort();
					}

					fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, partition_block_count, neighbor_block_count, exterior_block_count, cur_num_active_blocks);
					for(int i = 0; i < get_model_count(); ++i) {
						fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincount[i], cur_num_active_bins[i]);
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
		outer_loop_end:
		(void) nullptr;//We need a statement to have a valid jump label
	}
	//NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, readability-magic-numbers)

	//FIXME: no signed integer bitwise operations! (rollid)
	//TODO: Check magic numbers and replace by constants
	//NOLINTBEGIN(readability-magic-numbers)
	void output_model() {
		auto& cu_dev = Cuda::ref_cuda_context(gpuid);
		CudaTimer timer {cu_dev.stream_compute()};
		
		timer.tick();
		
		for(int i = 0; i < get_model_count(); ++i) {
			int particle_count	  = 0;
			int* d_particle_count = static_cast<int*>(cu_dev.borrow(sizeof(int)));
			
			//Init particle count with 0
			check_cuda_errors(cudaMemsetAsync(d_particle_count, 0, sizeof(int), cu_dev.stream_compute()));
			
			//Copy particle data to output buffer
			match(particle_bins[rollid][i])([this, &cu_dev, &i, &d_particle_count](const auto& particle_buffer) {
				cu_dev.compute_launch({partition_block_count, 128}, retrieve_particle_buffer, partitions[rollid], partitions[rollid ^ 1], particle_buffer, get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid ^ 1][i]), particles[i], d_particle_count);
			});
			
			//Retrieve particle count
			check_cuda_errors(cudaMemcpyAsync(&particle_count, d_particle_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			fmt::print(fg(fmt::color::red), "total number of particles {}\n", particle_count);
			
			//Resize the output model
			model.resize(particle_count);
			
			//Copy the data to the output model
			check_cuda_errors(cudaMemcpyAsync(model.data(), static_cast<void*>(&particles[i].val_1d(_0, 0)), sizeof(std::array<float, 3>) * (particle_count), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			std::string fn = std::string {"model"} + "_id[" + std::to_string(i) + "]_frame[" + std::to_string(cur_frame) + "].bgeo";
			
			//Write back file
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
			//TODO: Verify bounds when model offset is too large
			
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};
			
			timer.tick();
			
			//Activate blocks that contain particles
			for(int i = 0; i < get_model_count(); ++i) {
				cu_dev.compute_launch({(particle_counts[i] + 255) / 256, 256}, activate_blocks, particle_counts[i], particles[i], partitions[rollid ^ 1]);
			}

			//Store count of activated blocks
			check_cuda_errors(cudaMemcpyAsync(&partition_block_count, partitions[rollid ^ 1].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			timer.tock(fmt::format("GPU[{}] step {} init_table", gpuid, cur_step));
			
			timer.tick();
			cu_dev.reset_mem();
			
			//Store particle ids in block cells
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](auto& particle_buffer) {
					cu_dev.compute_launch({(particle_counts[i] + 255) / 256, 256}, build_particle_cell_buckets, particle_counts[i], particles[i], particle_buffer, partitions[rollid ^ 1]);
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Check size
			if(partition_block_count > config::G_MAX_ACTIVE_BLOCK){
				std::cerr << "Too much active blocks: " << partition_block_count << std::endl;
				std::abort();
			}

			//Copy cell buckets from partition to particle buffer
			for(int i = 0; i < get_model_count(); ++i) {
				std::cout << i << " ";
				match(particle_bins[rollid][i])([this, &cu_dev](auto& particle_buffer) {
					//First init sizes with 0
					check_cuda_errors(cudaMemsetAsync(particle_buffer.particle_bucket_sizes, 0, sizeof(int) * (partition_block_count + 1), cu_dev.stream_compute()));
					
					cu_dev.compute_launch({partition_block_count, config::G_BLOCKVOLUME}, cell_bucket_to_block, particle_buffer.cell_particle_counts, particle_buffer.cellbuckets, particle_buffer.particle_bucket_sizes, particle_buffer.blockbuckets);
					// partitions[rollid ^ 1].buildParticleBuckets(cu_dev, partition_block_count);
				});
			}
			
			//Compute bin capacities, bin offsets and the summed bin size
			//Then initializes the particle buffer
			int* bin_sizes = tmps.bin_sizes;
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &bin_sizes, &i](auto& particle_buffer) {
					cu_dev.compute_launch({(partition_block_count + 1 + 127) / 128, 128}, compute_bin_capacity, partition_block_count + 1, static_cast<const int*>(particle_buffer.particle_bucket_sizes), bin_sizes);
					
					//Stores aggregated bin sizes in particle_buffer 
					exclusive_scan(partition_block_count + 1, bin_sizes, particle_buffer.bin_offsets, cu_dev);
					
					//Stores last aggregated size == whole size in bincount
					check_cuda_errors(cudaMemcpyAsync(&bincount[i], particle_buffer.bin_offsets + partition_block_count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
					cu_dev.syncStream<streamIdx::COMPUTE>();
					
					//Initialize particle buffer
					cu_dev.compute_launch({partition_block_count, 128}, array_to_buffer, particles[i], particle_buffer);
				});
			}
			
			
			//Activate blocks near active blocks
			cu_dev.compute_launch({(partition_block_count + 127) / 128, 128}, register_neighbor_blocks, static_cast<uint32_t>(partition_block_count), partitions[rollid ^ 1]);
			
			//Retrieve total count
			check_cuda_errors(cudaMemcpyAsync(&neighbor_block_count, partitions[rollid ^ 1].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Check size
			if(neighbor_block_count > config::G_MAX_ACTIVE_BLOCK){
				std::cerr << "Too much neighbour blocks: " << partition_block_count << std::endl;
				std::abort();
			}
			
			//Activate blocks near active blocks, including those before that block
			//TODO: Only these with offset -1 are not already activated as neighbours
			cu_dev.compute_launch({(partition_block_count + 127) / 128, 128}, register_exterior_blocks, static_cast<uint32_t>(partition_block_count), partitions[rollid ^ 1]);
			
			//Retrieve total count
			check_cuda_errors(cudaMemcpyAsync(&exterior_block_count, partitions[rollid ^ 1].count, sizeof(int), cudaMemcpyDefault, cu_dev.stream_compute()));
			cu_dev.syncStream<streamIdx::COMPUTE>();
			
			//Check size
			if(exterior_block_count > config::G_MAX_ACTIVE_BLOCK){
				std::cerr << "Too much exterior blocks: " << partition_block_count << std::endl;
				std::abort();
			}
			
			timer.tock(fmt::format("GPU[{}] step {} init_partition", gpuid, cur_step));

			fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "block count on device {}: {}, {}, {} [{}]\n", gpuid, partition_block_count, neighbor_block_count, exterior_block_count, cur_num_active_blocks);
			for(int i = 0; i < get_model_count(); ++i) {
				fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow), "bin count on device {}: model {}: {} [{}]\n", gpuid, i, bincount[i], cur_num_active_bins[i]);
			}
		}

		{
			auto& cu_dev = Cuda::ref_cuda_context(gpuid);
			CudaTimer timer {cu_dev.stream_compute()};

			//Copy all blocks to background partition
			partitions[rollid ^ 1].copy_to(partitions[rollid], exterior_block_count, cu_dev.stream_compute());
			check_cuda_errors(cudaMemcpyAsync(partitions[rollid].active_keys, partitions[rollid ^ 1].active_keys, sizeof(ivec3) * exterior_block_count, cudaMemcpyDefault, cu_dev.stream_compute()));
			
			//Copy all particle data to background particle buffer
			for(int i = 0; i < get_model_count(); ++i) {
				match(particle_bins[rollid][i])([this, &cu_dev, &i](const auto& particle_buffer) {
					// bin_offsets, particle_bucket_sizes
					particle_buffer.copy_to(get<typename std::decay_t<decltype(particle_buffer)>>(particle_bins[rollid ^ 1][i]), partition_block_count, cu_dev.stream_compute());
				});
			}
			cu_dev.syncStream<streamIdx::COMPUTE>();

			timer.tick();
			
			//Clear the grid
			grid_blocks[0].reset(neighbor_block_count, cu_dev);
			
			//Initialize the grid and advection buckets
			for(int i = 0; i < get_model_count(); ++i) {
				//Initialize mass and momentum
				cu_dev.compute_launch({(particle_counts[i] + 255) / 256, 256}, rasterize, particle_counts[i], particles[i], grid_blocks[0], partitions[rollid], dt, get_mass(i), vel0[i].data_arr());
				
				//Init advection source at offset 0 of destination
				match(particle_bins[rollid ^ 1][i])([this, &cu_dev](auto& particle_buffer) {
					cu_dev.compute_launch({partition_block_count, 128}, init_adv_bucket, static_cast<const int*>(particle_buffer.particle_bucket_sizes), particle_buffer.blockbuckets);
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