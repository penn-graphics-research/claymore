#include <cuda_occupancy.h>	  ///<	for optimal kernel launching
#include <cuda_profiler_api.h>///<	for evaluating kernel performance
#include <cuda_runtime.h>

#include <cstdio>
#include <utility>

#include "Cuda.h"

constexpr size_t MEM_POOL_CTRL = 3;

namespace mn {

//NOLINTBEGIN(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print
KernelConfig::KernelConfig(KernelFunc f, cudaFuncCache cache_config, bool is_wave)
	: attribs()
	, func(f)
	, cache_preference(cache_config)
	, wave_fashion(is_wave) {
	cudaFuncGetAttributes(&attribs, f);
	max_occ_block_size = Cuda::eval_optimal_block_size(attribs, cache_preference);
	if(cache_config != cudaFuncCachePreferNone) {///< should be different from
		///< device cache preference
		check_cuda_errors(cudaFuncSetCacheConfig(f, cache_config));
	}
}

Cuda::Cuda()
	: default_dev_id {0}
	, dev_num_using()
	, i_dev_id() {
	printf("[Init -- Begin] Cuda\n");
	//< acquire devices
	dev_num_available	 = 0;
	cudaError_t error_id = cudaGetDeviceCount(&dev_num_available);
	if(error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		throw std::runtime_error("Failure");
	}
	if(dev_num_available == 0) {
		printf(
			"\t[InitInfo -- DevNum] There are no available device(s) that "
			"support CUDA\n"
		);
	} else {
		printf("\t[InitInfo -- DevNum] Detected %d CUDA Capable device(s)\n", dev_num_available);
	}
	set_max_device();//<[TMP]
	printf("\t[InitInfo -- DevNum] Prepare to use %d device(s) in Multi-GPU test\n", dev_num_using);

	ak_device_props.resize(dev_num_available);
	ak_streams.resize(dev_num_available);
	ak_events.resize(dev_num_available);

	for(int i = 0; i < dev_num_available; i++) {
		// check_cuda_errors(cudaSetDevice(i));
		check_cuda_errors(cudaSetDevice(i));
		///< device properties
		check_cuda_errors(cudaGetDeviceProperties(&ak_device_props[i], i));
		const auto& prop {ak_device_props[i]};
		printf(
			"\t[InitInfo -- Dev Property] GPU device %d (%d-th group on "
			"board)\n\t\tglobal memory: %llu bytes,\n\t\tshared memory per "
			"block: %llu bytes,\n\t\tregisters per SM: %d,\n\t\tMulti-Processor "
			"count: %d,\n\t\tSM compute capabilities: %d.%d.\n",
			i,
			prop.multiGpuBoardGroupID,
			static_cast<long long unsigned int>(prop.totalGlobalMem),
			static_cast<long long unsigned int>(prop.sharedMemPerBlock),
			prop.regsPerBlock,
			prop.multiProcessorCount,
			prop.major,
			prop.minor
		);

		///< streams
		// int leastPriority = 0;
		// int greatestPriority = leastPriority;
		// cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
		// cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault,
		// leastPriority); cudaStreamCreateWithPriority(&push_top_stream,
		// cudaStreamDefault, greatestPriority);
		// cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault,
		// greatestPriority);
		// for (auto &stream : ak_streams[i])
		for(int j = 0; j < static_cast<int>(StreamIndex::TOTAL); ++j) {
			check_cuda_errors(cudaStreamCreate(&ak_streams[i][j]));
		}
		printf("\t[InitInfo -- stream] Create %lu streams for device %d\n", static_cast<long unsigned>(ak_streams[i].extent), i);

		// for (auto &event : ak_events[i])
		for(int j = 0; j < static_cast<int>(EventIndex::TOTAL); ++j) {
			check_cuda_errors(cudaEventCreateWithFlags(
				&ak_events[i][j],// cudaEventDefault | cudaEventBlockingSync |
				// cudaEventInterprocess |
				cudaEventDisableTiming
			));
		}

		///< memory allocator
		std::size_t free_byte, total_byte;
		check_cuda_errors(cudaMemGetInfo(&free_byte, &total_byte));
		///
		ak_monotonic_allocators.emplace_back(std::make_unique<MonotonicAllocator>(prop.textureAlignment,
																				  free_byte >> MEM_POOL_CTRL));///< preserve 1/4 space for intermediate
		///< computations
		cudaDeviceSynchronize();
		printf(
			"\t[InitInfo -- memory] device %d\n\t\tfree bytes/total bytes: "
			"%lu/%lu,\n\t\tpre-allocated size: %lu bytes\n\n",
			i,
			static_cast<long unsigned>(free_byte),
			static_cast<long unsigned>(total_byte),
			static_cast<long unsigned>(free_byte >> MEM_POOL_CTRL)
		);
	}

	//< enable peer access
	for(int i = 0; i < dev_num_available; i++) {
		for(int j = 0; j < dev_num_available; j++) {
			if(i != j) {
				establish_peer_access(i, j);
				printf("\t[InitInfo -- Peer Access] Enable peer access from %d to %d\n", i, j);
			}
		}
	}
	//< init cuda context
	for(int i = 0; i < dev_num_available; i++) {
		ak_cu_dev_contexts.emplace_back(i);//< set device when construct
	}

	printf("\t[InitInfo -- Default Dev] Default context: %d\n", default_dev_id);
	check_cuda_errors(cudaSetDevice(default_dev_id));
	printf("\n[Init -- End] == Finished \'Cuda\' initialization\n\n");
	// getchar();
}

Cuda::~Cuda() {
	// cudaStreamDestroy(_kMemCopyStream);
	get_instance()->ak_device_props.clear();
	for(auto& streams: get_instance()->ak_streams) {
		// for (auto &stream : streams)
		for(int j = 0; j < static_cast<int>(StreamIndex::TOTAL); ++j) {
			check_cuda_errors(cudaStreamDestroy(streams[j]));
		}
	}
	for(auto& events: get_instance()->ak_events) {
		// for (auto &event : events)
		for(int j = 0; j < static_cast<int>(EventIndex::TOTAL); ++j) {
			check_cuda_errors(cudaEventDestroy(events[j]));
		}
	}
	//for (auto &monoAllocator : ak_monotonic_allocators){
	//for (int i = 0; i < get_instance()->ak_monotonic_allocators.size(); ++i){
	//  get_instance()->ak_monotonic_allocators[i].~MonotonicAllocator();
	//}
	//}
	printf("  Finished \'Cuda\' termination\n");
}

int Cuda::general_grid_size(int& thread_num, int& block_size) const {
	return (thread_num + block_size - 1) / block_size;
}

int Cuda::wave_grid_size(int& thread_num, int& block_size) const {
	auto blocksPerSM = (thread_num / block_size / get_instance()->ak_device_props[i_dev_id].multiProcessorCount) * get_instance()->ak_device_props[i_dev_id].multiProcessorCount;
	return (blocksPerSM != 0) ? blocksPerSM : 1;
}

/// static methods
int Cuda::eval_optimal_block_size(cudaFuncAttributes attribs, cudaFuncCache cache_preference, std::size_t smem_bytes) {
	auto* instance					  = get_instance();
	cudaOccDeviceProp prop			  = get_instance()->ak_device_props[instance->i_dev_id];///< cache preference
	cudaOccFuncAttributes occ_attribs = attribs;
	cudaOccDeviceState occ_cache;
	switch(cache_preference) {
		case cudaFuncCachePreferNone:
			occ_cache.cacheConfig = CACHE_PREFER_NONE;
			break;
		case cudaFuncCachePreferShared:
			occ_cache.cacheConfig = CACHE_PREFER_SHARED;
			break;
		case cudaFuncCachePreferL1:
			occ_cache.cacheConfig = CACHE_PREFER_L1;
			break;
		case cudaFuncCachePreferEqual:
			occ_cache.cacheConfig = CACHE_PREFER_EQUAL;
			break;
		default:;///< should throw error
	}
	int minGridSize;
	//TODO: Make constant?
	int block_size = 32;//NOLINT(readability-magic-numbers) Is block size
	cudaOccMaxPotentialOccupancyBlockSize(&minGridSize, &block_size, &prop, &occ_attribs, &occ_cache, nullptr, smem_bytes);
	return block_size;
}

ExecutionPolicy Cuda::launch_config(std::string kernel_name, int thread_num, bool sync, std::size_t smem_size, cudaStream_t sid) const {
	(void) sid;

	auto* instance = get_instance();
	if(instance->k_func_table.find(kernel_name) == instance->k_func_table.end()) {
		//TODO: Make constant?
		int bs = 256;//NOLINT(readability-magic-numbers) Is block size
		printf("Warning: Kernel function %s not registered! Use 256 setting!\n", kernel_name.data());
		return {general_grid_size(thread_num, bs), bs, smem_size, sync};
	}
	auto& config = instance->k_func_table[kernel_name.data()];
	int bs		 = config.max_occ_block_size;
	if(smem_size > 0) {
		bs = eval_optimal_block_size(config.attribs, config.cache_preference, smem_size);
	}
	// printf("configurating for kernel[%s] blocksize: %d\n", kernel_name.c_str(),
	// bs);
	if(config.wave_fashion) {
		return {wave_grid_size(thread_num, bs), bs, smem_size, sync};
	}
	return {general_grid_size(thread_num, bs), bs, smem_size, sync};
}

void Cuda::register_kernel(std::string tag, KernelFunc f, cudaFuncCache cache_config, bool wave_fashion) {
	auto* instance = get_instance();
	instance->k_func_table.emplace(tag, KernelConfig(f, cache_config, wave_fashion));
	printf("Kernel[%s](%s) block size configuration: %d\n", tag.data(), wave_fashion ? "wave" : "general", instance->k_func_table[tag.data()].max_occ_block_size);
}
const KernelConfig& Cuda::find_kernel(std::string tag) {
	auto* instance = get_instance();
	return instance->k_func_table[tag.data()];
}

//NOLINTEND(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print

}// namespace mn
