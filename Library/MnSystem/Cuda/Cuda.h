#ifndef SYSTEM_CUDA_H
#define SYSTEM_CUDA_H

#include <string>
//#include <helper_cuda.h>
#include <driver_types.h>

#include <unordered_map>

#include "Allocators.cuh"
#include "ExecutionPolicy.h"
#include "HostUtils.hpp"
#include "MnBase/Math/Vec.h"
#include "MnBase/Meta/AllocMeta.cuh"
#include "MnBase/Object/Function.h"
#include "MnBase/Singleton.h"

namespace mn {

using KernelFunc = const void*;

struct KernelConfig {///< static kernel attrib, could contain run-time debugger
	///< setting(error checking/ time recording etc...)
	KernelFunc func;
	cudaFuncAttributes attribs;
	cudaFuncCache cache_preference;
	bool wave_fashion;	   ///< general fashion or loop fashion
	int max_occ_block_size;///< condition: use no shared memory
	explicit KernelConfig(KernelFunc f = nullptr, cudaFuncCache cache_config = cudaFuncCachePreferNone, bool is_wave = false);
};

class Cuda : public ManagedSingleton<Cuda> {
   public:
	/// kernel launching
	enum class StreamIndex {
		COMPUTE = 0,
		H2DCOPY,
		D2HCOPY,
		D2DCOPY,
		SPARE,
		TOTAL = 32
	};

	enum class EventIndex {
		COMPUTE = 0,
		H2DCOPY,
		D2HCOPY,
		D2DCOPY,
		SPARE,
		TOTAL = 32
	};

	struct CudaContext {
	   private:
		int dev_id;

		auto monotonicAllocator() -> MonotonicAllocator& {
			// setContext();
			return *Cuda::get_instance()->ak_monotonic_allocators[dev_id];
		}

	   public:
		explicit CudaContext(int dev_id = -1)
			: dev_id {dev_id} {
			if(dev_id != -1) {
				printf("\t[Init] CudaContext %d\n", dev_id);
				check_cuda_errors(cudaSetDevice(dev_id));
			}
		}

		//< context & prop
		void set_context() {
			check_cuda_errors(cudaSetDevice(dev_id));
		}

		auto get_dev_id() const noexcept {
			return dev_id;
		}

		auto get_context_info() noexcept {
			return dev_id;
		}

		const auto& get_dev_prop() const noexcept {
			return Cuda::get_instance()->ak_device_props[dev_id];
		}

		/// stream & event
		// stream
		template<StreamIndex sid>
		auto stream() const -> cudaStream_t {
			return Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(sid)];
		}

		auto stream(unsigned sid) const -> cudaStream_t {
			return Cuda::get_instance()->ak_streams[dev_id][sid];
		}

		auto stream_compute() const -> cudaStream_t {
			return Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(StreamIndex::COMPUTE)];
		}

		auto stream_spare(unsigned sid = 0) const -> cudaStream_t {
			return Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(StreamIndex::SPARE) + sid];
		}

		void syncCompute() const {
			check_cuda_errors(cudaStreamSynchronize(Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(StreamIndex::COMPUTE)]));
		}

		template<StreamIndex sid>
		void syncStream() const {
			check_cuda_errors(cudaStreamSynchronize(Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(sid)]));
		}

		void syncStream(unsigned sid) const {
			check_cuda_errors(cudaStreamSynchronize(Cuda::get_instance()->ak_streams[dev_id][sid]));
		}

		void syncStreamSpare(unsigned sid = 0) const {
			check_cuda_errors(cudaStreamSynchronize(Cuda::get_instance()->ak_streams[dev_id][static_cast<unsigned int>(StreamIndex::SPARE) + sid]));
		}

		// event
		auto event_compute() const -> cudaEvent_t {
			return Cuda::get_instance()->ak_events[dev_id][static_cast<unsigned int>(EventIndex::COMPUTE)];
		}

		auto event_spare(unsigned eid = 0) const -> cudaEvent_t {
			return Cuda::get_instance()->ak_events[dev_id][static_cast<unsigned int>(EventIndex::SPARE) + eid];
		}

		auto compute_event_record() {
			check_cuda_errors(cudaEventRecord(event_compute(), stream_compute()));
		}

		auto spare_event_record(unsigned id = 0) {
			check_cuda_errors(cudaEventRecord(event_spare(id), stream_spare(id)));
		}

		void computeStreamWaitForEvent(cudaEvent_t event) {
			check_cuda_errors(cudaStreamWaitEvent(stream_compute(), event, 0));
		}

		void spareStreamWaitForEvent(unsigned sid, cudaEvent_t event) {
			check_cuda_errors(cudaStreamWaitEvent(stream_spare(sid), event, 0));
		}

		/// kernel launch
		///< 1. compute stream
		template<typename Func, typename... Arguments>
		void compute_launch(
			LaunchConfig&& lc,
			Func&& f,
			Arguments... args
		) {///< launch on the current device
			static_assert(!std::disjunction<std::is_reference<Arguments>...>::value, "Cannot pass values to Cuda kernels by reference");
			/// compiler will handle type conversions
			if(lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
				std::forward<Func>(f)<<<lc.dg, lc.db, lc.shmem, stream<StreamIndex::COMPUTE>()>>>(args...);
				cudaError_t error = cudaGetLastError();
				if(error != cudaSuccess) {
					printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n", dev_id, cudaGetErrorString(error));
				}
			}
		}

		template<typename... Arguments>
		void compute_launch(
			LaunchConfig&& lc,
			void (*f)(Arguments...),
			Arguments... args
		) {///< launch on the current device
			static_assert(!std::disjunction<std::is_reference<Arguments>...>::value, "Cannot pass values to Cuda kernels by reference");
			/// compiler will handle type conversions
			if(lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
				f<<<lc.dg, lc.db, lc.shmem, stream<StreamIndex::COMPUTE>()>>>(args...);
				cudaError_t error = cudaGetLastError();
				if(error != cudaSuccess) {
					printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n", dev_id, cudaGetErrorString(error));
				}
				// if(error!= cudaSuccess) getchar();
			}
		}

		template<typename Func, typename... Arguments>
		void spare_launch(
			unsigned sid,
			LaunchConfig&& lc,
			Func&& f,
			Arguments... args
		) {///< launch on the current device
			static_assert(!std::disjunction<std::is_reference<Arguments>...>::value, "Cannot pass values to Cuda kernels by reference");
			/// compiler will handle type conversions
			if(lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
				std::forward<Func>(f)<<<lc.dg, lc.db, lc.shmem, stream_spare(sid)>>>(std::forward<Arguments>(args)...);
				cudaError_t error = cudaGetLastError();
				if(error != cudaSuccess) {
					printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n", dev_id, cudaGetErrorString(error));
				}
			}
		}

		template<typename... Arguments>
		void spare_launch(
			unsigned sid,
			LaunchConfig&& lc,
			void (*f)(Arguments...),
			Arguments... args
		) {///< launch on the current device
			static_assert(!std::disjunction<std::is_reference<Arguments>...>::value, "Cannot pass values to Cuda kernels by reference");
			/// compiler will handle type conversions
			if(lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
				f<<<lc.dg, lc.db, lc.shmem, stream_spare(sid)>>>(std::forward<Arguments>(args)...);
				cudaError_t error = cudaGetLastError();
				if(error != cudaSuccess) {
					printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n", dev_id, cudaGetErrorString(error));
				}
				// if(error!= cudaSuccess) getchar();
			}
		}

		/// 2. general stream
		// template <StreamIndex sid, typename Func, typename... Arguments>
		// void general_launch(LaunchConfig&& lc, Func&& f, Arguments... args) {
		//     std::forward<Func>(f) <<<lc.dg, lc.db, lc.shmem, stream<sid>() >>>
		//     (std::forward<Arguments>(args)...); cudaError_t error =
		//     cudaGetLastError(); if (error != cudaSuccess) printf("[Dev %d] Kernel
		//     launch failure on [stream %lu] %s\n", dev_id, (unsigned long)sid,
		//     cudaGetErrorString(error));
		// }
		template<StreamIndex sid, typename... Arguments>
		void general_launch(
			LaunchConfig&& lc,
			void (*f)(Arguments...),
			Arguments... args
		) {///< launch on the current device
			static_assert(!std::disjunction<std::is_reference<Arguments>...>::value, "Cannot pass values to Cuda kernels by reference");
			if(lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
				f<<<lc.dg, lc.db, lc.shmem, stream<sid>()>>>(std::forward<Arguments>(args)...);
				cudaError_t error = cudaGetLastError();
				if(error != cudaSuccess) {
					printf("[Dev %d] Kernel launch failure on [stream %lu] %s\n", dev_id, (unsigned long) sid, cudaGetErrorString(error));
				}
			}
		}

		///< 3. pre-allocated memory
		auto borrow(std::size_t bytes) -> void* {
			return monotonicAllocator().borrow(bytes);
		}

		void reset_mem() {
			monotonicAllocator().reset();
		}

		///< 4. cuda memset

		template<typename Type, typename Integer, typename Attrib>
		void memset(StreamIndex sid, const Integer size, Attrib addr, Type value) {
			// setContext();
			check_cuda_errors(cudaMemsetAsync(addr, value, sizeof(Type) * size, stream((unsigned) sid)));
		}
	};//< [end] struct CudaContext

   private:
	int dev_num_using;
	int dev_num_available;

	int default_dev_id;
	std::vector<CudaContext> ak_cu_dev_contexts;

	std::vector<cudaDeviceProp> ak_device_props;
	std::vector<vec<cudaStream_t, (int) StreamIndex::TOTAL>> ak_streams;///< 16 is enough for most needs
	std::vector<vec<cudaEvent_t, (int) EventIndex::TOTAL>> ak_events;
	std::vector<std::unique_ptr<MonotonicAllocator>> ak_monotonic_allocators;///< an allocator is a handle to a heap

	std::unordered_map<std::string, KernelConfig> k_func_table;

	int i_dev_id;///< need changing

   public:
	Cuda();
	~Cuda();

	static void register_kernel(std::string tag, KernelFunc f, cudaFuncCache cache_config = cudaFuncCachePreferL1, bool wave_fashion = true);
	static const KernelConfig& find_kernel(std::string tag);

	int general_grid_size(int& thread_num, int& block_size) const;
	int wave_grid_size(int& thread_num, int& block_size) const;
	static int eval_optimal_block_size(cudaFuncAttributes attribs, cudaFuncCache cache_preference, size_t smem_bytes = 0);
	ExecutionPolicy launch_config(std::string kernel_name, int thread_num, bool sync = false, size_t smem_size = 0, cudaStream_t sid = cudaStreamDefault) const;

	auto establish_peer_access(int devA, int devB) {
		check_cuda_errors(cudaSetDevice(devA));
		int canAccessPeer = 0;
		check_cuda_errors(cudaDeviceCanAccessPeer(&canAccessPeer, devA, devB));
		if(canAccessPeer) {
			check_cuda_errors(cudaDeviceEnablePeerAccess(devB, 0));
			// cudaSetDevice(i_dev_id);
			return true;
		}
		// cudaSetDevice(i_dev_id);
		return false;
	}

	//< dev_num info
	auto dev_using_count() noexcept {
		return dev_num_using;
	}
	auto dev_available_count() noexcept {
		return dev_num_available;
	}

	void set_max_device() noexcept {
		dev_num_using = dev_num_available;
	}

	//< context ref
	int get_default_dev_id() noexcept {
		return default_dev_id;
	}

	static auto get_device_count() noexcept -> int {
		return get_instance()->dev_num_available;
	}
	static auto ref_cuda_context(int dev_id) noexcept -> CudaContext& {
		return get_instance()->ak_cu_dev_contexts[dev_id];
	}
};

}// namespace mn

#endif