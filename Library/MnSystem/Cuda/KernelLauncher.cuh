#ifndef __KERNEL_LAUNCHER_CUH_
#define __KERNEL_LAUNCHER_CUH_

#include "ExecutionPolicy.h"
#include <MnUtility/Profiler/Performance/CudaTimer.cuh>

namespace mn {

	template <typename Func, typename... Arguments>
	void cuda_launch(LaunchConfig&& lc, Func&& f, Arguments... args) {	///< launch on the current device
		/// compiler will handle type conversions
		std::forward<Func>(f) <<<lc.dg, lc.db, lc.shmem, lc.sid >>> (std::forward<Arguments>(args)...);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) printf("Kernel launch failure %s\n", cudaGetErrorString(error));
	}

	/// backup
	template <typename... Arguments>
	void cuda_launch(LaunchConfig&& lc, void(*f)(Arguments...), Arguments... args) {	///< launch on the current device
		/// compiler will handle type conversions
		f << <lc.dg, lc.db, lc.shmem, lc.sid >> > (std::forward<Arguments>(args)...);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) printf("Kernel launch failure %s\n", cudaGetErrorString(error));
	}

	/// option: 1 error checking 2 execution time recording 3 synchronizing
	template <typename... Arguments>
	void debugLaunch(int gs, int bs, void(*f)(Arguments...), Arguments... args) {
		cudaError_t error;
		f << <gs, bs >> >(std::forward<Arguments>(args)...);
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess) printf("Kernel launch failure %s\n", cudaGetErrorString(error));
	}

	template <typename... Arguments>
	void recordLaunch(std::string&& tag, int gs, int bs, std::size_t mem, void(*f)(Arguments...), Arguments... args) {
		CudaTimer timer;
		if (!mem) {
			timer.tick();
			f << <gs, bs >> > (std::forward<Arguments>(args)...);
			timer.tock(tag);
		} else {
			timer.tick();
			f << <gs, bs, mem >> > (std::forward<Arguments>(args)...);
			timer.tock(tag);
		}
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) printf("Kernel launch failure %s\n", cudaGetErrorString(error));
	}

	template <typename... Arguments>
	void cleanLaunch(int gs, int bs, void(*f)(Arguments...), Arguments... args) {
		f << <gs, bs >> >(args...);
	}

}

#endif