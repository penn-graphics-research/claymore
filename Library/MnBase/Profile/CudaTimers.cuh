#ifndef CUDA_TIMERS_CUH
#define CUDA_TIMERS_CUH

#include <cuda_runtime_api.h>
#include <fmt/color.h>
#include <fmt/core.h>

namespace mn {

struct CudaTimer {
	using TimeStamp = cudaEvent_t;

   private:
	cudaStream_t stream_id;
	TimeStamp last;
	TimeStamp cur;

   public:
	explicit CudaTimer(cudaStream_t sid)
		: stream_id {sid} {
		cudaEventCreate(&last);
		cudaEventCreate(&cur);
	}

	~CudaTimer() {
		cudaEventDestroy(last);
		cudaEventDestroy(cur);
	}

	void tick() {
		cudaEventRecord(last, stream_id);
	}

	void tock() {
		cudaEventRecord(cur, stream_id);
	}

	float elapsed() {
		float duration;
		cudaEventSynchronize(cur);
		cudaEventElapsedTime(&duration, last, cur);
		return duration;
	}

	void tock(std::string tag) {
		tock();
		fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag.c_str(), elapsed());
	}
};

}// namespace mn

#endif