#ifndef __CUDA_TIMERS_CUH_
#define __CUDA_TIMERS_CUH_

#include <cuda_runtime_api.h>
#include <fmt/color.h>
#include <fmt/core.h>

namespace mn {

struct CudaTimer {
  using TimeStamp = cudaEvent_t;
  explicit CudaTimer(cudaStream_t sid) : streamId{sid} {
    cudaEventCreate(&last);
    cudaEventCreate(&cur);
  }
  ~CudaTimer() {
    cudaEventDestroy(last);
    cudaEventDestroy(cur);
  }
  void tick() { cudaEventRecord(last, streamId); }
  void tock() { cudaEventRecord(cur, streamId); }
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

private:
  cudaStream_t streamId;
  TimeStamp last, cur;
};

} // namespace mn

#endif