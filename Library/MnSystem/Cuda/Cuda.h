#ifndef __SYSTEM_CUDA_H_
#define __SYSTEM_CUDA_H_

#include <string>
//#include <helper_cuda.h>
#include <driver_types.h>
#include <unordered_map>

#include <MnBase/Math/Vec.h>

#include <MnBase/Meta/AllocMeta.cuh>
#include <MnBase/Object/Function.h>
#include <MnBase/Singleton.h>

#include "Allocators.cuh"
#include "ExecutionPolicy.h"
#include "HostUtils.hpp"

namespace mn {

using KernelFunc = const void *;

struct KernelConfig { ///< static kernel attrib, could contain run-time debugger
  ///< setting(error checking/ time recording etc...)
  KernelFunc func;
  cudaFuncAttributes attribs;
  cudaFuncCache cachePreference;
  bool waveFashion;    ///< general fashion or loop fashion
  int maxOccBlockSize; ///< condition: use no shared memory
  explicit KernelConfig(KernelFunc f = nullptr,
                        cudaFuncCache cacheConfig = cudaFuncCachePreferNone,
                        bool isWave = false);
};

class Cuda : public ManagedSingleton<Cuda> {
public:
  Cuda();
  ~Cuda();

  /// kernel launching
  enum class StreamIndex {
    Compute = 0,
    H2DCopy,
    D2HCopy,
    D2DCopy,
    Spare,
    Total = 32
  };
  enum class EventIndex {
    Compute = 0,
    H2DCopy,
    D2HCopy,
    D2DCopy,
    Spare,
    Total = 32
  };

  static void registerKernel(std::string tag, KernelFunc f,
                             cudaFuncCache cacheConfig = cudaFuncCachePreferL1,
                             bool waveFashion = true);
  static const KernelConfig &findKernel(std::string name);

  int generalGridSize(int &threadNum, int &blockSize) const;
  int waveGridSize(int &threadNum, int &blockSize) const;
  static int evalOptimalBlockSize(cudaFuncAttributes attribs,
                                  cudaFuncCache cachePreference,
                                  size_t smemBytes = 0);
  ExecutionPolicy launchConfig(std::string kernelName, int threadNum,
                               bool sync = false, size_t smemSize = 0,
                               cudaStream_t sid = cudaStreamDefault) const;

  struct CudaContext {
    CudaContext(int devId = -1) : _devId{devId} {
      if (devId != -1) {
        printf("\t[Init] CudaContext %d\n", _devId);
        checkCudaErrors(cudaSetDevice(devId));
      }
    }
    //< context & prop
    void setContext() { checkCudaErrors(cudaSetDevice(_devId)); }
    auto getDevId() const noexcept { return _devId; }
    auto getContextInfo() noexcept { return _devId; }
    const auto &getDevProp() const noexcept {
      return Cuda::getInstance()->_akDeviceProps[_devId];
    }

    /// stream & event
    // stream
    template <StreamIndex sid> auto stream() const -> cudaStream_t {
      return Cuda::getInstance()
          ->_akStreams[_devId][static_cast<unsigned int>(sid)];
    }
    auto stream(unsigned sid) const -> cudaStream_t {
      return Cuda::getInstance()->_akStreams[_devId][sid];
    }
    auto stream_compute() const -> cudaStream_t {
      return Cuda::getInstance()
          ->_akStreams[_devId][static_cast<unsigned int>(StreamIndex::Compute)];
    }
    auto stream_spare(unsigned sid = 0) const -> cudaStream_t {
      return Cuda::getInstance()
          ->_akStreams[_devId]
                      [static_cast<unsigned int>(StreamIndex::Spare) + sid];
    }

    void syncCompute() const {
      checkCudaErrors(cudaStreamSynchronize(
          Cuda::getInstance()->_akStreams[_devId][static_cast<unsigned int>(
              StreamIndex::Compute)]));
    }
    template <StreamIndex sid> void syncStream() const {
      checkCudaErrors(cudaStreamSynchronize(
          Cuda::getInstance()
              ->_akStreams[_devId][static_cast<unsigned int>(sid)]));
    }
    void syncStream(unsigned sid) const {
      checkCudaErrors(
          cudaStreamSynchronize(Cuda::getInstance()->_akStreams[_devId][sid]));
    }
    void syncStreamSpare(unsigned sid = 0) const {
      checkCudaErrors(cudaStreamSynchronize(
          Cuda::getInstance()->_akStreams
              [_devId][static_cast<unsigned int>(StreamIndex::Spare) + sid]));
    }

    // event
    auto event_compute() const -> cudaEvent_t {
      return Cuda::getInstance()
          ->_akEvents[_devId][static_cast<unsigned int>(EventIndex::Compute)];
    }
    auto event_spare(unsigned eid = 0) const -> cudaEvent_t {
      return Cuda::getInstance()
          ->_akEvents[_devId]
                     [static_cast<unsigned int>(EventIndex::Spare) + eid];
    }
    auto compute_event_record() {
      checkCudaErrors(cudaEventRecord(event_compute(), stream_compute()));
    }
    auto spare_event_record(unsigned id = 0) {
      checkCudaErrors(cudaEventRecord(event_spare(id), stream_spare(id)));
    }
    void computeStreamWaitForEvent(cudaEvent_t event) {
      checkCudaErrors(cudaStreamWaitEvent(stream_compute(), event, 0));
    }
    void spareStreamWaitForEvent(unsigned sid, cudaEvent_t event) {
      checkCudaErrors(cudaStreamWaitEvent(stream_spare(sid), event, 0));
    }

    /// kernel launch
    ///< 1. compute stream
    template <typename Func, typename... Arguments>
    void compute_launch(LaunchConfig &&lc, Func &&f,
                        Arguments... args) { ///< launch on the current device
      /// compiler will handle type conversions
      if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
        std::forward<Func>(
            f)<<<lc.dg, lc.db, lc.shmem, stream<StreamIndex::Compute>()>>>(
            args...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n",
                 _devId, cudaGetErrorString(error));
      }
    }
    template <typename... Arguments>
    void compute_launch(LaunchConfig &&lc, void (*f)(Arguments...),
                        Arguments... args) { ///< launch on the current device
      /// compiler will handle type conversions
      if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
        f<<<lc.dg, lc.db, lc.shmem, stream<StreamIndex::Compute>()>>>(args...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n",
                 _devId, cudaGetErrorString(error));
        // if(error!= cudaSuccess) getchar();
      }
    }
    template <typename Func, typename... Arguments>
    void spare_launch(unsigned sid, LaunchConfig &&lc, Func &&f,
                      Arguments... args) { ///< launch on the current device
      /// compiler will handle type conversions
      if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
        std::forward<Func>(f)<<<lc.dg, lc.db, lc.shmem, stream_spare(sid)>>>(
            std::forward<Arguments>(args)...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n",
                 _devId, cudaGetErrorString(error));
      }
    }
    template <typename... Arguments>
    void spare_launch(unsigned sid, LaunchConfig &&lc, void (*f)(Arguments...),
                      Arguments... args) { ///< launch on the current device
      /// compiler will handle type conversions
      if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
        f<<<lc.dg, lc.db, lc.shmem, stream_spare(sid)>>>(
            std::forward<Arguments>(args)...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n",
                 _devId, cudaGetErrorString(error));
        // if(error!= cudaSuccess) getchar();
      }
    }
    /// 2. general stream
    // template <StreamIndex sid, typename Func, typename... Arguments>
    // void general_launch(LaunchConfig&& lc, Func&& f, Arguments... args) {
    //     std::forward<Func>(f) <<<lc.dg, lc.db, lc.shmem, stream<sid>() >>>
    //     (std::forward<Arguments>(args)...); cudaError_t error =
    //     cudaGetLastError(); if (error != cudaSuccess) printf("[Dev %d] Kernel
    //     launch failure on [stream %lu] %s\n", _devId, (unsigned long)sid,
    //     cudaGetErrorString(error));
    // }
    template <StreamIndex sid, typename... Arguments>
    void general_launch(LaunchConfig &&lc, void (*f)(Arguments...),
                        Arguments... args) { ///< launch on the current device
      if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
        f<<<lc.dg, lc.db, lc.shmem, stream<sid>()>>>(
            std::forward<Arguments>(args)...);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          printf("[Dev %d] Kernel launch failure on [stream %lu] %s\n", _devId,
                 (unsigned long)sid, cudaGetErrorString(error));
      }
    }

    ///< 3. pre-allocated memory
    auto borrow(std::size_t bytes) -> void * {
      return monotonicAllocator().borrow(bytes);
    }
    void resetMem() { monotonicAllocator().reset(); }

    ///< 4. pre-allocated virtual memory
    auto borrowVirtual(std::size_t bytes) -> void * {
      return monotonicVirtualAllocator().borrow(bytes);
    }
    void resetVirtualMem() { monotonicVirtualAllocator().reset(); }

  private:
    auto monotonicAllocator() -> MonotonicAllocator & {
      // setContext();
      return *Cuda::getInstance()->_akMonotonicAllocators[_devId];
    }
    auto monotonicVirtualAllocator() -> MonotonicVirtualAllocator & {
      // setContext();
      return *Cuda::getInstance()->_akMonotonicVirtualAllocators[_devId];
    }

    ///< 4. cuda memset
  public:
    template <typename Type, typename Integer, typename Attrib>
    void memset(StreamIndex sid, const Integer size, Attrib addr, Type value) {
      // setContext();
      checkCudaErrors(cudaMemsetAsync(addr, value, sizeof(Type) * size,
                                      stream((unsigned)sid)));
    }

  private:
    int _devId;
  }; //< [end] struct CudaContext

  auto establishPeerAccess(int devA, int devB) {
    checkCudaErrors(cudaSetDevice(devA));
    int canAccessPeer = 0;
    checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, devA, devB));
    if (canAccessPeer) {
      checkCudaErrors(cudaDeviceEnablePeerAccess(devB, 0));
      // cudaSetDevice(_iDevID);
      return true;
    }
    // cudaSetDevice(_iDevID);
    return false;
  }

  //< dev_num info
  auto dev_using_count() noexcept { return _dev_num_using; }
  auto dev_available_count() noexcept { return _dev_num_available; }

  void set_max_device() noexcept { _dev_num_using = _dev_num_available; }

  //< context ref
  auto &refCudaContext(int devId) noexcept { return _akCuDev_contexts[devId]; }
  auto &refDefaultContext() noexcept {
    return _akCuDev_contexts[_default_devId];
  }
  int getDefaultDevId() noexcept { return _default_devId; }

  static auto get_device_count() noexcept -> int {
    return getInstance()->_dev_num_available;
  }
  static auto ref_cuda_context(int devId) noexcept -> CudaContext & {
    return getInstance()->_akCuDev_contexts[devId];
  }

private:
  int _dev_num_using;
  int _dev_num_available;

  int _default_devId;
  std::vector<CudaContext> _akCuDev_contexts;

  std::vector<cudaDeviceProp> _akDeviceProps;
  std::vector<vec<cudaStream_t, (int)StreamIndex::Total>>
      _akStreams; ///< 16 is enough for most needs
  std::vector<vec<cudaEvent_t, (int)EventIndex::Total>> _akEvents;
  std::vector<std::unique_ptr<MonotonicAllocator>>
      _akMonotonicAllocators; ///< an allocator is a handle to a heap
  std::vector<std::unique_ptr<MonotonicVirtualAllocator>>
      _akMonotonicVirtualAllocators;

  std::unordered_map<std::string, KernelConfig> _kFuncTable;

  int _iDevID; ///< need changing
};

} // namespace mn

#endif