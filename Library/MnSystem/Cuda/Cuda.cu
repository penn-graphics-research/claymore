#include "Cuda.h"
#include <cstdio>
#include <utility>

#include <cuda_occupancy.h>    ///<	for optimal kernel launching
#include <cuda_profiler_api.h> ///<	for evaluating kernel performance
#include <cuda_runtime.h>

#define MEM_POOL_CTRL 3

namespace mn {

KernelConfig::KernelConfig(KernelFunc f, cudaFuncCache cacheConfig, bool isWave)
    : func(f), cachePreference(cacheConfig), waveFashion(isWave) {
  cudaFuncGetAttributes(&attribs, f);
  maxOccBlockSize = Cuda::evalOptimalBlockSize(attribs, cachePreference);
  if (cacheConfig != cudaFuncCachePreferNone) ///< should be different from
    ///< device cache preference
    checkCudaErrors(cudaFuncSetCacheConfig(f, cacheConfig));
}

Cuda::Cuda() : _default_devId{0} {
  printf("[Init -- Begin] Cuda\n");
  //< acquire devices
  _dev_num_available = 0;
  cudaError_t error_id = cudaGetDeviceCount(&_dev_num_available);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id,
           cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }
  if (_dev_num_available == 0)
    printf("\t[InitInfo -- DevNum] There are no available device(s) that "
           "support CUDA\n");
  else
    printf("\t[InitInfo -- DevNum] Detected %d CUDA Capable device(s)\n",
           _dev_num_available);
  set_max_device(); //<[TMP]
  printf(
      "\t[InitInfo -- DevNum] Prepare to use %d device(s) in Multi-GPU test\n",
      _dev_num_using);

  _akDeviceProps.resize(_dev_num_available);
  _akStreams.resize(_dev_num_available);
  _akEvents.resize(_dev_num_available);

  for (int i = 0; i < _dev_num_available; i++) {
    // checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaSetDevice(i));
    ///< device properties
    checkCudaErrors(cudaGetDeviceProperties(&_akDeviceProps[i], i));
    const auto &prop{_akDeviceProps[i]};
    printf("\t[InitInfo -- Dev Property] GPU device %d (%d-th group on "
           "board)\n\t\tglobal memory: %llu bytes,\n\t\tshared memory per "
           "block: %llu bytes,\n\t\tregisters per SM: %d,\n\t\tMulti-Processor "
           "count: %d,\n\t\tSM compute capabilities: %d.%d.\n",
           i, prop.multiGpuBoardGroupID,
           (long long unsigned int)prop.totalGlobalMem,
           (long long unsigned int)prop.sharedMemPerBlock, prop.regsPerBlock,
           prop.multiProcessorCount, prop.major, prop.minor);

    ///< streams
    // int leastPriority = 0;
    // int greatestPriority = leastPriority;
    // cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    // cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault,
    // leastPriority); cudaStreamCreateWithPriority(&push_top_stream,
    // cudaStreamDefault, greatestPriority);
    // cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault,
    // greatestPriority);
    // for (auto &stream : _akStreams[i])
    for (int j = 0; j < (int)StreamIndex::Total; ++j)
      checkCudaErrors(cudaStreamCreate(&_akStreams[i][j]));
    printf("\t[InitInfo -- stream] Create %lu streams for device %d\n",
           (long unsigned)_akStreams[i].extent, i);

    // for (auto &event : _akEvents[i])
    for (int j = 0; j < (int)EventIndex::Total; ++j)
      checkCudaErrors(cudaEventCreateWithFlags(
          &_akEvents[i][j], // cudaEventDefault | cudaEventBlockingSync |
          // cudaEventInterprocess |
          cudaEventDisableTiming));

    ///< memory allocator
    std::size_t free_byte, total_byte;
    checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
    /// CRITICAL: not sure what goes wrong with virtual allocator
    _akMonotonicVirtualAllocators.emplace_back(
        std::make_unique<MonotonicVirtualAllocator>(i, prop.textureAlignment,
                                                    1));
    ///
    _akMonotonicAllocators.emplace_back(std::make_unique<MonotonicAllocator>(
        prop.textureAlignment,
        free_byte >> MEM_POOL_CTRL)); ///< preserve 1/4 space for intermediate
    ///< computations
    cudaDeviceSynchronize();
    printf("\t[InitInfo -- memory] device %d\n\t\tfree bytes/total bytes: "
           "%lu/%lu,\n\t\tpre-allocated size: %lu bytes\n\n",
           i, (long unsigned)free_byte, (long unsigned)total_byte,
           (long unsigned)(free_byte >> MEM_POOL_CTRL));
  }

  //< enable peer access
  for (int i = 0; i < _dev_num_available; i++) {
    for (int j = 0; j < _dev_num_available; j++) {
      if (i != j) {
        establishPeerAccess(i, j);
        printf("\t[InitInfo -- Peer Access] Enable peer access from %d to %d\n",
               i, j);
      }
    }
  }
  //< init cuda context
  for (int i = 0; i < _dev_num_available; i++) {
    _akCuDev_contexts.emplace_back(i); //< set device when construct
  }

  printf("\t[InitInfo -- Default Dev] Default context: %d\n", _default_devId);
  checkCudaErrors(cudaSetDevice(_default_devId));
  printf("\n[Init -- End] == Finished \'Cuda\' initialization\n\n");
  // getchar();
}

Cuda::~Cuda() {
  // cudaStreamDestroy(_kMemCopyStream);
  getInstance()->_akDeviceProps.clear();
  for (auto &streams : getInstance()->_akStreams)
    // for (auto &stream : streams)
    for (int j = 0; j < (int)StreamIndex::Total; ++j)
      checkCudaErrors(cudaStreamDestroy(streams[j]));
  for (auto &events : getInstance()->_akEvents)
    // for (auto &event : events)
    for (int j = 0; j < (int)EventIndex::Total; ++j)
      checkCudaErrors(cudaEventDestroy(events[j]));
#if 0
  for (auto &monoAllocator : _akMonotonicAllocators)
    for (int i = 0; i < getInstance()->_akMonotonicAllocators.size(); ++i)
      getInstance()->_akMonotonicAllocators[i].~MonotonicAllocator();
  for (int i = 0; i < getInstance()->_akMonotonicVirtualAllocators.size(); ++i)
    getInstance()
        ->_akMonotonicVirtualAllocators[i]
        .~MonotonicVirtualAllocator();
#endif
  printf("  Finished \'Cuda\' termination\n");
}

int Cuda::generalGridSize(int &threadNum, int &blockSize) const {
  return (threadNum + blockSize - 1) / blockSize;
}
int Cuda::waveGridSize(int &threadNum, int &blockSize) const {
  auto blocksPerSM =
      (threadNum / blockSize /
       getInstance()->_akDeviceProps[_iDevID].multiProcessorCount) *
      getInstance()->_akDeviceProps[_iDevID].multiProcessorCount;
  return blocksPerSM ? blocksPerSM : 1;
}

/// static methods
int Cuda::evalOptimalBlockSize(cudaFuncAttributes attribs,
                               cudaFuncCache cachePreference,
                               std::size_t smemBytes) {
  auto instance = getInstance();
  cudaOccDeviceProp prop =
      getInstance()->_akDeviceProps[instance->_iDevID]; ///< cache preference
  cudaOccFuncAttributes occAttribs = attribs;
  cudaOccDeviceState occCache;
  switch (cachePreference) {
  case cudaFuncCachePreferNone:
    occCache.cacheConfig = CACHE_PREFER_NONE;
    break;
  case cudaFuncCachePreferShared:
    occCache.cacheConfig = CACHE_PREFER_SHARED;
    break;
  case cudaFuncCachePreferL1:
    occCache.cacheConfig = CACHE_PREFER_L1;
    break;
  case cudaFuncCachePreferEqual:
    occCache.cacheConfig = CACHE_PREFER_EQUAL;
    break;
  default:; ///< should throw error
  }
  int minGridSize, blockSize = 32;
  cudaOccMaxPotentialOccupancyBlockSize(&minGridSize, &blockSize, &prop,
                                        &occAttribs, &occCache, nullptr,
                                        smemBytes);
  return blockSize;
}

ExecutionPolicy Cuda::launchConfig(std::string kernelName, int threadNum,
                                   bool sync, std::size_t smemSize,
                                   cudaStream_t sid) const {
  auto instance = getInstance();
  if (instance->_kFuncTable.find(kernelName) == instance->_kFuncTable.end()) {
    int bs = 256;
    printf("Warning: Kernel function %s not registered! Use 256 setting!\n",
           kernelName.data());
    return {generalGridSize(threadNum, bs), bs, smemSize, sync};
  }
  auto &config = instance->_kFuncTable[kernelName.data()];
  int bs = config.maxOccBlockSize;
  if (smemSize > 0)
    bs = evalOptimalBlockSize(config.attribs, config.cachePreference, smemSize);
  // printf("configurating for kernel[%s] blocksize: %d\n", kernelName.c_str(),
  // bs);
  if (config.waveFashion)
    return {waveGridSize(threadNum, bs), bs, smemSize, sync};
  return {generalGridSize(threadNum, bs), bs, smemSize, sync};
}

void Cuda::registerKernel(std::string tag, KernelFunc f,
                          cudaFuncCache cacheConfig, bool waveFashion) {
  auto instance = getInstance();
  instance->_kFuncTable.emplace(tag, KernelConfig(f, cacheConfig, waveFashion));
  printf("Kernel[%s](%s) block size configuration: %d\n", tag.data(),
         waveFashion ? "wave" : "general",
         instance->_kFuncTable[tag.data()].maxOccBlockSize);
}
const KernelConfig &Cuda::findKernel(std::string tag) {
  auto instance = getInstance();
  return instance->_kFuncTable[tag.data()];
}

} // namespace mn
