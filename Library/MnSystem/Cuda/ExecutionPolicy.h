#ifndef __CUDA_EXECUTION_POLICY_H_
#define __CUDA_EXECUTION_POLICY_H_
#include <string>
#include <driver_types.h>

namespace mn {

struct LaunchConfig {
    template <typename IndexType0, typename IndexType1>
    LaunchConfig(IndexType0 gs, IndexType1 bs)
        : dg{ static_cast<unsigned int>(gs) }, db{ static_cast<unsigned int>(bs) }, shmem{ 0 }, sid{ cudaStreamDefault }
    {
    }
    template <typename IndexType0, typename IndexType1, typename IndexType2>
    LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem)
        : dg{ static_cast<unsigned int>(gs) }, db{ static_cast<unsigned int>(bs) }, shmem{ static_cast<std::size_t>(mem) }, sid{ cudaStreamDefault }
    {
    }
    template <typename IndexType0, typename IndexType1, typename IndexType2>
    LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem, cudaStream_t stream)
        : dg{ static_cast<unsigned int>(gs) }, db{ static_cast<unsigned int>(bs) }, shmem{ static_cast<std::size_t>(mem) }, sid{ stream }
    {
    }
    dim3 dg{};
    dim3 db{};
    std::size_t shmem{ 0 };
    cudaStream_t sid{ cudaStreamDefault };
};

struct LaunchInput { ///< could contain more information on operation (error checking/ time recording/ etc...)
    LaunchInput() = delete;
    LaunchInput(std::string kernel, int taskNum, std::size_t sharedMemBytes = 0)
        : kernelName(kernel), numThreads(taskNum), sharedMemBytes(sharedMemBytes) {}
    const std::string& name() { return kernelName; }
    const int& threads() { return numThreads; }
    const std::size_t& memBytes() { return sharedMemBytes; }

private:
    const std::string kernelName;
    const int numThreads;
    const std::size_t sharedMemBytes;
};

/// kernel launching configuration
struct ExecutionPolicy {
    ExecutionPolicy() {}
    ExecutionPolicy(int gs, int bs, std::size_t memsize, bool s)
        : gridSize(gs), blockSize(bs), sharedMemBytes(memsize), sync(s) {}
    int getGridSize() const { return gridSize; }
    int getBlockSize() const { return blockSize; }
    std::size_t getSharedMemBytes() const { return sharedMemBytes; }
    bool needSync() const { return sync; }

private:
    int gridSize{ 0 };
    int blockSize{ 0 };
    std::size_t sharedMemBytes{ 0 };
    bool sync{ false };
};

} // namespace mn

#endif