#ifndef __ALLOCATORS_CUH_
#define __ALLOCATORS_CUH_
#include "HostUtils.hpp"
#include <MnBase/Memory/Allocator.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

namespace mn {

struct device_memory_resource : Singleton<device_memory_resource>,
                                memory_resource<device_memory_resource> {
  device_memory_resource() = default;
  void *do_allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment > 256)
      throw std::bad_alloc{};
    void *ret;
    if (cudaMalloc((void **)&ret, bytes) != cudaSuccess)
      throw std::bad_alloc{};
    return ret;
  }
  void do_deallocate(void *ptr, std::size_t, std::size_t) {
    if (cudaFree(ptr) != cudaSuccess)
      throw std::bad_alloc{};
  }
};

struct unified_memory_resource : Singleton<unified_memory_resource>,
                                 memory_resource<unified_memory_resource> {
  unified_memory_resource() = default;
  void *do_allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment > 256)
      throw std::bad_alloc{};
    void *ret;
    if (cudaMallocManaged((void **)&ret, bytes) != cudaSuccess)
      throw std::bad_alloc{};
    return ret;
  }
  void do_deallocate(void *ptr, std::size_t, std::size_t) {
    if (cudaFree(ptr) != cudaSuccess)
      throw std::bad_alloc{};
  }
};

/// std::allocator
/// stateless allocator
struct device_allocator {
  using mr_type = memory_resource<device_memory_resource>;
  device_allocator() = default;
  device_allocator(const device_allocator &o) noexcept {}
  mr_type *resource() const { return &device_memory_resource::instance(); }

  void *allocate(std::size_t bytes) { return resource()->allocate(bytes); }
  void deallocate(void *p, std::size_t) { resource()->deallocate(p); }
};

struct unified_allocator {
  using mr_type = memory_resource<unified_memory_resource>;
  unified_allocator() = default;
  unified_allocator(const unified_allocator &o) noexcept {}
  mr_type *resource() const { return &unified_memory_resource::instance(); }

  void *allocate(std::size_t bytes) { return resource()->allocate(bytes); }
  void deallocate(void *p, std::size_t) { resource()->deallocate(p); }
};
/// memory types: device only, pageable, pinned
/// usage: static, intermediate, dynamic small, dynamic large, dynamic random
struct MonotonicAllocator : stack_allocator<device_memory_resource> {
  using base_t = stack_allocator<device_memory_resource>;
  ~MonotonicAllocator() = default;

  MonotonicAllocator(std::size_t textureAlignBytes, std::size_t totalMemBytes)
      : stack_allocator<device_memory_resource>{
            &device_memory_resource::instance(), textureAlignBytes,
            totalMemBytes} {
    std::cout << std::string{"monotonic allocator alignment (Bytes): "}
              << textureAlignBytes << std::string{"\tsize (MB): "}
              << totalMemBytes / 1024.0 / 1024.0 << std::endl;
  }

  auto borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      base_t::resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(base_t::resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }
};

struct MonotonicVirtualAllocator : stack_allocator<unified_memory_resource> {
  using base_t = stack_allocator<device_memory_resource>;
  ~MonotonicVirtualAllocator() = default;

  MonotonicVirtualAllocator(int devId, std::size_t textureAlignBytes,
                            std::size_t totalMemBytes)
      : stack_allocator<unified_memory_resource>{
            &unified_memory_resource::instance(), textureAlignBytes,
            totalMemBytes} {
    // checkCudaErrors(cudaMemAdvise(addr, totalMemBytes,
    //                              cudaMemAdviseSetPreferredLocation, devId));
    std::cout << std::string{"monotonic virtual allocator alignment (Bytes): "}
              << textureAlignBytes << std::string{"\tsize (MB): "}
              << totalMemBytes / 1024.0 / 1024.0 << std::endl;
  }

  auto borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      this->resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(this->resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }
};

} // namespace mn

#endif