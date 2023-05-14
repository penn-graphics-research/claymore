#ifndef ALLOCATORS_CUH
#define ALLOCATORS_CUH
#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>

#include "HostUtils.hpp"
#include "MnBase/Memory/Allocator.h"

namespace mn {

struct DeviceMemoryResource
	: Singleton<DeviceMemoryResource>
	, MemoryResource<DeviceMemoryResource> {
	static constexpr size_t MAX_ALIGNMENT = 256;

	DeviceMemoryResource() = default;

	void* do_allocate(std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		if(alignment > MAX_ALIGNMENT) {
			throw std::bad_alloc {};
		}
		void* ret;
		if(cudaMalloc((void**) &ret, bytes) != cudaSuccess) {
			throw std::bad_alloc {};
		}
		return ret;
	}

	void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		(void) bytes;
		(void) alignment;

		if(cudaFree(ptr) != cudaSuccess) {
			throw std::bad_alloc {};
		}
	}
};

struct UnifiedMemoryResource
	: Singleton<UnifiedMemoryResource>
	, MemoryResource<UnifiedMemoryResource> {
	static constexpr size_t MAX_ALIGNMENT = 256;

	UnifiedMemoryResource() = default;

	void* do_allocate(std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		if(alignment > MAX_ALIGNMENT) {
			throw std::bad_alloc {};
		}
		void* ret;
		if(cudaMallocManaged((void**) &ret, bytes) != cudaSuccess) {
			throw std::bad_alloc {};
		}
		return ret;
	}

	void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		(void) bytes;
		(void) alignment;

		if(cudaFree(ptr) != cudaSuccess) {
			throw std::bad_alloc {};
		}
	}
};

/// std::allocator
/// stateless allocator
struct DeviceAllocator {
	using mr_type = MemoryResource<DeviceMemoryResource>;

	DeviceAllocator()											  = default;
	DeviceAllocator(const DeviceAllocator& o) noexcept			  = default;
	DeviceAllocator(DeviceAllocator&& o) noexcept				  = default;
	DeviceAllocator& operator=(const DeviceAllocator& o) noexcept = default;
	DeviceAllocator& operator=(DeviceAllocator&& o) noexcept	  = default;

	[[nodiscard]] mr_type* resource() const {
		return &DeviceMemoryResource::instance();
	}

	void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		return resource()->allocate(bytes);
	}
	void deallocate(void* p, std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		(void) bytes;

		resource()->deallocate(p);
	}
};

struct UnifiedAllocator {
	using mr_type = MemoryResource<UnifiedMemoryResource>;

	UnifiedAllocator()												= default;
	UnifiedAllocator(const UnifiedAllocator& o) noexcept			= default;
	UnifiedAllocator(UnifiedAllocator&& o) noexcept					= default;
	UnifiedAllocator& operator=(const UnifiedAllocator& o) noexcept = default;
	UnifiedAllocator& operator=(UnifiedAllocator&& o) noexcept		= default;

	[[nodiscard]] mr_type* resource() const {
		return &UnifiedMemoryResource::instance();
	}

	void* allocate(std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		return resource()->allocate(bytes);
	}
	void deallocate(void* p, std::size_t bytes) {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		(void) bytes;

		resource()->deallocate(p);
	}
};
/// memory types: device only, pageable, pinned
/// usage: static, intermediate, dynamic small, dynamic large, dynamic random
struct MonotonicAllocator : StackAllocator<DeviceMemoryResource> {
	using base_t = StackAllocator<DeviceMemoryResource>;

	MonotonicAllocator(std::size_t texturealign_bytes, std::size_t total_mem_bytes)
		: StackAllocator<DeviceMemoryResource> {&DeviceMemoryResource::instance(), texturealign_bytes, total_mem_bytes} {
		std::cout << std::string {"monotonic allocator alignment (Bytes): "} << texturealign_bytes << std::string {"\tsize (MB): "} << static_cast<double>(total_mem_bytes) / static_cast<double>(1 << 20) << std::endl;//NOLINT(readability-magic-numbers) Unit conversion
	}

	~MonotonicAllocator() = default;

	MonotonicAllocator(const MonotonicAllocator& o) noexcept			= default;
	MonotonicAllocator(MonotonicAllocator&& o) noexcept					= default;
	MonotonicAllocator& operator=(const MonotonicAllocator& o) noexcept = default;
	MonotonicAllocator& operator=(MonotonicAllocator&& o) noexcept		= default;

	auto borrow(std::size_t bytes) -> void* {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		return allocate(bytes);
	}

	void reset() {
		std::size_t used_bytes	= head - data;
		std::size_t total_bytes = tail - data;
		if(used_bytes >= total_bytes * 3 / 4) {//NOLINT(readability-magic-numbers) Threshold values?
			base_t::resource()->deallocate((void*) this->data, total_bytes);
			std::size_t total_mem_bytes = total_bytes * 3 / 2;//NOLINT(readability-magic-numbers) Threshold values?
			this->data					= static_cast<char*>(base_t::resource()->allocate(total_mem_bytes));
			this->head					= this->data;
			this->tail					= this->data + total_mem_bytes;
		} else {
			this->head = this->data;
		}
	}
};

struct MonotonicVirtualAllocator : StackAllocator<UnifiedMemoryResource> {
	using base_t				 = StackAllocator<DeviceMemoryResource>;
	~MonotonicVirtualAllocator() = default;

	MonotonicVirtualAllocator(int devId, std::size_t texturealign_bytes, std::size_t total_mem_bytes)
		: StackAllocator<UnifiedMemoryResource> {&UnifiedMemoryResource::instance(), texturealign_bytes, total_mem_bytes} {
		// checkCudaErrors(cudaMemAdvise(addr, total_mem_bytes,
		//                              cudaMemAdviseSetPreferredLocation, devId));
		std::cout << std::string {"monotonic virtual allocator alignment (Bytes): "} << texturealign_bytes << std::string {"\tsize (MB): "} << static_cast<double>(total_mem_bytes) / static_cast<double>(1 << 20) << std::endl;//NOLINT(readability-magic-numbers) Unit conversion
	}

	auto borrow(std::size_t bytes) -> void* {//NOLINT(readability-convert-member-functions-to-static) Method is designed to be a non-static class member
		return allocate(bytes);
	}

	void reset() {
		std::size_t used_bytes	= head - data;
		std::size_t total_bytes = tail - data;
		if(used_bytes >= total_bytes * 3 / 4) {//NOLINT(readability-magic-numbers) Threshold values?
			this->resource()->deallocate((void*) this->data, total_bytes);
			std::size_t total_mem_bytes = total_bytes * 3 / 2;//NOLINT(readability-magic-numbers) Threshold values?
			this->data					= static_cast<char*>(this->resource()->allocate(total_mem_bytes));
			this->head					= this->data;
			this->tail					= this->data + total_mem_bytes;
		} else {
			this->head = this->data;
		}
	}
};

}// namespace mn

#endif