#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <type_traits>

#include "MemoryResource.h"

namespace mn {

/// stateless allocator
struct HeapAllocator {
	using mr_type = MemoryResource<HeapMemoryResource>;

	HeapAllocator() = default;
	HeapAllocator(const HeapAllocator& o) noexcept {}

	mr_type* resource() const {
		return &HeapMemoryResource::instance();
	}

	void* allocate(std::size_t bytes) {
		return resource()->allocate(bytes);
	}
	void deallocate(void* p, std::size_t bytes) {
		(void) bytes;

		resource()->deallocate(p);
	}
};

/// stateful allocator
template<typename MemoryResource_t>
struct StackAllocator {
	using mr_type = MemoryResource<MemoryResource_t>;

   private:
	mr_type* mr;

   public:
	char* data;
	char* head;
	char* tail;
	std::size_t align;

	StackAllocator() = delete;
	explicit StackAllocator(mr_type* mr, std::size_t align_bytes, std::size_t total_mem_bytes)
		: mr {mr}
		, align {align_bytes} {
		data = head = (char*) (mr->allocate(total_mem_bytes));
		tail		= head + total_mem_bytes;
	};

	~StackAllocator() {
		mr->deallocate((void*) data, (std::size_t)(tail - data));
	}

	mr_type* resource() const noexcept {
		return mr;
	}

	/// learnt from taichi
	void* allocate(std::size_t bytes) {
		/// first align head
		char* ret = head + align - 1 - ((std::size_t) head + align - 1) % align;
		head	  = ret + bytes;
		if(head > tail) {
			throw std::bad_alloc {};
		} else {
			return ret;
		}
	}
	void deallocate(void* p, std::size_t bytes) {
		(void) bytes;

		if(p >= head) {
			throw std::bad_alloc {};
		} else if(p < data) {
			throw std::bad_alloc {};
		}
		head = (char*) p;
	}

	void reset() {
		head = data;
	}
};

template<typename value_t, typename MemoryResource_t>
struct ObjectAllocator {
	using value_type = value_t;
	using mr_type	 = MemoryResource<MemoryResource_t>;

   private:
	mr_type* mr;

   public:
	ObjectAllocator() = delete;
	ObjectAllocator(mr_type* mr)
		: mr {mr} {};

	template<typename other_value_t>
	explicit ObjectAllocator(const ObjectAllocator<other_value_t, MemoryResource_t>& o) noexcept {
		mr = o.resource();
	}

	mr_type* resource() const noexcept {
		return mr;
	}

	value_type* allocate(std::size_t n) {
		return (value_type*) (mr->allocate(n * sizeof(value_type), alignof(value_type)));
	}

	void deallocate(value_type* p, std::size_t n) {
		mr->deallocate((void*) p, n * sizeof(value_type), alignof(value_type));
	}
};

template<std::size_t chunk_size_v, typename MemoryResource_t>
struct PoolAllocator {
	static constexpr std::size_t chunk_size = chunk_size_v;

	using mr_type = MemoryResource<MemoryResource_t>;
};

template<typename MemoryResource_t>
using page_allocator = PoolAllocator<4096, MemoryResource_t>;

template<typename structural_t, typename MemoryResource_t>
struct structural_allocator : PoolAllocator<structural_t::size, MemoryResource_t> {};

/// 4K, 64K, 2M
template<typename chunk_sizes_t, typename MemoryResource_t>
struct multiPoolAllocator;

template<std::size_t... chunk_sizes_v, typename MemoryResource_t>
struct multiPoolAllocator<std::index_sequence<chunk_sizes_v...>, MemoryResource_t> : PoolAllocator<chunk_sizes_v, MemoryResource_t>... {};

}// namespace mn

#endif