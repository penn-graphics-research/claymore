#ifndef MEMORY_RESOURCE_H
#define MEMORY_RESOURCE_H
#include "MnBase/Singleton.h"

namespace mn {

template<typename derived_t>
struct MemoryResource {
	using pointer_t = void*;

	pointer_t allocate(std::size_t bytes, std::size_t alignment = alignof(max_align_t)) {
		return static_cast<derived_t&>(*this).do_allocate(bytes, alignment);
	}

	void deallocate(pointer_t ptr, std::size_t bytes = 0, std::size_t alignment = alignof(max_align_t)) {
		static_cast<derived_t&>(*this).do_deallocate(ptr, bytes, alignment);
	}

	bool is_equal(const MemoryResource& other) const noexcept {
		return (this == &other);
	}
};

struct HeapMemoryResource
	: Singleton<HeapMemoryResource>
	, MemoryResource<HeapMemoryResource> {
	void* do_allocate(std::size_t bytes, std::size_t align) {
		// return ::operator new(bytes, std::align_val_t(align));
		return ::operator new(bytes);
	}

	void do_deallocate(void* ptr, std::size_t bytes, std::size_t align) {
		//::operator delete(ptr, bytes, std::align_val_t(align));
		::operator delete(ptr);
	}
};

}// namespace mn

#endif