#ifndef __MEMORY_RESOURCE_H_
#define __MEMORY_RESOURCE_H_
#include <MnBase/Singleton.h>

namespace mn {

template <typename derived_t> struct memory_resource {
  using pointer_t = void *;
  pointer_t allocate(std::size_t bytes,
                     std::size_t alignment = alignof(max_align_t)) {
    return static_cast<derived_t &>(*this).do_allocate(bytes, alignment);
  }
  void deallocate(pointer_t ptr, std::size_t bytes = 0,
                  std::size_t alignment = alignof(max_align_t)) {
    static_cast<derived_t &>(*this).do_deallocate(ptr, bytes, alignment);
  }
  bool is_equal(const memory_resource &other) const noexcept {
    return (this == &other);
  }
};

struct heap_memory_resource : Singleton<heap_memory_resource>,
                              memory_resource<heap_memory_resource> {
  void *do_allocate(std::size_t bytes, std::size_t align) {
    // return ::operator new(bytes, std::align_val_t(align));
    return ::operator new(bytes);
  }
  void do_deallocate(void *ptr, std::size_t bytes, std::size_t align) {
    //::operator delete(ptr, bytes, std::align_val_t(align));
    ::operator delete(ptr);
  }
};

} // namespace mn

#endif