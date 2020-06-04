#ifndef __ALLOCATOR_H_
#define __ALLOCATOR_H_

#include "MemoryResource.h"
#include <type_traits>

namespace mn {

/// stateless allocator
struct heap_allocator {
  using mr_type = memory_resource<heap_memory_resource>;
  heap_allocator() = default;
  heap_allocator(const heap_allocator &o) noexcept {}
  mr_type *resource() const { return &heap_memory_resource::instance(); }

  void *allocate(std::size_t bytes) { return resource()->allocate(bytes); }
  void deallocate(void *p, std::size_t) { resource()->deallocate(p); }
};

/// stateful allocator
template <typename memory_resource_t> struct stack_allocator {
  using mr_type = memory_resource<memory_resource_t>;

  explicit stack_allocator(mr_type *mr, std::size_t alignBytes,
                           std::size_t totalMemBytes)
      : _mr{mr}, _align{alignBytes} {
    _data = _head = (char *)(_mr->allocate(totalMemBytes));
    _tail = _head + totalMemBytes;
  };
  stack_allocator() = delete;
  ~stack_allocator() {
    _mr->deallocate((void *)_data, (std::size_t)(_tail - _data));
  }

  mr_type *resource() const noexcept { return _mr; }

  /// learnt from taichi
  void *allocate(std::size_t bytes) {
    /// first align head
    char *ret = _head + _align - 1 - ((std::size_t)_head + _align - 1) % _align;
    _head = ret + bytes;
    if (_head > _tail)
      throw std::bad_alloc{};
    else
      return ret;
  }
  void deallocate(void *p, std::size_t) {
    if (p >= _head)
      throw std::bad_alloc{};
    else if (p < _data)
      throw std::bad_alloc{};
    _head = (char *)p;
  }
  void reset() { _head = _data; }

  char *_data, *_head, *_tail;
  std::size_t _align;

private:
  mr_type *_mr;
};

template <typename value_t, typename memory_resource_t>
struct object_allocator {
  using value_type = value_t;
  using mr_type = memory_resource<memory_resource_t>;

  template <typename other_value_t>
  explicit object_allocator(
      const object_allocator<other_value_t, memory_resource_t> &o) noexcept {
    _mr = o.resource();
  }
  object_allocator(mr_type *mr) : _mr{mr} {};
  object_allocator() = delete;

  mr_type *resource() const noexcept { return _mr; }
  value_type *allocate(std::size_t n) {
    return (value_type *)(_mr->allocate(n * sizeof(value_type),
                                        alignof(value_type)));
  }
  void deallocate(value_type *p, std::size_t n) {
    _mr->deallocate((void *)p, n * sizeof(value_type), alignof(value_type));
  }

private:
  mr_type *_mr;
};

template <std::size_t chunk_size_v, typename memory_resource_t>
struct pool_allocator {
  static constexpr std::size_t chunk_size = chunk_size_v;
  using mr_type = memory_resource<memory_resource_t>;
};
template <typename memory_resource_t>
using page_allocator = pool_allocator<4096, memory_resource_t>;
template <typename structural_t, typename memory_resource_t>
struct structural_allocator
    : pool_allocator<structural_t::size, memory_resource_t> {};
/// 4K, 64K, 2M
template <typename chunk_sizes_t, typename memory_resource_t>
struct multipool_allocator;
template <std::size_t... chunk_sizes_v, typename memory_resource_t>
struct multipool_allocator<std::index_sequence<chunk_sizes_v...>,
                           memory_resource_t>
    : pool_allocator<chunk_sizes_v, memory_resource_t>... {};

} // namespace mn

#endif