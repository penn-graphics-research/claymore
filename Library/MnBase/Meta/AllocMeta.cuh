#ifndef ALLOC_META_H
#define ALLOC_META_H

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "MnSystem/Cuda/HostUtils.hpp"
#include "TypeMeta.h"

namespace mn {

/** HEAP **/

template<typename... Args>
auto make_vector(Args&&... args) {
	using Item = std::common_type_t<Args...>;
	std::vector<Item> result(sizeof...(Args));
	// works as a building block
	forArgs(
		[&result](auto&& x) {
			result.emplace_back(std::forward<decltype(x)>(x));
		},
		std::forward<Args>(args)...
	);
	return result;
}

/** CUDA **/

template<typename Type, typename Integer>
auto cuda_alloc(Integer size) -> Type* {
	Type* addr {nullptr};
	checkCudaErrors(cudaMalloc(static_cast<void**>(static_cast<void*>(&addr)), sizeof(Type) * size));
	return addr;
}

template<typename Type, typename Integer>
auto cuda_virtual_alloc(Integer size) -> Type* {
	Type* addr {nullptr};
	checkCudaErrors(cudaMallocManaged(static_cast<void**>(static_cast<void*>(&addr)), sizeof(Type) * size));
	return addr;
}

template<typename Type, typename Integer, typename AllocFunc>
auto cuda_alloc(Integer size, AllocFunc&& allocFunc) -> Type* {
	return static_cast<Type*>(static_cast<void*>((std::forward<AllocFunc>(allocFunc)(sizeof(Type) * size))));
}

#if 0
template <typename Tuple, typename Integer> auto cuda_allocs(Integer size) {
  std::array<void *, std::tuple_size<std::decay_t<Tuple>>::value> addrs;
  forIndexAlloc<Tuple>(
      [&addrs, size](std::size_t &&i, std::size_t &&typeSize) {
        checkCudaErrors(cudaMalloc(&addrs[i], typeSize * size));
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  return std::move(addrs);
}

template <typename Tuple, typename Integer>
auto cuda_virtual_allocs(Integer size) {
  std::array<void *, std::tuple_size<std::decay_t<Tuple>>::value> addrs;
  forIndexAlloc<Tuple>(
      [&addrs, size](std::size_t &&i, std::size_t &&typeSize) {
        checkCudaErrors(cudaMallocManaged(&addrs[i], typeSize * size));
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  return std::move(addrs);
}

template <typename Tuple, typename Integer>
auto cuda_allocs(const Integer *sizes) {
  std::array<void *, std::tuple_size<std::decay_t<Tuple>>::value> addrs;
  forIndexAlloc<Tuple>(
      [&addrs, &sizes](std::size_t &&i, std::size_t &&typeSize) {
        checkCudaErrors(cudaMalloc(&addrs[i], typeSize * sizes[i]));
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  return std::move(addrs);
}

template <typename Tuple, typename Integer>
auto cuda_virtual_allocs(const Integer *sizes) {
  std::array<void *, std::tuple_size<std::decay_t<Tuple>>::value> addrs;
  forIndexAlloc<Tuple>(
      [&addrs, &sizes](std::size_t &&i, std::size_t &&typeSize) {
        checkCudaErrors(cudaMallocManaged(&addrs[i], typeSize * sizes[i]));
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  return std::move(addrs);
}

template <typename Tuple, typename Integer, typename AllocFunc>
auto cuda_allocs(Integer size, AllocFunc &&allocFunc) {
  std::array<void *, std::tuple_size<std::decay_t<Tuple>>::value> addrs;
  forIndexAlloc<Tuple>(
      [&addrs, &allocFunc, size](std::size_t &&i, std::size_t &&typeSize) {
        addrs[i] = std::forward<AllocFunc>(allocFunc)(typeSize * size);
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  return std::move(addrs);
}

template <int num> void cuda_frees(std::array<void *, num> &_attribs) {
  for (auto &attrib : _attribs){
    checkCudaErrors(cudaFree(attrib));
  }
}

template <typename Type> void cuda_free(Type *addr) {
  checkCudaErrors(cudaFree(addr));
}

template <typename Tuple, typename Integer, typename Attribs>
void cuda_memcpys(const Integer size, Attribs &&from, Attribs &&to,
                  cudaStream_t stream = cudaStreamDefault) {
  forIndexAlloc<Tuple>(
      [&from, &to, &size, &stream](std::size_t &&i, std::size_t &&typeSize) {
        cudaMemcpyAsync(to[i], from[i], typeSize * size,
                        cudaMemcpyDeviceToDevice, stream);
      },
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}
#endif

}// namespace mn

#endif