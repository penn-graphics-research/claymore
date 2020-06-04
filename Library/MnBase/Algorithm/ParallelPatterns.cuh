#ifndef __PARALLEL_PATTERNS_CUH_
#define __PARALLEL_PATTERNS_CUH_
#include <MnSystem/Cuda/HostUtils.hpp>
#include <MnSystem/Cuda/MemoryUtils.cuh>
//#include <cub/device/device_radix_sort.cuh>
//#include <cub/device/device_scan.cuh>
//#include <cub/device/device_select.cuh>
#include <thrust/fill.h>
#include <thrust/scan.h>

//#include <helper_cuda.h>

namespace mn {

/// sequence
template <typename T>
void sequence(std::size_t count, T *_array, T value = 0,
              cudaStream_t stream = cudaStreamDefault) {
  checkThrustErrors(thrust::sequence(thrust::cuda::par.on(stream),
                                     getDevicePtr(_array),
                                     getDevicePtr(_array) + count, value));
}

/// reduece
template <typename T>
T reduce(std::size_t count, T *_array, T value = 0,
         cudaStream_t stream = cudaStreamDefault) {
  T result = (thrust::reduce(thrust::cuda::par.on(stream), getDevicePtr(_array),
                             getDevicePtr(_array) + count, value));
  return result;
}

/// fill
template <typename T>
void fill_with(std::size_t count, T *_array, T value,
               cudaStream_t stream = cudaStreamDefault) {
  checkThrustErrors(thrust::fill(thrust::cuda::par.on(stream),
                                 getDevicePtr(_array),
                                 getDevicePtr(_array) + count, value));
}
template <typename T, unsigned Size>
void fill_with(std::size_t count, AttribPort<T, Size> _attribs, T value,
               cudaStream_t stream = cudaStreamDefault) {
  for (int i = 0; i < (int)Size; ++i)
    fill_with<T>(count, _attribs[i], value, stream);
}
/// set
template <typename T, typename ValueOp>
__global__ void set_values(std::size_t num, T *_array, ValueOp op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num)
    return;
  _array[idx] = op(idx);
}
template <typename T, typename ValueOp>
void set_with(std::size_t count, T *_array, ValueOp value_op) {
  cuda_launch({(count + 255) / 256, 256}, set_values<T, ValueOp>, count, _array,
              value_op);
}
template <typename T, unsigned Size, typename ValueOp>
void set_with(std::size_t count, AttribPort<T, Size> _attribs,
              ValueOp value_op) {
  for (int i = 0; i < Size; ++i)
    set_with<T, ValueOp>(count, _attribs[i], value_op);
}

#if 0
	template<typename Key, typename Value>
	void sort_by_key(std::size_t count, const Key* _keysIn, const Value* _valuesIn, Key* _keysOut, Value* _valuesOut, cudaStream_t stream=cudaStreamDefault) {
		void     *d_temp_storage = nullptr;
		std::size_t   temp_storage_bytes = 0;
		checkCudaErrors(cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes, _keysIn, _keysOut, _valuesIn, _valuesOut, count, 0, sizeof(Key) * 8, stream));
		d_temp_storage = cudaMemBorrow(temp_storage_bytes);	///< more managable memory consumption
		//printf("requires %lu bytes for sort_by_key\n", temp_storage_bytes);
		checkCudaErrors(cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes, _keysIn, _keysOut, _valuesIn, _valuesOut, count, 0, sizeof(Key) * 8, stream));
	}

	template<typename Key, typename Value>
	void sort_by_key_bits(std::size_t count, const Key* _keysIn, const Value* _valuesIn, Key* _keysOut, Value* _valuesOut, int least_sig_bits=12, int most_sig_bits=sizeof(Key)*8, cudaStream_t stream=cudaStreamDefault) {
		void     *d_temp_storage = nullptr;
		std::size_t   temp_storage_bytes = 0;
		checkCudaErrors(cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes, _keysIn, _keysOut, _valuesIn, _valuesOut, count, least_sig_bits, most_sig_bits, stream));
		d_temp_storage = cudaMemBorrow(temp_storage_bytes);	///< more managable memory consumption
		//printf("requires %lu bytes for sort_by_key\n", temp_storage_bytes);
		checkCudaErrors(cub::DeviceRadixSort::SortPairs(
			d_temp_storage, temp_storage_bytes, _keysIn, _keysOut, _valuesIn, _valuesOut, count, least_sig_bits, most_sig_bits, stream));
	}

	/// exclusive scan
	template<typename TIn, typename TOut, typename BinaryOp>
	void exclusive_scan(std::size_t count, const TIn* _input, TOut* _output, BinaryOp&& op, cudaStream_t stream=cudaStreamDefault) {
		void     *d_temp_storage = nullptr;
		std::size_t   temp_storage_bytes = 0;
		checkCudaErrors(cub::DeviceScan::ExclusiveScan(
			d_temp_storage, temp_storage_bytes, _input, _output, std::forward<BinaryOp>(op), 0, count, stream));
		d_temp_storage = cudaMemBorrow(temp_storage_bytes);	///< more managable memory consumption
		//printf("requires %llu bytes for exclusive scan\n", (long long unsigned int)temp_storage_bytes);
		checkCudaErrors(cub::DeviceScan::ExclusiveScan(
			d_temp_storage, temp_storage_bytes, _input, _output, std::forward<BinaryOp>(op), 0, count, stream));
	}

	/// count if
	template<typename T, typename CounterType, typename UnaryOp>
	void select_if(std::size_t inputcount, const T* _input, CounterType* _outputcount, T* _output, UnaryOp&& op, cudaStream_t stream=cudaStreamDefault) {
		void     *d_temp_storage = nullptr;
		std::size_t   temp_storage_bytes = 0;
		cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, _input, _output, _outputcount, inputcount, std::forward<UnaryOp>(op), stream);
		d_temp_storage = cudaMemBorrow(temp_storage_bytes);	///< more managable memory consumption
		//printf("requires %llu bytes for select if\n", (long long unsigned int)temp_storage_bytes);
		cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, _input, _output, _outputcount, inputcount, std::forward<UnaryOp>(op), stream);
	}
#endif

} // namespace mn

#endif
