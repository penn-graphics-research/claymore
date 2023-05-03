#ifndef DEVICE_UTILS_CUH
#define DEVICE_UTILS_CUH

#include <cooperative_groups.h>
#include <device_types.h>
#include <stdint.h>

namespace mn {

namespace cg = cooperative_groups;

__forceinline__ __device__ double unsigned_longlong_as_double(unsigned long long val) {
	const long long val_as_ll = *static_cast<long long*>(static_cast<void*>(&val));
	return __longlong_as_double(val_as_ll);
}

__forceinline__ __device__ unsigned long long double_as_unsigned_longlong(double val) {
	const long long val_as_ll = __double_as_longlong(val);
	return *static_cast<const unsigned long long*>(static_cast<const void*>(&val_as_ll));
}

__forceinline__ __device__ int clzull(unsigned long long val) {
	const long long val_as_ll = *static_cast<long long*>(static_cast<void*>(&val));
	return __clzll(val_as_ll);
}

template<typename T>
__forceinline__ __device__ bool atomic_min(T* address, T val);

template<typename T>
__forceinline__ __device__ bool atomic_max(T* address, T val);

template<typename T>
__forceinline__ __device__ T atomic_add_float(T* address, T val);

template<>
__forceinline__ __device__ float atomic_add_float<float>(float* address, float val) {
	int* address_as_i = static_cast<int*>(static_cast<void*>(address));
	int old			  = *address_as_i;
	int assumed;
	do {
		assumed = old;
		old		= atomicCAS(address_as_i, assumed, __float_as_int(val + __int_as_float(assumed)));
	} while(assumed != old);
	return __int_as_float(old);
}
template<>
__forceinline__ __device__ double atomic_add_float<double>(double* address, double val) {
	unsigned long long* address_as_ull = static_cast<unsigned long long*>(static_cast<void*>(address));
	unsigned long long old			   = *address_as_ull;
	unsigned long long assumed;
	do {
		assumed = old;
		old		= atomicCAS(address_as_ull, assumed, double_as_unsigned_longlong(val + unsigned_longlong_as_double(assumed)));
	} while(assumed != old);
	return unsigned_longlong_as_double(old);
}

template<>
__forceinline__ __device__ bool atomic_min<float>(float* address, float val) {
	int* address_as_i = static_cast<int*>(static_cast<void*>(address));
	int old			  = *address_as_i;
	int assumed;
	if(*address <= val) {
		return false;
	}
	do {
		assumed = old;
		old		= ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
	} while(assumed != old);
	return true;
}

template<>
__forceinline__ __device__ bool atomic_max<float>(float* address, float val) {
	int* address_as_i = static_cast<int*>(static_cast<void*>(address));
	int old			  = *address_as_i;
	int assumed;
	if(*address >= val) {
		return false;
	}
	do {
		assumed = old;
		old		= ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while(assumed != old);
	return true;
}

template<>
__forceinline__ __device__ bool atomic_min<double>(double* address, double val) {
#ifdef _WIN32
	uint64_t* address_as_ull = static_cast<uint64_t*>(static_cast<void*>(address));
	uint64_t old			 = *address_as_ull;
	uint64_t assumed;
#else
	unsigned long long* address_as_ull = static_cast<unsigned long long*>(static_cast<void*>(address));
	unsigned long long old			   = *address_as_ull;
	unsigned long long assumed;
#endif// _WIN32
	if(*address <= val) {
		return false;
	}
	do {
		assumed = old;
		old		= ::atomicCAS(address_as_ull, assumed, double_as_unsigned_longlong(::fmin(val, unsigned_longlong_as_double(assumed))));
	} while(assumed != old);
	return true;
}

template<>
__forceinline__ __device__ bool atomic_max<double>(double* address, double val) {
#ifdef _WIN32
	uint64_t* address_as_ull = static_cast<uint64_t*>(static_cast<void*>(address));
	uint64_t old			 = *address_as_ull;
	uint64_t assumed;
#else
	unsigned long long* address_as_ull = static_cast<unsigned long long*>(static_cast<void*>(address));
	unsigned long long old			   = *address_as_ull;
	unsigned long long assumed;
#endif// _WIN32
	if(*address >= val) {
		return false;
	}
	do {
		assumed = old;
		old		= ::atomicCAS(address_as_ull, assumed, double_as_unsigned_longlong(::fmax(val, unsigned_longlong_as_double(assumed))));
	} while(assumed != old);
	return true;
}

__device__ uint64_t packed_add(const uint64_t* masks, const uint64_t i, const uint64_t j);

template<int NumPageBits, int NumLevelBits>
__device__ uint64_t packed_add_father_neighbor(const uint64_t* masks, const uint64_t child_offset, const int father_level, const uint64_t father_neighbor_offset);

template<int NumPageBits>
__device__ int retrieve_block_local_offset(int level, uint64_t block_offset);

__forceinline__ __device__ uint64_t bit_spread_cuda(const uint64_t mask, int data) {
	uint64_t rmask	= __brevll(mask);
	uint64_t result = 0;
	unsigned char lz;
	unsigned char offset = clzull(rmask);
	while(rmask != 0) {
		lz	   = clzull(rmask) + 1;
		result = result << lz | (data & 1);
		data >>= 1, rmask <<= lz;
	}
	result = __brevll(result) >> clzull(mask);
	return result;
}

//NOLINTBEGIN(cppcoreguidelines-pro-type-union-access) Allowed in CUDA
__forceinline__ __device__ int bit_pack_cuda(const uint64_t mask, uint64_t data) {
	union {
		uint64_t slresult;
		uint64_t ulresult;
	} un		= {};//FIXME:Why this union? (Both members have the same type)
	int count	= 0;
	un.ulresult = 0;

	uint64_t rmask = __brevll(mask);
	unsigned char lz;

	while(rmask != 0) {
		lz = clzull(rmask);
		data >>= lz;
		un.ulresult <<= 1;
		count++;
		un.ulresult |= (data & 1);
		data >>= 1;
		rmask <<= lz + 1;
	}
	//NOLINTNEXTLINE(readability-magic-numbers) Define as constant?
	un.ulresult <<= 64 - count;// 64 means 64 bits ... maybe not use a constant 64 ...
	un.ulresult = __brevll(un.ulresult);
	return static_cast<int>(un.slresult);
}
//NOLINTEND(cppcoreguidelines-pro-type-union-access)

template<typename T>
__forceinline__ __device__ T atomic_agg_inc(T* p) {
	//Create group for all threads in this warp
	cg::coalesced_group g = cg::coalesced_threads();

	//First thread increases size
	T prev;
	if(g.thread_rank() == 0) {
		prev = atomicAdd(p, g.size());
	}

	//All threads fetch the value from the first thread and add their rank as offset
	prev = g.thread_rank() + g.shfl(prev, 0);
	return prev;
}

}// namespace mn

#endif
