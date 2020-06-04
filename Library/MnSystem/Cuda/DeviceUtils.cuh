#ifndef __DEVICE_UTILS_CUH_
#define __DEVICE_UTILS_CUH_

#include <stdint.h>
#include <device_types.h>
#include <cooperative_groups.h>

namespace mn {

	namespace cg = cooperative_groups;

	template<typename T>
	__forceinline__ __device__ bool atomicMin(T* address, T val);
	template<typename T>
	__forceinline__ __device__ bool atomicMax(T* address, T val);
	template<typename T>
	__forceinline__ __device__ T atomicAddFloat(T* address, T val);

	template<>
	__forceinline__ __device__ float atomicAddFloat<float>(float* address, float val)
	{
    	int* address_as_ull = (int*)address;
    	int old = *address_as_ull, assumed;
    	do {
        	assumed = old;
        	old = atomicCAS(address_as_ull, assumed, 
                __float_as_int(val + __int_as_float(assumed)));
    	} while (assumed != old);
    	return __int_as_float(old);
	}
	template<>
	__forceinline__ __device__ double atomicAddFloat<double>(double* address, double val)
	{
    	unsigned long long * address_as_ull =
                                          (unsigned long long*)address;
    	unsigned long long old = *address_as_ull, assumed;
    	do {
        	assumed = old;
        	old = atomicCAS(address_as_ull, assumed, 
                __double_as_longlong(val + __longlong_as_double(assumed)));
    	} while (assumed != old);
    	return __longlong_as_double(old);
	}

	template<>
	__forceinline__ __device__ bool atomicMin<float>(float* address, float val) {
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		if (*address <= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fminf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return true;
	}

	template<>
	__forceinline__	__device__ bool atomicMax<float>(float* address, float val) {
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		if (*address >= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return true;
	}
	
	template<>
	__forceinline__	__device__ bool atomicMin<double>(double* address, double val) {
#ifdef _WIN32
		uint64_t* address_as_ull = (uint64_t*)address;
		uint64_t old = *address_as_ull, assumed;
#else
		unsigned long long * address_as_ull = (unsigned long long*)address;
		unsigned long long old = *address_as_ull, assumed;
#endif // _WIN32
		if (*address <= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_ull, assumed,
				__double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return true;
	}

	template<>
	__forceinline__	__device__ bool atomicMax<double>(double* address, double val) {
#ifdef _WIN32
		uint64_t* address_as_ull = (uint64_t*)address;
		uint64_t old = *address_as_ull, assumed;
#else
		unsigned long long * address_as_ull = (unsigned long long*)address;
		unsigned long long old = *address_as_ull, assumed;
#endif // _WIN32
		if (*address >= val) return false;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_ull, assumed,
				__double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
		} while (assumed != old);
		return true;
	}

    __device__ uint64_t Packed_Add(const uint64_t* masks,const uint64_t i,const uint64_t j);
	template<int NumPageBits, int NumLevelBits>
    __device__ uint64_t Packed_Add_Father_Neighbor(const uint64_t* masks, const uint64_t childOffset, const int fatherLevel, const uint64_t fatherNeighborOffset);
	template<int NumPageBits>
    __device__ int Retrieve_Block_Local_Offset(int level, uint64_t blockOffset);
    __forceinline__ __device__ uint64_t bit_spread_cuda(const uint64_t mask, int data) {
		uint64_t rmask = __brevll(mask);
        uint64_t result = 0;
        unsigned char lz, offset = __clzll(rmask);
        while (rmask) {
            lz = __clzll(rmask) + 1;
            result = result << lz | (data & 1);
            data >>= 1, rmask <<= lz;
        }
        result = __brevll(result) >> __clzll(mask);
        return result;
	}
	__forceinline__ __device__ int bit_pack_cuda(const uint64_t mask, uint64_t data) {
		union{ uint64_t slresult; uint64_t ulresult; };
		int count=0; ulresult=0;

		uint64_t rmask = __brevll(mask);
        unsigned char lz;

        while(rmask) {
            lz = __clzll(rmask) ;
            data>>=lz;
            ulresult<<=1;
            count++;
            ulresult |= (data & 1);
            data>>=1;
            rmask <<= lz+1;
        }
        ulresult <<= 64-count; // 64 means 64 bits ... maybe not use a constant 64 ...
        ulresult = __brevll(ulresult);
        return (int)slresult;
	}

template<typename T>
__forceinline__ __device__ T atomicAggInc(T *p) {
    cg::coalesced_group g = cg::coalesced_threads();
    T prev;
    if (g.thread_rank() == 0) {
        prev = atomicAdd(p, g.size());
    }
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

}

#endif
