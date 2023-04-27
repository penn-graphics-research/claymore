#ifndef MAPPING_KERNELS_CUH
#define MAPPING_KERNELS_CUH
#include <driver_types.h>

#define PRINT_FIRST_INDICES 0

namespace mn {

template<typename EntryType, typename MarkerType, typename BinaryOp>
__global__ void mark_boundary(int num, const EntryType* entries, MarkerType* marks, BinaryOp op) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	/// mark the tails of all segments
	marks[idx] = (idx != num - 1) ? op(entries[idx], entries[idx + 1]) : 1;
}

template<typename IndexType>
__global__ void set_inverse(int num, const IndexType* map, IndexType* map_inv) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	auto map_idx = map[idx];
	if(idx == 0 || map_idx != map[idx - 1]) {
		map_inv[map_idx] = idx;
#if PRINT_FIRST_INDICES
		if(map_idx < 5) {
			printf("%d-th block starts at %d\n", map_idx, map_inv[map_idx]);
		}
#endif
	}
	if(idx == num - 1) {
		map_inv[map_idx + 1] = num;
	}
#if PRINT_FIRST_INDICES
	if(idx < 5) {
		printf("%d-th particle belongs to block %d\n", idx, map_idx);
	}
#endif
}

template<typename CounterType, typename IndexType>
__global__ void exclusive_scan_inverse(CounterType num, const IndexType* map, IndexType* map_inv) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	
	const auto map_idx = map[idx];
	if(map_idx != map[idx + 1]) {
		map_inv[map_idx] = idx;
	}
}

template<typename IndexType>
__global__ void map_inverse(int num, const IndexType* map, IndexType* map_inv) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num) {
		return;
	}
	map_inv[map[idx]] = idx;
}

template<typename MappingType, typename CounterType>
__global__ void set_range_inverse(int count, const MappingType* to_packed_range_map, const MappingType* to_range_map, CounterType* num_packed_range, MappingType* range_ids, MappingType* range_left_bound, MappingType* range_right_bound) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count) {
		return;
	}
	auto packed_range_idx = to_packed_range_map[idx];
	auto range_idx		  = to_range_map[idx];
	bool l_tag			  = idx == 0 || range_idx != to_range_map[idx - 1];
	bool r_tag			  = idx == count - 1 || range_idx != to_range_map[idx + 1];
	/// left bound
	if(l_tag) {
		range_left_bound[packed_range_idx] = idx;
		range_ids[packed_range_idx]		   = range_idx;
#if PRINT_FIRST_INDICES
		if(packed_range_idx == 0) {
			printf("%d-th block st (%d): %d\n", packed_range_idx, range_idx, range_left_bound[packed_range_idx]);
		}
#endif
	}
	/// right bound
	if(r_tag) {
		range_right_bound[packed_range_idx] = idx + 1;
#if PRINT_FIRST_INDICES
		if(packed_range_idx == 0) {
			printf("%d-th block ed: %d\n", packed_range_idx, range_right_bound[packed_range_idx]);
		}
#endif
	}
	if(idx == count - 1) {
		*num_packed_range = packed_range_idx + 1;
	}
}

}// namespace mn

#endif