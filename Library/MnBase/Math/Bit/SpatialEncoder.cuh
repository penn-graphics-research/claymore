#ifndef SPATIAL_ENCODER_CUH
#define SPATIAL_ENCODER_CUH
#include <driver_types.h>
#include <stdint.h>

#include "MnBase/AggregatedAttribs.cuh"
#include "MnBase/Math/Bit/Bits.h"

#define CHECK_ERRORS 0

namespace mn {

__forceinline__ __device__ int bit_pack_64(const uint64_t mask, const uint64_t data) {
	union {
		uint64_t slresult;
		uint64_t ulresult;
	};//FIXME:Why this union? (Both members have the same type)
	uint64_t uldata = data;
	int count		= 0;
	ulresult		= 0;

	uint64_t rmask = __brevll(mask);
	unsigned char lz;

	while(rmask) {
		lz = __clzll(rmask);
		uldata >>= lz;
		ulresult <<= 1;
		count++;
		ulresult |= (uldata & 1);
		uldata >>= 1;
		rmask <<= lz + 1;
	}
	//NOLINTNEXTLINE(readability-magic-numbers) Define as constant?
	ulresult <<= 64 - count;// 64 means 64 bits ... maybe not use a constant 64 ...
	ulresult = __brevll(ulresult);
	return (int) slresult;
}

__forceinline__ __device__ uint64_t bit_spread_64(const uint64_t mask, const int data) {
	uint64_t rmask	= __brevll(mask);
	int dat			= data;
	uint64_t result = 0;
	unsigned char lz, offset = __clzll(rmask);
	while(rmask) {
		lz	   = __clzll(rmask) + 1;
		result = result << lz | (dat & 1);
		dat >>= 1, rmask <<= lz;
	}
	result = __brevll(result) >> __clzll(mask);
	return result;
}

template<typename CoordType, typename IndexType, typename MaskType, int Dim = 3>
struct CoordToOffset {
	const CoordType dx_inv;
	std::array<MaskType, Dim> masks;

	CoordToOffset() = delete;

	__host__ CoordToOffset(CoordType dxinv, std::array<MaskType, Dim> masks)
		: dx_inv(std::move(dxinv)) {
		std::move(masks.begin(), masks.end(), masks);
	}

	template<typename In = CoordType>
	__forceinline__ __host__ __device__ auto operator()(int dim_no, In&& in) -> MaskType {
		static_assert(std::is_convertible<In, CoordType>::value, "CoordToOffset: coordinate type not convertible.");
#if CHECK_ERRORS
		uint64_t res		= bit_spread(masks[dim_no], static_cast<IndexType>(std::forward<In>(in) * static_cast<CoordType>(dx_inv) + static_cast<CoordType>(0.5)) - 1);
		unsigned pcoordtrue = static_cast<IndexType>(std::forward<In>(in) * static_cast<CoordType>(dx_inv) + static_cast<CoordType>(0.5)) - 1;
		unsigned pcoord		= bit_pack(masks[dim_no], res);
		if(pcoord != pcoordtrue) {
			printf("particle's %d-th comp coord(%d)-offset(%u) mapping error!\n", dim_no, pcoordtrue, pcoord);
		}
#endif
		return bit_spread_64(masks[dim_no], static_cast<IndexType>(std::forward<In>(in) * static_cast<CoordType>(dx_inv) + static_cast<CoordType>(0.5)) - 1);
	}
};

template<typename MaskType, typename IndexType, int Dim = 3>
struct OffsetToIndex {
	std::array<MaskType, Dim> masks;

	OffsetToIndex() = delete;

	__host__ OffsetToIndex(std::array<MaskType, Dim> masks) {
		std::move(masks.begin(), masks.end(), masks);
	}

	template<typename MT = MaskType>
	__forceinline__ __host__ __device__ auto operator()(int dim_no, MT&& offset) -> IndexType {
		return bit_pack_64(masks[dim_no], std::forward<MT>(offset));
	}
};

template<typename SPGTraits>
struct ComputeNeighborKey {
	static constexpr int DIM = SPGTraits::dimension;

	template<int neighborid, typename SPGTs = SPGTraits>
	__forceinline__ __host__ __device__ std::enable_if_t<SPGTs::dimension == 2, uint64_t> operator()(uint64_t key) {
		constexpr uint64_t NEIGHBOR_OFFSET = SPGTraits::neighbor_block_offset((neighborid & 2) != 0 ? 1 : 0, neighborid & 1);

		//printf("%d-th neighbor %llx\n", neighborid, NEIGHBOR_OFFSET);

		uint64_t x_result = ((key | SPGTraits::MXADD_Xmask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Xmask)) & ~SPGTraits::MXADD_Xmask;
		uint64_t y_result = ((key | SPGTraits::MXADD_Ymask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Ymask)) & ~SPGTraits::MXADD_Ymask;
		uint64_t w_result = ((key | SPGTraits::MXADD_Wmask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Wmask)) & ~SPGTraits::MXADD_Wmask;
		uint64_t result	  = x_result | y_result | w_result;

		return result >> SPGTraits::page_bits;
	}

	template<int neighborid, typename SPGTs = SPGTraits>
	__forceinline__ __host__ __device__ std::enable_if_t<SPGTs::dimension == 3, uint64_t> operator()(uint64_t key) {
		constexpr uint64_t NEIGHBOR_OFFSET = SPGTraits::neighbor_block_offset((neighborid & 4) != 0 ? 1 : 0, neighborid & 2 ? 1 : 0, neighborid & 1);

		//printf("%d-th neighbor %llx\n", neighborid, NEIGHBOR_OFFSET);

		uint64_t x_result = ((key | SPGTraits::MXADD_Xmask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Xmask)) & ~SPGTraits::MXADD_Xmask;
		uint64_t y_result = ((key | SPGTraits::MXADD_Ymask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Ymask)) & ~SPGTraits::MXADD_Ymask;
		uint64_t z_result = ((key | SPGTraits::MXADD_Zmask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Zmask)) & ~SPGTraits::MXADD_Zmask;
		uint64_t w_result = ((key | SPGTraits::MXADD_Wmask) + (NEIGHBOR_OFFSET & ~SPGTraits::MXADD_Wmask)) & ~SPGTraits::MXADD_Wmask;
		uint64_t result	  = x_result | y_result | z_result | w_result;

		return result >> SPGTraits::page_bits;
	}
};

}// namespace mn

#endif