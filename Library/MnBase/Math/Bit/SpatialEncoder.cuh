#ifndef __SPATIAL_ENCODER_CUH_
#define __SPATIAL_ENCODER_CUH_
#include <MnBase/Math/Bit/Bits.h>
#include <MnBase/AggregatedAttribs.cuh>
#include <driver_types.h>
#include <stdint.h>

namespace mn {

	__forceinline__ __device__ int bit_pack_64(const uint64_t mask, const uint64_t data) {
		union { uint64_t slresult; uint64_t ulresult; };
		uint64_t uldata = data; int count = 0; ulresult = 0;

		uint64_t rmask = __brevll(mask);
		unsigned char lz;

		while (rmask) {
			lz = __clzll(rmask);
			uldata >>= lz;
			ulresult <<= 1;
			count++;
			ulresult |= (uldata & 1);
			uldata >>= 1;
			rmask <<= lz + 1;
		}
		ulresult <<= 64 - count; // 64 means 64 bits ... maybe not use a constant 64 ...
		ulresult = __brevll(ulresult);
		return (int)slresult;
	}

	__forceinline__ __device__ uint64_t bit_spread_64(const uint64_t mask, const int data) {
		uint64_t rmask = __brevll(mask);
		int dat = data;
		uint64_t result = 0;
		unsigned char lz, offset = __clzll(rmask);
		while (rmask) {
			lz = __clzll(rmask) + 1;
			result = result << lz | (dat & 1);
			dat >>= 1, rmask <<= lz;
		}
		result = __brevll(result) >> __clzll(mask);
		return result;
	}

	template<typename CoordType, typename IndexType, typename MaskType, int Dim = 3>
	struct coord_to_offset {
		coord_to_offset() = delete;
		__host__ coord_to_offset(CoordType dxinv, std::array<MaskType, Dim> masks) : _dxInv(std::move(dxinv)) {
			std::move(masks.begin(), masks.end(), _masks);
		}
		template<typename In = CoordType>
		__forceinline__ __host__ __device__ auto operator()(int dimNo, In&& in) -> MaskType {
			static_assert(std::is_convertible<In, CoordType>::value, "coord_to_offset: coordinate type not convertible.");
			#if 0
			uint64_t res = bit_spread(_masks[dimNo], static_cast<IndexType>(
				std::forward<In>(in) * static_cast<CoordType>(_dxInv) + static_cast<CoordType>(0.5)) - 1);
			unsigned pcoordtrue = static_cast<IndexType>(std::forward<In>(in) * static_cast<CoordType>(_dxInv) + static_cast<CoordType>(0.5)) - 1;
			unsigned pcoord = bit_pack(_masks[dimNo], res);
			if (pcoord != pcoordtrue)
				printf("particle's %d-th comp coord(%d)-offset(%u) mapping error!\n", dimNo, pcoordtrue, pcoord);
			#endif
			return bit_spread_64(_masks[dimNo], static_cast<IndexType>(
				std::forward<In>(in) * static_cast<CoordType>(_dxInv) + static_cast<CoordType>(0.5)) - 1);
		}
		const CoordType _dxInv;
		MaskType		_masks[Dim];
	};

	template<typename MaskType, typename IndexType, int Dim = 3>
	struct offset_to_index {
		offset_to_index() = delete;
		__host__ offset_to_index(std::array<MaskType, Dim> masks) {
			std::move(masks.begin(), masks.end(), _masks);
		}
		template<typename MT = MaskType>
		__forceinline__ __host__ __device__ auto operator()(int dimNo, MT&& offset) -> IndexType {
			return bit_pack_64(_masks[dimNo], std::forward<MT>(offset));
		}
		MaskType		_masks[Dim];
	};

	template<typename SPGTraits>
	struct compute_neighbor_key {
		static constexpr int dim = SPGTraits::dimension;
		template<int neighborid, typename SPGTs = SPGTraits>
		__forceinline__ __host__ __device__ std::enable_if_t<SPGTs::dimension == 2, uint64_t> operator()(uint64_t key) {
			constexpr uint64_t neighborOffset = SPGTraits::neighbor_block_offset(neighborid & 2 ? 1 : 0, neighborid & 1);
			//printf("%d-th neighbor %llx\n", neighborid, neighborOffset);
			uint64_t x_result = ((key | SPGTraits::MXADD_Xmask) + (neighborOffset & ~SPGTraits::MXADD_Xmask)) & ~SPGTraits::MXADD_Xmask;
			uint64_t y_result = ((key | SPGTraits::MXADD_Ymask) + (neighborOffset & ~SPGTraits::MXADD_Ymask)) & ~SPGTraits::MXADD_Ymask;
			uint64_t w_result = ((key | SPGTraits::MXADD_Wmask) + (neighborOffset & ~SPGTraits::MXADD_Wmask)) & ~SPGTraits::MXADD_Wmask;
			uint64_t result = x_result | y_result | w_result;
			return result >> SPGTraits::page_bits;
		}
		template<int neighborid, typename SPGTs = SPGTraits>
		__forceinline__ __host__ __device__ std::enable_if_t<SPGTs::dimension == 3, uint64_t> operator()(uint64_t key) {
			constexpr uint64_t neighborOffset = SPGTraits::neighbor_block_offset(
				neighborid & 4 ? 1 : 0, neighborid & 2 ? 1 : 0, neighborid & 1);
			//printf("%d-th neighbor %llx\n", neighborid, neighborOffset);
			uint64_t x_result = ((key | SPGTraits::MXADD_Xmask) + (neighborOffset & ~SPGTraits::MXADD_Xmask)) & ~SPGTraits::MXADD_Xmask;
			uint64_t y_result = ((key | SPGTraits::MXADD_Ymask) + (neighborOffset & ~SPGTraits::MXADD_Ymask)) & ~SPGTraits::MXADD_Ymask;
			uint64_t z_result = ((key | SPGTraits::MXADD_Zmask) + (neighborOffset & ~SPGTraits::MXADD_Zmask)) & ~SPGTraits::MXADD_Zmask;
			uint64_t w_result = ((key | SPGTraits::MXADD_Wmask) + (neighborOffset & ~SPGTraits::MXADD_Wmask)) & ~SPGTraits::MXADD_Wmask;
			uint64_t result = x_result | y_result | z_result | w_result;
			return result >> SPGTraits::page_bits;
		}
	};

}

#endif