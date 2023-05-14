#ifndef STRUCTURAL_AUXILIARY_H
#define STRUCTURAL_AUXILIARY_H
#include "MnBase/Math/Vec.h"
#include "MnBase/Meta/MathMeta.h"

namespace mn {

struct MemoryLayout {
	enum class element : unsigned char {
		AOS = 0,
		SOA = 1
	};
};
enum class StructuralAllocationPolicy : std::size_t {
	FULL_ALLOCATION = 0,
	ON_DEMAND		= 1,
	TOTAL
};
enum class StructuralPaddingPolicy : std::size_t {
	COMPACT		   = 0,
	SUM_POW2_ALIGN = 1,
	MAX_POW2_ALIGN = 2,
	TOTAL
};

using attrib_layout = MemoryLayout::element;

struct IdentityDecorator {};

template<StructuralAllocationPolicy AllocPolicy = StructuralAllocationPolicy::FULL_ALLOCATION, StructuralPaddingPolicy PaddingPolicy = StructuralPaddingPolicy::COMPACT>
struct Decorator : IdentityDecorator {
	static constexpr auto alloc_policy	 = AllocPolicy;
	static constexpr auto padding_policy = PaddingPolicy;
};

struct IdentityStructuralIndex {};

template<typename Tn, Tn... Ns>
struct CompactDomain
	: IdentityStructuralIndex
	, indexer<Tn, Ns...> {
	using base_t = indexer<Tn, Ns...>;
	using index	 = vec<typename base_t::index_type, base_t::dim>;
};

template<typename Tn, Tn... Ns>
using AlignedDomain = CompactDomain<Tn, next_2pow<Tn, Ns>::value...>;

}// namespace mn

#endif