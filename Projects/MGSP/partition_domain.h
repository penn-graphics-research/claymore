#ifndef PARTITION_DOMAIN_H
#define PARTITION_DOMAIN_H
#include <MnBase/Math/Vec.h>
#include <MnBase/Meta/Polymorphism.h>

namespace mn {

template<typename Derived, typename Tn, int Dim>
struct PartitionDomain {
	using index = std::array<Tn, Dim>;

	template<typename Index = index>
	constexpr bool inside(Index&& id) const noexcept {
		return self().inside(std::forward<Index>(id));
	}
	template<typename Offset, typename Index = index>
	constexpr bool within(Index&& id, Offset&& l, Offset&& u) const noexcept {
		return self().within(std::forward<Index>(id), std::forward<Index>(l), std::forward<Index>(u));
	}

   protected:
	auto& self() noexcept {
		return static_cast<Derived&>(*this);
	}
};

template<typename Tn, int Dim>
struct BoxDomain : PartitionDomain<BoxDomain<Tn, Dim>, Tn, Dim> {
	using base_t = PartitionDomain<BoxDomain<Tn, Dim>, Tn, Dim>;
	using index	 = typename base_t::index;

	index min;
	index max;

	constexpr BoxDomain() noexcept = default;

	constexpr BoxDomain(index lower, index upper) {
		for(int d = 0; d < Dim; ++d) {
			min[d] = lower[d];
			max[d] = upper[d];
		}
	}

	constexpr BoxDomain(vec<Tn, Dim> lower, vec<Tn, Dim> upper) {
		for(int d = 0; d < Dim; ++d) {
			min[d] = lower[d];
			max[d] = upper[d];
		}
	}

	template<typename Index = index>
	constexpr bool inside(Index&& id) const noexcept {
		for(int d = 0; d < Dim; ++d) {
			if(id[d] < min[d] || id[d] > max[d]) {
				return false;
			}
		}
		return true;
	}

	template<typename Offset, typename Index = index>
	constexpr bool within(Index&& id, Offset&& l, Offset&& u) const noexcept {
		for(int d = 0; d < Dim; ++d) {
			if(id[d] < min[d] + l[d] || id[d] > max[d] + u[d]) {
				return false;
			}
		}
		return true;
	}
};

}// namespace mn

#endif