#ifndef __PARTITION_DOMAIN_H_
#define __PARTITION_DOMAIN_H_
#include <MnBase/Math/Vec.h>
#include <MnBase/Meta/Polymorphism.h>

namespace mn {

template <typename Derived, typename Tn, int dim> struct partition_domain {
  using index = Tn[dim];

  template <typename Index = index>
  constexpr bool inside(Index &&id) const noexcept {
    return self().inside(std::forward<Index>(id));
  }
  template <typename Offset, typename Index = index>
  constexpr bool within(Index &&id, Offset &&l, Offset &&u) const noexcept {
    return self().within(std::forward<Index>(id), std::forward<Index>(l),
                         std::forward<Index>(u));
  }

protected:
  auto &self() noexcept { return static_cast<Derived &>(*this); }
};

template <typename Tn, int dim>
struct box_domain : partition_domain<box_domain<Tn, dim>, Tn, dim> {
  using base_t = partition_domain<box_domain<Tn, dim>, Tn, dim>;
  using index = typename base_t::index;
  constexpr box_domain() noexcept {}
  constexpr box_domain(index lower, index upper) {
    for (int d = 0; d < dim; ++d) {
      _min[d] = lower[d];
      _max[d] = upper[d];
    }
  }
  constexpr box_domain(vec<Tn, dim> lower, vec<Tn, dim> upper) {
    for (int d = 0; d < dim; ++d) {
      _min[d] = lower[d];
      _max[d] = upper[d];
    }
  }
  template <typename Index = index>
  constexpr bool inside(Index &&id) const noexcept {
    for (int d = 0; d < dim; ++d)
      if (id[d] < _min[d] || id[d] > _max[d])
        return false;
    return true;
  }
  template <typename Offset, typename Index = index>
  constexpr bool within(Index &&id, Offset &&l, Offset &&u) const noexcept {
    for (int d = 0; d < dim; ++d)
      if (id[d] < _min[d] + l[d] || id[d] > _max[d] + u[d])
        return false;
    return true;
  }
  index _min, _max;
};

} // namespace mn

#endif