#ifndef __STRUCTURAL_AUXILIARY_H_
#define __STRUCTURAL_AUXILIARY_H_
#include <MnBase/Math/Vec.h>
#include <MnBase/Meta/MathMeta.h>

namespace mn {

struct memory_layout {
  enum class element : unsigned char { aos = 0, soa = 1 };
};
using attrib_layout = memory_layout::element;

struct identity_decorator {};

enum class structural_allocation_policy : std::size_t {
  full_allocation = 0,
  on_demand = 1,
  total
};
enum class structural_padding_policy : std::size_t {
  compact = 0,
  sum_pow2_align = 1,
  max_pow2_align = 2,
  total
};
template <structural_allocation_policy alloc_policy_ =
              structural_allocation_policy::full_allocation,
          structural_padding_policy padding_policy_ =
              structural_padding_policy::compact>
struct decorator : identity_decorator {
  static constexpr auto alloc_policy = alloc_policy_;
  static constexpr auto padding_policy = padding_policy_;
};

struct identity_structural_index {};
template <typename Tn, Tn... Ns>
struct compact_domain : identity_structural_index, indexer<Tn, Ns...> {
  using base_t = indexer<Tn, Ns...>;
  using index = vec<typename base_t::index_type, base_t::dim>;
};
template <typename Tn, Tn... Ns>
using aligned_domain = compact_domain<Tn, next_2pow<Tn, Ns>::value...>;

} // namespace mn

#endif