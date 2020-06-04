#ifndef __PATTERN_META_H_
#define __PATTERN_META_H_

#include "TypeMeta.h"
#include <initializer_list>
#include <type_traits>

namespace mn {

template <typename, typename> struct gather;

/// Seq must be integer_sequence
template <std::size_t... Is, typename ValueSeq>
struct gather<std::index_sequence<Is...>, ValueSeq> {
  using type = std::integer_sequence<typename ValueSeq::value_type,
                                select_value<Is, ValueSeq>::value...>;
};
template <typename Indices, typename ValueSeq>
using gather_t = typename gather<Indices, ValueSeq>::type;

} // namespace mn

#endif