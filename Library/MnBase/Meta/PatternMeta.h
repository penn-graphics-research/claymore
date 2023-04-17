#ifndef PATTERN_META_H
#define PATTERN_META_H

#include <initializer_list>
#include <type_traits>

#include "TypeMeta.h"

namespace mn {

template<typename, typename>
struct gather;

/// Seq must be integer_sequence
template<std::size_t... Is, typename ValueSeq>
struct gather<std::index_sequence<Is...>, ValueSeq> {
	using type = std::integer_sequence<typename ValueSeq::value_type, select_value<Is, ValueSeq>::value...>;
};
template<typename Indices, typename ValueSeq>
using gather_t = typename gather<Indices, ValueSeq>::type;

}// namespace mn

#endif