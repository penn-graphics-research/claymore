#ifndef __TYPE_META_H_
#define __TYPE_META_H_

#include <MnBase/Reflection.h>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mn {

/// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
template <class T> struct unwrap_refwrapper { using type = T; };
template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> {
  using type = T &;
};
template <class T>
using special_decay_t =
    typename unwrap_refwrapper<typename std::decay_t<T>>::type;

/// indexable type list to avoid recursion
namespace meta_impl {
template <typename T> struct type_wrapper { using type = T; };
template <std::size_t, typename T> struct indexed_type { using type = T; };

template <typename, typename... Ts> struct indexed_types;

template <std::size_t... Is, typename... Ts>
struct indexed_types<std::index_sequence<Is...>, Ts...>
    : indexed_type<Is, Ts>... {};

/// use pointer rather than reference as in taocpp! [incomplete type error]
template <std::size_t I, typename T>
indexed_type<I, T> extract_type(indexed_type<I, T> *);
} // namespace meta_impl

template <typename... Ts> struct type_seq {
  using indices = std::index_sequence_for<Ts...>;
  template <std::size_t I>
  using type = typename decltype(meta_impl::extract_type<I>(
      std::add_pointer_t<meta_impl::indexed_types<indices, Ts...>>{}))::type;
};
template <typename T, T... Ns>
struct type_seq<std::integer_sequence<T, Ns...>> {
  using indices = std::make_index_sequence<sizeof...(Ns)>;
  template <std::size_t I>
  using type = typename decltype(meta_impl::extract_type<I>(
      std::add_pointer_t<meta_impl::indexed_types<
          indices, std::integral_constant<T, Ns>...>>{}))::type;
};

/// select type by index
template <std::size_t I, typename... Ts>
using select_indexed_type = typename type_seq<Ts...>::template type<I>;
template <std::size_t I, typename TypeSeq>
using select_type = typename TypeSeq::template type<I>;

/// select (constant integral) value (integral_constant<T, N>) by index
template <std::size_t I, typename T, T... Ns>
using select_indexed_value =
    select_indexed_type<I, std::integral_constant<T, Ns>...>;
template <std::size_t I, typename ValueSeq>
using select_value = typename type_seq<ValueSeq>::template type<I>;

/// reverse seq
namespace meta_impl {
template <typename, typename> struct reverse_type_seq_impl;
template <std::size_t... Is, typename... Ts>
struct reverse_type_seq_impl<std::index_sequence<Is...>, type_seq<Ts...>> {
  static_assert(sizeof...(Is) == sizeof...(Ts), "reverse_type_seq_impl error");
  using type = type_seq<
      typename type_seq<Ts...>::template type<sizeof...(Is) - 1 - Is>...>;
};
template <typename, typename> struct reverse_value_seq_impl;
template <std::size_t... Is, typename T, T... Ns>
struct reverse_value_seq_impl<std::index_sequence<Is...>,
                              std::integer_sequence<T, Ns...>> {
  static_assert(sizeof...(Is) == sizeof...(Ns), "reverse_value_seq_impl error");
  using type = std::integer_sequence<
      T, (typename type_seq<typename std::integral_constant<T, Ns>...>::
              template type<sizeof...(Is) - 1 - Is>{})...>;
};
} // namespace meta_impl
template <typename... Ts>
using type_seq_rev =
    typename meta_impl::reverse_type_seq_impl<std::index_sequence_for<Ts...>,
                                              type_seq<Ts...>>::type;
template <typename TypeSeq> struct rev_type_seq_impl;
template <typename... Ts> struct rev_type_seq_impl<type_seq<Ts...>> {
  using type =
      typename meta_impl::reverse_type_seq_impl<std::index_sequence_for<Ts...>,
                                                type_seq<Ts...>>::type;
};
template <typename TypeSeq>
using rev_type_seq = typename rev_type_seq_impl<TypeSeq>::type;

template <typename T, T... Ns>
using value_seq_rev = typename meta_impl::reverse_value_seq_impl<
    std::make_index_sequence<sizeof...(Ns)>,
    std::integer_sequence<T, Ns...>>::type;

template <typename ValueSeq> struct rev_value_seq_impl;
template <typename T, T... Ns>
struct rev_value_seq_impl<std::integer_sequence<T, Ns...>> {
  using type = typename meta_impl::reverse_value_seq_impl<
      std::make_index_sequence<sizeof...(Ns)>,
      std::integer_sequence<T, Ns...>>::type;
};
template <typename ValueSeq>
using rev_value_seq = typename rev_value_seq_impl<ValueSeq>::type;

/// legacy
template <typename... Ts> struct type_list {
  using Indices = std::make_integer_sequence<int, sizeof...(Ts)>;
  using types = std::tuple<std::decay_t<Ts>...>;
  template <std::size_t I>
  using type = typename std::tuple_element<I, types>::type;
  static constexpr auto count = sizeof...(Ts);
  static constexpr auto size =
      sizeof(types); // wrong: std::tuple_size<types>::value;
  // template <std::size_t I>
  // static constexpr auto offset = tupleOffset<I, types>::value;
  // using Indices = ;
};

template <std::size_t I, typename... ValueTypes>
using AttribType = typename type_seq<ValueTypes...>::template type<I>;

namespace placeholder {
using placeholder_type = unsigned;
constexpr auto _0 = std::integral_constant<placeholder_type, 0>{};
constexpr auto _1 = std::integral_constant<placeholder_type, 1>{};
constexpr auto _2 = std::integral_constant<placeholder_type, 2>{};
constexpr auto _3 = std::integral_constant<placeholder_type, 3>{};
constexpr auto _4 = std::integral_constant<placeholder_type, 4>{};
constexpr auto _5 = std::integral_constant<placeholder_type, 5>{};
constexpr auto _6 = std::integral_constant<placeholder_type, 6>{};
constexpr auto _7 = std::integral_constant<placeholder_type, 7>{};
constexpr auto _8 = std::integral_constant<placeholder_type, 8>{};
constexpr auto _9 = std::integral_constant<placeholder_type, 9>{};
constexpr auto _10 = std::integral_constant<placeholder_type, 10>{};
constexpr auto _11 = std::integral_constant<placeholder_type, 11>{};
constexpr auto _12 = std::integral_constant<placeholder_type, 12>{};
constexpr auto _13 = std::integral_constant<placeholder_type, 13>{};
constexpr auto _14 = std::integral_constant<placeholder_type, 14>{};
constexpr auto _15 = std::integral_constant<placeholder_type, 15>{};
constexpr auto _16 = std::integral_constant<placeholder_type, 16>{};
constexpr auto _17 = std::integral_constant<placeholder_type, 17>{};
constexpr auto _18 = std::integral_constant<placeholder_type, 18>{};
constexpr auto _19 = std::integral_constant<placeholder_type, 19>{};
constexpr auto _20 = std::integral_constant<placeholder_type, 20>{};
constexpr auto _21 = std::integral_constant<placeholder_type, 21>{};
constexpr auto _22 = std::integral_constant<placeholder_type, 22>{};
constexpr auto _23 = std::integral_constant<placeholder_type, 23>{};
constexpr auto _24 = std::integral_constant<placeholder_type, 24>{};
constexpr auto _25 = std::integral_constant<placeholder_type, 25>{};
constexpr auto _26 = std::integral_constant<placeholder_type, 26>{};
constexpr auto _27 = std::integral_constant<placeholder_type, 27>{};
constexpr auto _28 = std::integral_constant<placeholder_type, 28>{};
constexpr auto _29 = std::integral_constant<placeholder_type, 29>{};
constexpr auto _30 = std::integral_constant<placeholder_type, 30>{};
constexpr auto _31 = std::integral_constant<placeholder_type, 31>{};
} // namespace placeholder
using place_id = typename placeholder::placeholder_type;

} // namespace mn

#endif