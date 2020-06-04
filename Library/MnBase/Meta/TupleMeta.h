#ifndef __TUPLE_META_H_
#define __TUPLE_META_H_

#include "MathMeta.h"
#include "Meta.h"
#include "PatternMeta.h"
#include "TypeMeta.h"
#include <array>
#include <tuple>
#include <utility>
#include <vector>

namespace mn {

/// Jorg Brown, Cppcon2019, reducing template compilation overhead using
/// features from C++11, 14, 17, and 20
struct identity_tuple {};
template <std::size_t I, typename T> struct tuple_value {
  T value;
  constexpr tuple_value() = default;
  template <typename... Args>
  explicit constexpr tuple_value(Args &&... args)
      : value{std::forward<Args>(args)...} {}
  // constexpr auto &get(std::integral_constant<std::size_t, I>) noexcept {
  constexpr T &get() noexcept { return value; }
  constexpr const T &get() const noexcept { return value; }
  static constexpr std::size_t tuple_size = 1;
};

template <typename... T> struct tuple_base;

template <std::size_t... Is, typename... Ts>
struct tuple_base<std::index_sequence<Is...>, Ts...> : tuple_value<Is, Ts>... {
  static constexpr std::size_t tuple_size = sizeof...(Ts);
  using tuple_types = type_seq<Ts...>;
  // explicit constexpr tuple_base(Ts &&... args) : tuple_leaf<Is, Ts>(args)...
  // {}
  constexpr tuple_base() = default;
  template <typename... Args>
  explicit constexpr tuple_base(Args &&... args)
      : tuple_value<Is, Ts>(std::forward<Args>(args))... {}
  /// function overload, disabled due to c++17 feature using expansion
  // using tuple_leaf<Is, Ts>::get...;
  // template <std::size_t I> auto &get() {
  //  return get(std::integral_constant<std::size_t, I>{});
  //}
  /// c++14 impl
  template <std::size_t I, typename T>
  constexpr auto &get_element(tuple_value<I, T> *leaf) noexcept {
    return *leaf;
  }
  template <std::size_t I>
  constexpr typename tuple_types::type<I> &get() noexcept {
    return get_element<I>(this).value;
  }
  template <std::size_t I>
  constexpr const typename tuple_types::type<I> &get() const noexcept {
    return get_element<I>(this).value;
  }
};

template <typename TypeSeq> struct tuple_impl;
template <typename... Ts> struct tuple_impl<type_seq<Ts...>> : identity_tuple {
  using base_t = tuple_base<std::index_sequence_for<Ts...>, Ts...>;
  static constexpr std::size_t tuple_size = sizeof...(Ts);
  using tuple_types = type_seq<Ts...>;

  constexpr tuple_impl() = default;
  template <typename... Args>
  explicit constexpr tuple_impl(Args &&... args)
      : base{std::forward<Args>(args)...} {}

  base_t base;
  template <std::size_t I> constexpr auto &get() noexcept {
    return base.template get<I>();
  }
  template <std::size_t I> constexpr const auto &get() const noexcept {
    return base.template get<I>();
  }
};
template <typename... Ts> using tuple = tuple_impl<type_seq<Ts...>>;

template <typename T> using is_tuple = std::is_base_of<identity_tuple, T>;

template <typename T>
using tuple_size = std::integral_constant<std::size_t, T::tuple_size>;

template <std::size_t I, typename T>
using tuple_element = select_type<I, typename T::tuple_types>;
/// get (ridiculous syntax)
template <std::size_t I, typename TypeSeq>
constexpr const auto &get(const tuple_impl<TypeSeq> &t) noexcept {
  return t.template get<I>();
  // return static_cast<const tuple_value<I, typename TypeSeq::type<I>>
  // &>(t.base).get();
}
template <std::size_t I, typename TypeSeq>
constexpr typename TypeSeq::type<I> &get(tuple_impl<TypeSeq> &t) noexcept {
  return t.template get<I>();
  // return static_cast<tuple_value<I, typename TypeSeq::type<I>>
  // &>(t.base).get();
}
template <std::size_t I, typename TypeSeq>
constexpr typename TypeSeq::type<I> &&get(tuple_impl<TypeSeq> &&t) noexcept {
  return std::move(t).template get<I>();
  // return static_cast<tuple_value<I, typename TypeSeq::type<I>>
  // &&>(std::move(t.base)).get();
}

/// make_tuple
template <typename... Args, typename R = tuple<special_decay_t<Args>...>>
constexpr R make_tuple(Args &&... args) {
  return R{std::forward<Args>(args)...};
}
/// forward as tuple
template <typename... Ts, typename R = tuple<Ts &&...>>
constexpr R tuple_forward(Ts &&... ts) noexcept {
  return R{std::forward<Ts>(ts)...};
}

/// taocpp
namespace meta_impl {

template <std::size_t M, typename Seq> struct count_less_or_equal;
template <std::size_t M, std::size_t... Ns>
struct count_less_or_equal<M, std::index_sequence<Ns...>>
    : integral_seq_sum<std::size_t, ((Ns <= M) ? 1 : 0)...> {};

template <typename, typename> struct expand;
template <typename Is, typename Seq>
using expand_t = typename expand<Is, Seq>::type;
template <typename Seq, std::size_t... Is>
struct expand<std::index_sequence<Is...>, Seq>
    : std::index_sequence<count_less_or_equal<Is, Seq>::value...> {
  using type = std::index_sequence<count_less_or_equal<Is, Seq>::value...>;
};

template <typename...> struct tuple_cat_result;

template <std::size_t... Os, std::size_t... Is, typename... Ts>
struct tuple_cat_result<std::index_sequence<Os...>, std::index_sequence<Is...>,
                        Ts...> {
  using type =
      tuple<typename select_indexed_type<Os, Ts...>::tuple_types::type<Is>...>;
};
template <typename... Ts>
using tuple_cat_result_t = typename tuple_cat_result<Ts...>::type;

template <typename... Ts> struct tuple_cat_helper {
  using tuple_index_seq = std::index_sequence<sizeof...(Ts)>;
  using tuple_size_seq = std::index_sequence<tuple_size<Ts>::value...>;

  using result_index_seq = std::make_index_sequence<
      integral_seq_sum<std::size_t, tuple_size<Ts>::value...>::value>;

  using outer_index_seq =
      expand_t<result_index_seq, incl_scan_seq_t<tuple_size_seq>>;
  using inner_index_seq =
      minus_seq_t<result_index_seq,
                  gather_t<outer_index_seq, excl_scan_seq_t<tuple_size_seq>>>;
  using result_type =
      tuple_cat_result_t<outer_index_seq, inner_index_seq, Ts...>;
};
template <typename R, std::size_t... Os, std::size_t... Is, typename Tuple>
constexpr decltype(auto) tuple_cat_impl(std::index_sequence<Os...>,
                                        std::index_sequence<Is...>,
                                        Tuple &&tup) {
  return R{get<Is>(get<Os>(tup))...};
}
} // namespace meta_impl

template <typename... Ts,
          typename H = meta_impl::tuple_cat_helper<
              typename std::remove_reference<Ts>::type...>,
          typename R = typename H::result_type>
constexpr R tuple_cat(Ts &&... tuples) {
  return meta_impl::tuple_cat_impl<R>(
      typename H::outer_index_seq{}, typename H::inner_index_seq{},
      tuple_forward(std::forward<Ts>(tuples)...));
}

/// legacy
// template <std::size_t I, typename Tuple>
// using TupleAttribType = std::decay_t<tuple_element<I, Tuple>>;

template <typename Tuple> struct tuple_type_list {
  template <std::size_t N>
  using type = typename std::tuple_element<N, Tuple>::type;
};

template <std::size_t i, typename Tuple>
using TupleAttribType =
    std::decay_t<typename tuple_type_list<Tuple>::template type<i>>;

template <typename T, std::size_t... I>
constexpr decltype(auto) gen_uniform_tuple(std::index_sequence<I...>) {
  using Tuple = std::tuple<std::enable_if_t<I >= 0, T>...>;
  return Tuple{};
};
template <typename T, unsigned N>
using UniformTuple =
    decltype(gen_uniform_tuple<std::decay_t<T>>(std::make_index_sequence<N>{}));

template <std::size_t I, typename Tuple> struct tupleOffset;

template <std::size_t I, typename Tuple> struct tupleOffset {
  static constexpr std::size_t value =
      tupleOffset<I - 1, Tuple>::value + sizeof(TupleAttribType<I - 1, Tuple>);
};
template <typename Tuple> struct tupleOffset<0, Tuple> {
  static constexpr std::size_t value = 0;
};
template <typename Tuple, std::size_t... Is>
constexpr void
gen_tuple_offsets_impl(std::array<std::size_t, sizeof...(Is)> &offsets,
                       std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      (offsets[Is] = tupleOffset<Is, Tuple>::value, 0)...};
}

template <typename Tuple> constexpr auto gen_tuple_offsets() {
  auto tuplesize = std::tuple_size<Tuple>::value;
  std::array<std::size_t, std::tuple_size<Tuple>::value> offsets;
  using Indices = std::make_index_sequence<std::tuple_size<Tuple>::value>;
  gen_tuple_offsets_impl<Tuple>(offsets, Indices{});
  return offsets;
}

template <typename Tuple, typename Func, std::size_t... I>
void forIndexAlloc(Func &&f, std::index_sequence<I...>) {
  return (void)std::initializer_list<int>{
      (std::forward<Func>(f)(std::forward<std::size_t>(I),
                             sizeof(TupleAttribType<I, Tuple>)),
       0)...};
}
/// tuple functionality
template <typename Func, typename... Args>
void forArgs(Func &&f, Args &&... args) {
  return (void)std::initializer_list<int>{(f(std::forward<Args>(args)), 0)...};
}

template <typename F, typename Tuple, std::size_t... I>
decltype(auto) apply_impl(F &&f, Tuple t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}
template <typename F, typename Tuple> decltype(auto) apply(F &&f, Tuple t) {
  using Indices =
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
  return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
}
template <typename Func, typename Tpl> void forTuple(Func &&f, Tpl &&tpl) {
  apply([&f](auto &&... xs) { forArgs(f, std::forward<decltype(xs)>(xs)...); },
        std::forward<Tpl>(tpl));
}

template <typename T, std::size_t... I>
constexpr decltype(auto) gen_uniform_channel_tuple(std::index_sequence<I...>) {
  // static_assert((sizeof(T) & 0x3) == 0);
  using Tuple = std::tuple<std::conditional_t<
      I == 0, std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>, T>...>;
  return Tuple{};
};

} // namespace mn

#endif
