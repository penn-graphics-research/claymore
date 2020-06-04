#ifndef __MATH_META_H_
#define __MATH_META_H_

#include "Meta.h"
#include "TypeMeta.h"
#include <MnBase/Math/Bit/Bits.h>
#include <type_traits>
#include <utility>

namespace mn {

template <typename T, T I> struct next_2pow {
  static_assert(std::is_integral<T>::value,
                "next_2pow: template parameter not an integral!\n");
  static constexpr T value = static_cast<T>(1) << bit_count(I);
};

template <typename T, T... Is> struct mul;
template <typename T, T I> struct mul<T, I> : std::integral_constant<T, I> {};
template <typename T, T I, T... Is>
struct mul<T, I, Is...> : std::integral_constant<T, I * mul<T, Is...>::value> {
};

template <typename T, T... Is> struct add;
template <typename T, T I> struct add<T, I> : std::integral_constant<T, I> {};
template <typename T, T I, T... Is>
struct add<T, I, Is...> : std::integral_constant<T, I + add<T, Is...>::value> {
};

/// use some designs from https://github.com/taocpp/sequences/
namespace meta_impl {
template <typename T, T... Ns> constexpr T integral_seq_mul_impl() noexcept {
  T res{1};
  (void)std::initializer_list<int>{(res *= Ns, 0)...};
  return res;
}
template <typename T, T... Ns> constexpr T integral_seq_sum_impl() noexcept {
  T res{0};
  (void)std::initializer_list<int>{(res += Ns, 0)...};
  return res;
}
} // namespace meta_impl

template <typename T> struct value_seq_traits_impl;
template <typename T, T... Ns>
struct value_seq_traits_impl<std::integer_sequence<T, Ns...>> {
  static constexpr T sum = meta_impl::integral_seq_sum_impl<T, Ns...>();
  static constexpr T prod = meta_impl::integral_seq_mul_impl<T, Ns...>();
};
template <typename T, T... Ns>
using value_seq_traits = value_seq_traits_impl<std::integer_sequence<T, Ns...>>;

/// seq reduction
template <typename T, T... Ns>
using integral_seq_sum = std::integral_constant<
    T, value_seq_traits<T, ((Ns > 0) ? Ns : 0)...>::sum -
           value_seq_traits<T, ((Ns < 0) ? -Ns : 0)...>::sum>;
template <typename T, T... Ns>
using integral_seq_mul =
    std::integral_constant<T, value_seq_traits<T, Ns...>::prod>;

template <typename T, T... Ns> struct integral_seq_max;
template <typename T, T N>
struct integral_seq_max<T, N> : std::integral_constant<T, N> {};
template <typename T, T N, T... Ns>
struct integral_seq_max<T, N, Ns...>
    : std::integral_constant<T, (N > integral_seq_max<T, Ns...>::value)
                                    ? N
                                    : integral_seq_max<T, Ns...>::value> {};

/// prefix sum
template <std::size_t I, typename Seq,
          typename Indices = std::make_index_sequence<Seq::size()>>
struct excl_prefix_sum;
template <std::size_t I, typename Seq,
          typename Indices = std::make_index_sequence<Seq::size()>>
struct incl_prefix_sum;

template <std::size_t I, typename T, T... Ns, std::size_t... Is>
struct excl_prefix_sum<I, std::integer_sequence<T, Ns...>,
                       std::index_sequence<Is...>>
    : integral_seq_sum<T, (Is < I ? Ns : 0)...> {
  static_assert(I < sizeof...(Ns), "excl_prefix_sum out-of-bound access ");
  using type = integral_seq_sum<T, (Is < I ? Ns : 0)...>;
};
template <std::size_t I, typename T, T... Ns, std::size_t... Is>
struct incl_prefix_sum<I, std::integer_sequence<T, Ns...>,
                       std::index_sequence<Is...>>
    : integral_seq_sum<T, (Is <= I ? Ns : 0)...> {
  static_assert(I < sizeof...(Ns), "incl_prefix_sum out-of-bound access ");
  using type = integral_seq_sum<T, (Is <= I ? Ns : 0)...>;
};
/// suffix mul
template <std::size_t I, typename Seq,
          typename Indices = std::make_index_sequence<Seq::size()>>
struct excl_suffix_mul;

template <std::size_t I, typename T, T... Ns, std::size_t... Is>
struct excl_suffix_mul<I, std::integer_sequence<T, Ns...>,
                       std::index_sequence<Is...>>
    : integral_seq_mul<T, (Is > I ? Ns : 1)...> {
  static_assert(I < sizeof...(Ns), "excl_suffix_mul out-of-bound access ");
  using type = integral_seq_mul<T, (Is > I ? Ns : 1)...>;
};

/// seq scan(sum)
template <typename Seq, typename = std::make_index_sequence<Seq::size()>>
struct excl_scan_seq;
template <typename Seq, std::size_t... Is>
struct excl_scan_seq<Seq, std::index_sequence<Is...>>
    : std::integer_sequence<typename Seq::value_type,
                            excl_prefix_sum<Is, Seq>::value...> {
  using type = std::integer_sequence<typename Seq::value_type,
                                     excl_prefix_sum<Is, Seq>::value...>;
};
template <typename Seq>
using excl_scan_seq_t = typename excl_scan_seq<Seq>::type;
template <typename Seq, typename = std::make_index_sequence<Seq::size()>>
struct incl_scan_seq;
template <typename Seq, std::size_t... Is>
struct incl_scan_seq<Seq, std::index_sequence<Is...>>
    : std::integer_sequence<typename Seq::value_type,
                            incl_prefix_sum<Is, Seq>::value...> {
  using type = std::integer_sequence<typename Seq::value_type,
                                     incl_prefix_sum<Is, Seq>::value...>;
};
template <typename Seq>
using incl_scan_seq_t = typename incl_scan_seq<Seq>::type;

/// seq coefficient-wise arithmetic
template <typename Seq0, typename Seq1,
          typename = std::make_index_sequence<Seq0::size()>>
struct minus_seq;
template <typename Seq0, typename Seq1, std::size_t... Is>
struct minus_seq<Seq0, Seq1, std::index_sequence<Is...>> {
  static_assert(Seq0::size() == Seq1::size(), "minus_seq error");
  using type = std::integer_sequence<
      std::common_type_t<typename Seq0::value_type, typename Seq1::value_type>,
      (select_value<Is, Seq0>::value - select_value<Is, Seq1>::value)...>;
};
template <typename Seq0, typename Seq1>
using minus_seq_t = typename minus_seq<Seq0, Seq1>::type;

template <typename ValueSeq, int Offset> struct translate_seq;
template <typename Tn, Tn... Ns, int Offset>
struct translate_seq<std::integer_sequence<Tn, Ns...>, Offset> {
  using type = std::integer_sequence<Tn, (Tn)(Ns + Offset)...>;
};
template <typename ValueSeq, int Offset>
using translate_seq_t = typename translate_seq<ValueSeq, Offset>::type;

} // namespace mn

#endif