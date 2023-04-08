#ifndef __META_H_
#define __META_H_

#include <initializer_list>
#include <type_traits>

namespace mn {

/// initializer list & function overload
/// or
constexpr bool logic_or(std::initializer_list<std::false_type>) {
  return false;
}
constexpr bool logic_or(std::initializer_list<bool>) { return true; }
/// and
constexpr bool logic_and(std::initializer_list<std::true_type>) { return true; }
constexpr bool logic_and(std::initializer_list<bool>) { return false; }

template <template <class T> class Predicate, typename... Ts>
constexpr bool all_satisfy() {
  return logic_and({typename Predicate<Ts>::type()...});
}
template <template <class T> class Predicate, typename... Ts>
constexpr bool any_satisfy() {
  return logic_or({typename Predicate<Ts>::type()...});
}

/// pre C++14 impl, https://zh.cppreference.com/w/cpp/types/void_t
/// check ill-formed types
template <typename... Ts> struct make_void { using type = void; };
template <typename... Ts> using void_t = typename make_void<Ts...>::type;

/// SFINAE
template <bool B> struct enable_if;
template <> struct enable_if<true> { using type = char; };
template <bool B> using enable_if_t = typename enable_if<B>::type;
template <bool... Bs>
using enable_if_all = typename enable_if<std::conjunction<std::integral_constant<bool,Bs>...>::value>::type;
template <bool... Bs>
using enable_if_any = typename enable_if<logic_or(
    {std::integral_constant<bool, Bs>()...})>::type;
/// underlying_type
/// common_type

} // namespace mn

#endif