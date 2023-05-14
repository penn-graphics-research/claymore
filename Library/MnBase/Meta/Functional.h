#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <functional>
#include <optional>

#include "Meta.h"
#include "TupleMeta.h"

namespace mn {

template<typename F, typename T>
constexpr decltype(auto) fold(F&& f, T&& v) {
	return std::forward<T>(v);
}

template<typename F, typename T, typename... Ts>
constexpr decltype(auto) fold(F&& f, T&& v, Ts&&... vs) {
	return std::forward<F>(f)(std::forward<T>(v), fold(std::forward<F>(f), std::forward<Ts>(vs)...));
}

/// map, transform

/// result_of

template<typename T>
struct add_optionality_t {
	using type = std::optional<T>;
};

template<typename T>
struct add_optionality_t<std::optional<T>> {
	using type = std::optional<T>;
};

template<typename T>
using add_optionality = typename add_optionality_t<T>::type;

template<typename T, typename Func>
auto operator|(std::optional<T> const& opt, Func const& func) -> add_optionality<decltype(func(*opt))> {
	if(opt) {
		return func(*opt);
	} else {
		return {};
	}
}

template<class T, class F, std::enable_if_t<detail::is_optional<std::decay_t<T>>::value, int> = 0>
auto operator|(T&& t, F&& f) -> decltype(detail::void_or_nullopt<decltype(f(std::forward<T>(t).operator*()))>()) {
	using return_type = decltype(f(std::forward<T>(t).operator*()));
	if(t) {
		return f(std::forward<T>(t).operator*());
	} else {
		return detail::void_or_nullopt<return_type>();
	}
}

}// namespace mn

#endif
