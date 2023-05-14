#ifndef CONTROL_FLOW_H
#define CONTROL_FLOW_H
#include <initializer_list>

#include "Meta.h"

namespace mn {

// enable_if
// conditional
template<bool B>
struct conditional_impl {
	template<class T, class F>
	using type = T;
};
template<>
struct conditional_impl<false> {
	template<class T, class F>
	using type = F;
};

template<bool B, class T, class F>
using conditional_t = typename conditional_impl<B>::type<T, F>;

template<bool B>
struct do_if;
template<>
struct do_if<true> {
	template<typename F, typename... Args>
	auto operator()(F&& f, Args&&... args) {
		return std::forward<F>(f)(std::forward<Args>(args)...);
	}
};
template<>
struct do_if<false> {
	template<typename F, typename... Args>
	auto operator()(F&& f, Args&&... args) {}
};

}// namespace mn

#endif