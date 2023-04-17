#ifndef FUNCTION_H
#define FUNCTION_H
#include <functional>
#include <tl/function_ref.hpp>
#include <tuple>

#include "MnBase/Meta/TypeMeta.h"

namespace mn {

//typename <typename R, typename... Args>
//using func_ref = tl::function_ref<R(Args...)>;

template<typename T>
struct FunctionTraits;

template<typename R, typename... Args>
struct FunctionTraits<std::function<R(Args...)>> {
	static constexpr std::size_t nargs = sizeof...(Args);

	using resultType = R;

	template<std::size_t i>
	using argType = AttribType<i, Args...>;
};

}// namespace mn

#endif