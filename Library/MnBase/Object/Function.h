#ifndef __FUNCTION_H_
#define __FUNCTION_H_
#include <tl/function_ref.hpp>
#include <MnBase/Meta/TypeMeta.h>
#include <functional>
#include <tuple>

namespace mn {

    //typename <typename R, typename... Args>
    //using func_ref = tl::function_ref<R(Args...)>;

template <typename T>
struct FunctionTraits;

template <typename R, typename... Args>
struct FunctionTraits<std::function<R(Args...)>> {
    static constexpr std::size_t nargs = sizeof...(Args);

    using resultType = R;

    template <std::size_t i>
    using argType = AttribType<i, Args...>;
};

} // namespace mn

#endif