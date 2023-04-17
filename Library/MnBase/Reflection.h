#ifndef REFLECTION_H
#define REFLECTION_H

#include <typeinfo>

namespace mn {

/// compile-time type inspection
template<class T>
class that_type;

template<class T>
void name_that_type(T& param) {
	that_type<T> tType;
	that_type<decltype(param)> paramType;
}

/// runtime
//typeid(x).name()
template<typename T>
constexpr decltype(auto) query_type_name(T&& x) {
	return typeid(std::forward<T>(x)).name();
}

}// namespace mn

#endif
