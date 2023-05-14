#ifndef TYPE_OPERATION_H
#define TYPE_OPERATION_H

#include "Meta.h"

namespace mn {

template<typename T, typename = void>
struct is_iterable : std::false_type {};
template<typename T>
struct is_iterable<T, void_t<decltype(*std::begin(std::declval<T>())), decltype(*std::end(std::declval<T>()))>> : std::true_type {};

}// namespace mn

#endif