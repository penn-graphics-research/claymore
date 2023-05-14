#ifndef TYPE_DECORATION_H
#define TYPE_DECORATION_H

#include "Meta.h"

namespace mn {

/// decay
/// const, volatile, reference, pointer, sign, extent
template<typename T>
using add_const_t = const T;
}// namespace mn

#endif