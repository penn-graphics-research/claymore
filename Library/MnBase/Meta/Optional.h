#ifndef OPTIONAL_H
#define OPTIONAL_H
//#include <mapbox/optional.hpp>
#include <tl/optional.hpp>

namespace mn {

// template <typename T> using optional = mapbox::util::optional<T>;
template<typename T>
using optional = tl::optional<T>;
}// namespace mn

#endif