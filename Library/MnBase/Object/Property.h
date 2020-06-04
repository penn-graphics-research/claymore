#ifndef __PROPERTY_H_
#define __PROPERTY_H_

#include <type_traits>

namespace mn {

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

template <typename Base, typename... Ts> struct is_base_of;
template <typename Base, typename T> struct is_base_of<Base, T> {
  static_assert(std::is_base_of<Base, T>::value,
                "T is not a subclass of Base!");
};
template <typename Base, typename T, typename... Ts>
struct is_base_of<Base, T, Ts...> : public is_base_of<Base, Ts...> {
  static_assert(std::is_base_of<Base, T>::value,
                "T is not a subclass of Base!");
};

template <template <class T> class Feature, typename... Ts> struct satisfy;
template <template <class T> class Feature, typename T>
struct satisfy<Feature, T> {
  static_assert(Feature<T>::value, "T does not satisfy the feature!");
};
template <template <class T> class Feature, typename T, typename... Ts>
struct satisfy<Feature, T, Ts...> : public satisfy<Feature, Ts...> {
  static_assert(Feature<T>::value, "T does not satisfy the feature!");
};

} // namespace mn

#endif