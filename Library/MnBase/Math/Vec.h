#ifndef __MN_VEC_H_
#define __MN_VEC_H_
#include <MnBase/Meta/MathMeta.h>
#include <MnBase/Meta/Meta.h>
#include <MnBase/Meta/PatternMeta.h>
#include <MnBase/Meta/Relationship.h>
#include <functional>
#include <type_traits>
#include <utility>

namespace mn {

struct identity_vec {};

/// declarations
template <typename> struct indexer_impl;
template <typename T, typename Extents> struct vec_view;
template <typename T, typename Extents> struct vec_impl;
template <typename T, int... Ns>
using vec = vec_impl<T, std::integer_sequence<int, Ns...>>;

/// indexer
template <typename Tn, Tn... Ns>
struct indexer_impl<std::integer_sequence<Tn, Ns...>> {
  static constexpr auto dim = sizeof...(Ns);
  static constexpr auto extent = integral_seq_mul<Tn, Ns...>::value;
  using index_type = Tn;
  using extents = std::integer_sequence<Tn, Ns...>;
  template <place_id I>
  static constexpr Tn range(std::integral_constant<place_id, I>) {
    return select_value<I, extents>::value; // select_indexed_value<I, Tn,
                                            // Ns...>::value;
  }
  template <std::size_t... Is, typename... Args>
  static constexpr Tn offset_impl(std::index_sequence<Is...>, Args &&... args) {
    Tn idx = 0;
    (void)std::initializer_list<int>{
        (idx += std::forward<Args>(args) * excl_suffix_mul<Is, extents>{},
         0)...};
    return idx;
  }
  template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
  static constexpr Tn offset(Args &&... args) {
    return offset_impl(std::index_sequence_for<Args...>{},
                       std::forward<Args>(args)...);
  }
};
template <typename Tn, Tn... Ns>
using indexer = indexer_impl<std::integer_sequence<Tn, Ns...>>;

/// vec without lifetime managing
template <typename T, typename Tn, Tn... Ns>
struct vec_view<T, std::integer_sequence<Tn, Ns...>> : identity_vec,
                                                       indexer<Tn, Ns...> {
  using base_t = indexer<Tn, Ns...>;
  using base_t::dim;
  using base_t::extent;
  using value_type = T;
  using base_t::offset;
  using typename base_t::extents;
  using typename base_t::index_type;

  constexpr vec_view() = delete;
  explicit constexpr vec_view(T *ptr) : _data{ptr} {}

  /// random access
  // ()
  template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
  constexpr T &operator()(Args &&... args) noexcept {
    return _data[offset(std::forward<Args>(args)...)];
  }
  template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
  constexpr const T &operator()(Args &&... args) const noexcept {
    return _data[offset(std::forward<Args>(args)...)];
  }
  // []
  template <
      typename Index,
      typename R = vec_view<
          T, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>,
                      extents>>,
      Tn d = dim, enable_if_t<(d > 1)> = 0>
  constexpr R operator[](Index &&index) noexcept {
    return R{_data + offset(std::forward<Index>(index))};
  }
  template <typename Index,
            typename R = vec_view<
                std::add_const_t<T>,
                gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>,
                         extents>>,
            Tn d = dim, enable_if_t<(d > 1)> = 0>
  constexpr R operator[](Index index) const noexcept {
    return R{_data + offset(std::forward<Index>(index))};
  }
  template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
  constexpr T &operator[](Index index) noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
  constexpr const T &operator[](Index index) const noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index> constexpr T &val(Index index) noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index> constexpr const T &val(Index index) const noexcept {
    return _data[std::forward<Index>(index)];
  }

private:
  T *_data;
};

/// vec
template <typename T, typename Tn, Tn... Ns>
struct vec_impl<T, std::integer_sequence<Tn, Ns...>> : identity_vec,
                                                       indexer<Tn, Ns...> {
  static_assert(std::is_trivial<T>::value,
                "Vec element type is not trivial!\n");
  using base_t = indexer<Tn, Ns...>;
  using base_t::dim;
  using base_t::extent;
  using value_type = T;
  using base_t::offset;
  using typename base_t::extents;
  using typename base_t::index_type;

private:
  T _data[extent];

public:
  /// expose internal
  constexpr auto data() noexcept -> T * { return _data; }
  constexpr auto data() const noexcept -> const T * { return _data; }

  /// construct
  constexpr vec_impl() = default;
  template <typename... Vals,
            enable_if_all<(sizeof...(Vals)) == extent,
                          std::is_convertible<Vals, value_type>::value...> = 0>
  explicit constexpr vec_impl(Vals &&... vals) noexcept
      : _data{std::forward<Vals>(vals)...} {}
  template <
      typename Val, decltype(extent) cnt = extent,
      enable_if_all<(cnt > 1), std::is_convertible<Val, value_type>::value> = 0>
  explicit constexpr vec_impl(Val &&val) noexcept {
    for (Tn idx = 0; idx < extent; ++idx)
      _data[idx] = std::forward<Val>(val);
  }
  constexpr void set(T val) noexcept {
    for (Tn idx = 0; idx < extent; ++idx)
      _data[idx] = val;
  }
  /// random access
  // ()
  template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
  constexpr T &operator()(Args &&... args) noexcept {
    return _data[offset(std::forward<Args>(args)...)];
  }
  template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
  constexpr const T &operator()(Args &&... args) const noexcept {
    return _data[offset(std::forward<Args>(args)...)];
  }
  // []
  template <
      typename Index,
      typename R = vec_view<
          T, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>,
                      extents>>,
      Tn d = dim, enable_if_t<(d > 1)> = 0>
  constexpr R operator[](Index &&index) noexcept {
    return R{_data + offset(std::forward<Index>(index))};
  }
  template <typename Index,
            typename R = vec_view<
                std::add_const_t<T>,
                gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>,
                         extents>>,
            Tn d = dim, enable_if_t<(d > 1)> = 0>
  constexpr R operator[](Index index) const noexcept {
    return R{_data + offset(std::forward<Index>(index))};
  }
  template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
  constexpr T &operator[](Index index) noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
  constexpr const T &operator[](Index index) const noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index> constexpr T &val(Index index) noexcept {
    return _data[std::forward<Index>(index)];
  }
  template <typename Index> constexpr const T &val(Index index) const noexcept {
    return _data[std::forward<Index>(index)];
  }
  ///

  template <typename TT> constexpr auto cast() const noexcept {
    vec_impl<TT, extents> r{};
    for (Tn idx = 0; idx < extent; ++idx)
      r.val(idx) = _data[idx];
    return r;
  }
  /// compare
  template <typename Vec,
            enable_if_all<
                std::is_base_of<identity_vec, Vec>::value,
                is_same<typename Vec::extents, extents>::value,
                std::is_convertible<T, typename Vec::value_type>::value> = 0>
  constexpr auto operator==(Vec &&o) noexcept -> bool {
    for (int i = 0; i < extent; ++i)
      if (_data[i] != std::forward<Vec>(o).val(i))
        return false;
    return true;
  }
  template <typename Vec,
            enable_if_all<
                std::is_base_of<identity_vec, Vec>::value,
                is_same<typename Vec::extents, extents>::value,
                std::is_convertible<T, typename Vec::value_type>::value> = 0>
  constexpr auto operator!=(Vec &&o) noexcept -> bool {
    for (int i = 0; i < extent; ++i)
      if (_data[i] == std::forward<Vec>(o).val(i))
        return false;
    return true;
  }

  /// linalg
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr R dot(vec_impl<TT, extents> const &o) const noexcept {
    R res{0};
    for (int i = 0; i < extent; ++i)
      res += _data[i] * o.val(i);
    return res;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr R dot(vec_impl<TT, extents> &&o) const noexcept {
    R res{0};
    for (int i = 0; i < extent; ++i)
      res += _data[i] * o.val(i);
    return res;
  }
  constexpr T prod() noexcept {
    T res{1};
    for (int i = 0; i < extent; ++i)
      res *= _data[i];
    return res;
  }
  constexpr T l2NormSqr() noexcept {
    T res{0};
    for (int i = 0; i < extent; ++i)
      res += _data[i] * _data[i];
    return res;
  }
  constexpr T infNormSqr() noexcept {
    T res{0};
    for (int i = 0; i < extent; ++i) {
      T sqr = _data[i] * _data[i];
      if (sqr > res)
        res = sqr;
    }
    return res;
  }

  /// borrowed from
  /// https://github.com/cemyuksel/cyCodeBase/blob/master/cyIVector.h
  /// east const
  //!@name Unary operators
  constexpr vec_impl operator-() const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = -_data[i];
    return r;
  }

  //!@name Binary operators
  // scalar
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator+(vec_impl const &e, TT const v) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) + v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator+(TT const v, vec_impl const &e) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) + v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator-(vec_impl const &e, TT const v) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) - v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator-(TT const v, vec_impl const &e) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = v - e.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator*(vec_impl const &e, TT const v) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) * v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator*(TT const v, vec_impl const &e) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) * v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator/(vec_impl const &e, TT const v) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = e.val(i) / v;
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  friend constexpr auto operator/(TT const v, vec_impl const &e) noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = v / e.val(i);
    return r;
  }
  // vector
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator+(vec_impl<TT, extents> const &o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] + o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator+(vec_impl<TT, extents> &&o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] + o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator-(vec_impl<TT, extents> const &o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] - o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator-(vec_impl<TT, extents> &&o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] - o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator*(vec_impl<TT, extents> const &o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] * o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator*(vec_impl<TT, extents> &&o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] * o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator/(vec_impl<TT, extents> const &o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] / o.val(i);
    return r;
  }
  template <typename TT, typename R = std::common_type_t<T, TT>>
  constexpr auto operator/(vec_impl<TT, extents> &&o) const noexcept {
    vec_impl<R, extents> r{};
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] / o.val(i);
    return r;
  }
  //!@name Assignment operators
  // scalar
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator+=(TT &&v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] += std::forward<TT>(v);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator-=(TT &&v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] -= std::forward<TT>(v);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator*=(TT &&v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] *= std::forward<TT>(v);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator/=(TT &&v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] /= std::forward<TT>(v);
    return *this;
  }
  // vector
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator+=(vec_impl<TT, extents> const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] += o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator+=(vec_impl<TT, extents> &&o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] += o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator-=(vec_impl<TT, extents> const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] -= o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator-=(vec_impl<TT, extents> &&o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] -= o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator*=(vec_impl<TT, extents> const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] *= o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator*=(vec_impl<TT, extents> &&o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] *= o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator/=(vec_impl<TT, extents> const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] /= o.val(i);
    return *this;
  }
  template <typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
  constexpr vec_impl &operator/=(vec_impl<TT, extents> &&o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] /= o.val(i);
    return *this;
  }

#if 0
  //!@name Bitwise operators
  vec_impl operator<<(vec_impl const &o) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] << o.val(i);
    return r;
  }
  vec_impl operator>>(vec_impl const &o) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] >> o.val(i);
    return r;
  }
  vec_impl operator&(vec_impl const &o) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] & o.val(i);
    return r;
  }
  vec_impl operator|(vec_impl const &o) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] | o.val(i);
    return r;
  }
  vec_impl operator^(vec_impl const &o) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] ^ o.val(i);
    return r;
  }
  vec_impl operator<<(T const v) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] << v;
    return r;
  }
  vec_impl operator>>(T const v) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] >> v;
    return r;
  }
  vec_impl operator&(T const v) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] & v;
    return r;
  }
  vec_impl operator|(T const v) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] | v;
    return r;
  }
  vec_impl operator^(T const v) const noexcept {
    vec_impl r;
    for (int i = 0; i < extent; ++i)
      r.val(i) = _data[i] ^ v;
    return r;
  }

  //!@name Bitwise Assignment operators
  vec_impl &operator<<=(vec_impl const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] <<= o.val(i);
    return *this;
  }
  vec_impl &operator>>=(vec_impl const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] >>= o.val(i);
    return *this;
  }
  vec_impl &operator&=(vec_impl const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] &= o.val(i);
    return *this;
  }
  vec_impl &operator|=(vec_impl const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] |= o.val(i);
    return *this;
  }
  vec_impl &operator^=(vec_impl const &o) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] ^= o.val(i);
    return *this;
  }
  vec_impl &operator<<=(T const v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] <<= v;
    return *this;
  }
  vec_impl &operator>>=(T const v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] >>= v;
    return *this;
  }
  vec_impl &operator&=(T const v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] &= v;
    return *this;
  }
  vec_impl &operator|=(T const v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] |= v;
    return *this;
  }
  vec_impl &operator^=(T const v) noexcept {
    for (int i = 0; i < extent; ++i)
      _data[i] ^= v;
    return *this;
  }
#endif
};

/// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
/// vector(vec+{0}) homogeneous coordinates

} // namespace mn

#endif