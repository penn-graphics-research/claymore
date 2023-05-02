#ifndef MN_VEC_H
#define MN_VEC_H
#include <functional>
#include <type_traits>
#include <utility>
#include <array>

#include "MnBase/Meta/MathMeta.h"
#include "MnBase/Meta/Meta.h"
#include "MnBase/Meta/PatternMeta.h"
#include "MnBase/Meta/Relationship.h"

#define ENABLE_BITWISE_OPERATORS 0

namespace mn {

struct identity_vec {};

/// declarations
template<typename>
struct indexer_impl;

template<typename T, typename Extents>
struct vec_view;

template<typename T, typename Extents>
struct vec_impl;

template<typename Tn, Tn... Ns>
using indexer = indexer_impl<std::integer_sequence<Tn, Ns...>>;

template<typename T, int... Ns>
using vec = vec_impl<T, std::integer_sequence<int, Ns...>>;

/// indexer
template<typename Tn, Tn... Ns>
struct indexer_impl<std::integer_sequence<Tn, Ns...>> {
	static constexpr auto dim	 = sizeof...(Ns);
	static constexpr auto extent = integral_seq_mul<Tn, Ns...>::value;

	using index_type = Tn;
	using extends	 = std::integer_sequence<Tn, Ns...>;

	template<place_id I>
	__host__ __device__ static constexpr Tn range(std::integral_constant<place_id, I> i) {
		(void) i;
		return select_value<I, extends>::value;// select_indexed_value<I, Tn,
											   // Ns...>::value;
	}

	template<std::size_t... Is, typename... Args>
	__host__ __device__ static constexpr Tn offset_impl(std::index_sequence<Is...> seq, Args&&... args) {
		(void) seq;

		Tn idx = 0;
		(void) std::initializer_list<int> {(idx += std::forward<Args>(args) * excl_suffix_mul<Is, extends> {}, 0)...};
		return idx;
	}

	template<typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
	__host__ __device__ static constexpr Tn offset(Args&&... args) {
		return offset_impl(std::index_sequence_for<Args...> {}, std::forward<Args>(args)...);
	}
};

/// vec without lifetime managing
template<typename T, typename Tn, Tn... Ns>
struct vec_view<T, std::integer_sequence<Tn, Ns...>>
	: identity_vec
	, indexer<Tn, Ns...> {
   private:
	T* m_data;

   public:
	using base_t = indexer<Tn, Ns...>;
	using base_t::dim;
	using base_t::extent;
	using value_type = T;
	using base_t::offset;
	using typename base_t::extends;
	using typename base_t::index_type;

	constexpr vec_view() = delete;
	__host__ __device__ explicit constexpr vec_view(T* ptr)
		: m_data {ptr} {}

	/// random access
	// ()
	template<typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
	__host__ __device__ constexpr T& operator()(Args&&... args) noexcept {
		return m_data[offset(std::forward<Args>(args)...)];
	}
	template<typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
	__host__ __device__ constexpr const T& operator()(Args&&... args) const noexcept {
		return m_data[offset(std::forward<Args>(args)...)];
	}

	// []
	template<typename Index, typename R = vec_view<T, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>, extends>>, Tn D = dim, enable_if_t<(D > 1)> = 0>
	__host__ __device__ constexpr R operator[](Index&& index) noexcept {
		return R {m_data + offset(std::forward<Index>(index))};
	}
	template<typename Index, typename R = vec_view<std::add_const_t<T>, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>, extends>>, Tn D = dim, enable_if_t<(D > 1)> = 0>
	__host__ __device__ constexpr R operator[](Index index) const noexcept {
		return R {m_data + offset(std::forward<Index>(index))};
	}
	template<typename Index, Tn D = dim, enable_if_t<D == 1> = 0>
	__host__ __device__ constexpr T& operator[](Index index) noexcept {
		return m_data[std::forward<Index>(index)];
	}
	template<typename Index, Tn D = dim, enable_if_t<D == 1> = 0>
	__host__ __device__ constexpr const T& operator[](Index index) const noexcept {
		return m_data[std::forward<Index>(index)];
	}

	template<typename Index>
	__host__ __device__ constexpr T& val(Index index) noexcept {
		return m_data[std::forward<Index>(index)];
	}
	template<typename Index>
	__host__ __device__ constexpr const T& val(Index index) const noexcept {
		return m_data[std::forward<Index>(index)];
	}
};

/// vec
template<typename T, typename Tn, Tn... Ns>
struct vec_impl<T, std::integer_sequence<Tn, Ns...>>
	: identity_vec
	, indexer<Tn, Ns...> {
	static_assert(std::is_trivial<T>::value, "Vec element type is not trivial!\n");

	using base_t = indexer<Tn, Ns...>;
	using base_t::dim;
	using base_t::extent;
	using value_type = T;
	using base_t::offset;
	using typename base_t::extends;
	using typename base_t::index_type;

   private:
	std::array<T, extent> m_data;

   public:
	/// construct
	constexpr vec_impl() = default;

	template<typename... Vals, enable_if_all<(sizeof...(Vals)) == extent, std::is_convertible<Vals, value_type>::value...> = 0>
	__host__ __device__ explicit constexpr vec_impl(Vals&&... vals) noexcept
		: m_data {std::forward<Vals>(vals)...} {}

	template<typename Val, decltype(extent) Cnt = extent, enable_if_all<(Cnt > 1), std::is_convertible<Val, value_type>::value> = 0>
	__host__ __device__ explicit constexpr vec_impl(Val&& val) noexcept {
		for(Tn idx = 0; idx < extent; ++idx) {
			m_data[idx] = std::forward<Val>(val);
		}
	}

	__host__ __device__ constexpr void set(T val) noexcept {
		for(Tn idx = 0; idx < extent; ++idx) {
			m_data[idx] = val;
		}
	}

	/// expose internal
	__host__ __device__ constexpr auto data() noexcept -> T* {
		return m_data.data();
	}
	__host__ __device__ constexpr auto data() const noexcept -> const T* {
		return m_data.data();
	}
	__host__ __device__ constexpr std::array<T, extent>& data_arr() noexcept {
		return m_data;
	}
	__host__ __device__ constexpr const std::array<T, extent>& data_arr() const noexcept {
		return m_data;
	}

	/// random access
	// ()
	template<typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
	__host__ __device__ constexpr T& operator()(Args&&... args) noexcept {
		return m_data[offset(std::forward<Args>(args)...)];
	}
	template<typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
	__host__ __device__ constexpr const T& operator()(Args&&... args) const noexcept {
		return m_data[offset(std::forward<Args>(args)...)];
	}

	// []
	template<typename Index, typename R = vec_view<T, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>, extends>>, Tn D = dim, enable_if_t<(D > 1)> = 0>
	__host__ __device__ constexpr R operator[](Index&& index) noexcept {
		return R {m_data.data() + offset(std::forward<Index>(index))};
	}
	template<typename Index, typename R = vec_view<std::add_const_t<T>, gather_t<translate_seq_t<std::make_index_sequence<dim - 1>, 1>, extends>>, Tn D = dim, enable_if_t<(D > 1)> = 0>
	__host__ __device__ constexpr R operator[](Index index) const noexcept {
		return R {m_data.data() + offset(std::forward<Index>(index))};
	}
	template<typename Index, Tn D = dim, enable_if_t<D == 1> = 0>
	__host__ __device__ constexpr T& operator[](Index index) noexcept {
		return m_data[std::forward<Index>(index)];
	}
	template<typename Index, Tn D = dim, enable_if_t<D == 1> = 0>
	__host__ __device__ constexpr const T& operator[](Index index) const noexcept {
		return m_data[std::forward<Index>(index)];
	}

	template<typename Index>
	__host__ __device__ constexpr T& val(Index index) noexcept {
		return m_data[std::forward<Index>(index)];
	}
	template<typename Index>
	__host__ __device__ constexpr const T& val(Index index) const noexcept {
		return m_data[std::forward<Index>(index)];
	}
	///

	template<typename TT>
	__host__ __device__ constexpr auto cast() const noexcept {
		vec_impl<TT, extends> r {};
		for(Tn idx = 0; idx < extent; ++idx) {
			r.val(idx) = m_data[idx];
		}
		return r;
	}

	/// compare
	template<typename Vec, enable_if_all<std::is_base_of<identity_vec, Vec>::value, is_same<typename Vec::extends, extends>::value, std::is_convertible<T, typename Vec::value_type>::value> = 0>
	__host__ __device__ constexpr auto operator==(Vec&& o) noexcept -> bool {
		for(int i = 0; i < extent; ++i) {
			if(m_data[i] != std::forward<Vec>(o).val(i)) {
				return false;
			}
		}
		return true;
	}
	template<typename Vec, enable_if_all<std::is_base_of<identity_vec, Vec>::value, is_same<typename Vec::extends, extends>::value, std::is_convertible<T, typename Vec::value_type>::value> = 0>
	__host__ __device__ constexpr auto operator!=(Vec&& o) noexcept -> bool {
		for(int i = 0; i < extent; ++i) {
			if(m_data[i] == std::forward<Vec>(o).val(i)) {
				return false;
			}
		}
		return true;
	}

	/// linalg
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr R dot(vec_impl<TT, extends> const& o) const noexcept {
		R res {0};
		for(int i = 0; i < extent; ++i) {
			res += m_data[i] * o.val(i);
		}
		return res;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr R dot(vec_impl<TT, extends>&& o) const noexcept {
		R res {0};
		for(int i = 0; i < extent; ++i) {
			res += m_data[i] * o.val(i);
		}
		return res;
	}
	__host__ __device__ constexpr T prod() noexcept {
		T res {1};
		for(int i = 0; i < extent; ++i) {
			res *= m_data[i];
		}
		return res;
	}
	__host__ __device__ constexpr T l2NormSqr() noexcept {
		T res {0};
		for(int i = 0; i < extent; ++i) {
			res += m_data[i] * m_data[i];
		}
		return res;
	}
	__host__ __device__ constexpr T infNormSqr() noexcept {
		T res {0};
		for(int i = 0; i < extent; ++i) {
			T sqr = m_data[i] * m_data[i];
			if(sqr > res) {
				res = sqr;
			}
		}
		return res;
	}

	/// borrowed from
	/// https://github.com/cemyuksel/cyCodeBase/blob/master/cyIVector.h
	/// east const
	//!@name Unary operators
	__host__ __device__ constexpr vec_impl operator-() const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = -m_data[i];
		}
		return r;
	}

	//!@name Binary operators
	// scalar
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator+(vec_impl const& e, TT const v) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) + v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator+(TT const v, vec_impl const& e) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) + v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator-(vec_impl const& e, TT const v) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) - v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator-(TT const v, vec_impl const& e) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = v - e.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator*(vec_impl const& e, TT const v) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) * v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator*(TT const v, vec_impl const& e) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) * v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator/(vec_impl const& e, TT const v) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = e.val(i) / v;
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ friend constexpr auto operator/(TT const v, vec_impl const& e) noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = v / e.val(i);
		}
		return r;
	}
	// vector
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator+(vec_impl<TT, extends> const& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] + o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator+(vec_impl<TT, extends>&& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] + o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator-(vec_impl<TT, extends> const& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] - o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator-(vec_impl<TT, extends>&& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] - o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator*(vec_impl<TT, extends> const& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] * o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator*(vec_impl<TT, extends>&& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] * o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator/(vec_impl<TT, extends> const& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] / o.val(i);
		}
		return r;
	}
	template<typename TT, typename R = std::common_type_t<T, TT>>
	__host__ __device__ constexpr auto operator/(vec_impl<TT, extends>&& o) const noexcept {
		vec_impl<R, extends> r {};
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] / o.val(i);
		}
		return r;
	}
	//!@name Assignment operators
	// scalar
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator+=(TT&& v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] += std::forward<TT>(v);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator-=(TT&& v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] -= std::forward<TT>(v);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator*=(TT&& v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] *= std::forward<TT>(v);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator/=(TT&& v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] /= std::forward<TT>(v);
		}
		return *this;
	}

	// vector
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator+=(vec_impl<TT, extends> const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] += o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator+=(vec_impl<TT, extends>&& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] += o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator-=(vec_impl<TT, extends> const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] -= o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator-=(vec_impl<TT, extends>&& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] -= o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator*=(vec_impl<TT, extends> const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] *= o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator*=(vec_impl<TT, extends>&& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] *= o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator/=(vec_impl<TT, extends> const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] /= o.val(i);
		}
		return *this;
	}
	template<typename TT, enable_if_t<std::is_convertible<T, TT>::value> = 0>
	__host__ __device__ constexpr vec_impl& operator/=(vec_impl<TT, extends>&& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] /= o.val(i);
		}
		return *this;
	}

#if ENABLE_BITWISE_OPERATORS
	//!@name Bitwise operators
	__host__ __device__ vec_impl operator<<(vec_impl const& o) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] << o.val(i);
		}
		return r;
	}
	__host__ __device__ vec_impl operator>>(vec_impl const& o) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] >> o.val(i);
		}
		return r;
	}
	__host__ __device__ vec_impl operator&(vec_impl const& o) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] & o.val(i);
		}
		return r;
	}
	__host__ __device__ vec_impl operator|(vec_impl const& o) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] | o.val(i);
		}
		return r;
	}
	__host__ __device__ vec_impl operator^(vec_impl const& o) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] ^ o.val(i);
		}
		return r;
	}
	__host__ __device__ vec_impl operator<<(T const v) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] << v;
		}
		return r;
	}
	__host__ __device__ vec_impl operator>>(T const v) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] >> v;
		}
		return r;
	}
	__host__ __device__ vec_impl operator&(T const v) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] & v;
		}
		return r;
	}
	__host__ __device__ vec_impl operator|(T const v) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] | v;
		}
		return r;
	}
	__host__ __device__ vec_impl operator^(T const v) const noexcept {
		vec_impl r;
		for(int i = 0; i < extent; ++i) {
			r.val(i) = m_data[i] ^ v;
		}
		return r;
	}

	//!@name Bitwise Assignment operators
	__host__ __device__ vec_impl& operator<<=(vec_impl const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] <<= o.val(i);
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator>>=(vec_impl const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] >>= o.val(i);
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator&=(vec_impl const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] &= o.val(i);
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator|=(vec_impl const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] |= o.val(i);
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator^=(vec_impl const& o) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] ^= o.val(i);
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator<<=(T const v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] <<= v;
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator>>=(T const v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] >>= v;
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator&=(T const v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] &= v;
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator|=(T const v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] |= v;
		}
		return *this;
	}
	__host__ __device__ vec_impl& operator^=(T const v) noexcept {
		for(int i = 0; i < extent; ++i) {
			m_data[i] ^= v;
		}
		return *this;
	}
#endif
};

/// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
/// vector(vec+{0}) homogeneous coordinates

}// namespace mn

#endif