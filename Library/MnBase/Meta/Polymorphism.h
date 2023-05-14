#ifndef POLYMORPHISM_H
#define POLYMORPHISM_H
//#include <mapbox/variant.hpp>
#include <mpark/variant.hpp>

#define ADVANCED_OVERLOAD_SET 0

namespace mn {

#if ADVANCED_OVERLOAD_SET
/// only in c++17
template<typename... TFs>
struct overload_set : TFs... {
	using TFs::operator()...;
};
template<typename... TFs>
overload(TFs...)->overload_set<TFs...>;
#else

template<typename TF, typename... TFs>
struct overload_set
	: TF
	, overload_set<TFs...> {
	using TF::operator();
	using overload_set<TFs...>::operator();
	template<typename TFFwd, typename... TFsFwd>
	overload_set(TFFwd&& f, TFsFwd&&... fs)
		: TF {std::forward<TFFwd>(f)}
		, overload_set<TFs...> {std::forward<TFsFwd>(fs)...} {}
};
template<typename TF>
struct overload_set<TF> : TF {
	using TF::operator();
	template<typename TFFwd>
	overload_set(TFFwd&& f)
		: TF {std::forward<TFFwd>(f)} {}
};

template<typename... TFs>
constexpr auto overload(TFs&&... fs) {
	return overload_set<std::remove_reference_t<TFs>...>(std::forward<TFs>(fs)...);
}

template<typename... Ts>
//using variant = mapbox::util::variant<Ts...>;
using variant = mpark::variant<Ts...>;

template<typename T, typename TVariant>
auto& get(TVariant&& variant) {
	//return mapbox::util::get<T>(std::forward<TVariant>(variant));
	return mpark::get<T>(std::forward<TVariant>(variant));
}

template<typename... TVariants>
constexpr auto match(TVariants&&... vs) {
	return [&vs...](auto&&... fs) -> decltype(auto) {
		auto visitor = overload(std::forward<decltype(fs)>(fs)...);
		//return mapbox::util::apply_visitor(visitor, std::forward<TVariants>(vs)...);
		return mpark::visit(visitor, std::forward<TVariants>(vs)...);
	};
}
#endif

}// namespace mn

#endif