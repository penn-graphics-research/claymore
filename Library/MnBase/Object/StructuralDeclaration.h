#ifndef STRUCTURAL_DECLARATION_H
#define STRUCTURAL_DECLARATION_H
#include <type_traits>

#include "MnBase/Math/Vec.h"
#include "MnBase/Memory/MemObj.h"
#include "MnBase/Meta/MathMeta.h"

namespace mn {

enum class StructuralType : std::size_t {
	/// no child
	SENTINEL = 0,
	ENTITY	 = 1,
	/// with child
	HASH	= 2,
	DENSE	= 3,
	DYNAMIC = 4,
	BINARY	= 5,
	TOTAL
};
using attrib_index = place_id;

struct IdentityStructural {};

template<StructuralType, typename Decoration, typename Domain, attrib_layout Layout, typename... Structurals>
struct Structural;

template<StructuralType NodeType, typename Decoration, typename Domain, attrib_layout Layout, typename... Structurals>
struct StructuralTraits {
	static constexpr StructuralType node_type	 = NodeType;
	static constexpr attrib_layout elementLayout = Layout;

	using decoration = Decoration;
	using domain	 = Domain;
	using attribs	 = type_seq<Structurals...>;
	using self		 = Structural<NodeType, decoration, domain, elementLayout, Structurals...>;
	using vt		 = void*;///< should be hidden by Structural ENTITY
	template<attrib_index I>
	using value_type = select_indexed_type<I, typename Structurals::vt...>;

	static constexpr auto attrib_count		  = attribs::indices::size();
	static constexpr std::size_t element_size = select_indexed_value<
		(std::size_t) decoration::padding_policy,
		std::size_t,
		// compact
		integral_seq_sum<std::size_t, Structurals::size...>::value,
		// sum aligned
		next_2pow<std::size_t, integral_seq_sum<std::size_t, Structurals::size...>::value>::value,
		// max aligned
		(integral_seq_max<std::size_t, (next_2pow<std::size_t, Structurals::size>::value)...>::value) * sizeof...(Structurals)>::value;
	static constexpr std::size_t element_storage_size = select_indexed_value<
		(std::size_t) decoration::alloc_policy,
		std::size_t,
		/// full-allocation
		element_size,
		/// on-demand
		sizeof(void*) * sizeof...(Structurals)>::value;
	/// for allocation
	static constexpr std::size_t size = domain::extent * element_storage_size;
	// SOA -> multipool, AOS -> pool

	template<attrib_index AttribNo>
	struct Accessor {
	   private:
		static constexpr uintptr_t element_stride_in_bytes_func() {
			switch(decoration::alloc_policy) {
				case StructuralAllocationPolicy::FULL_ALLOCATION:
					return (elementLayout == attrib_layout::AOS ? element_storage_size : attribs::template type<(std::size_t) AttribNo>::size);
				case StructuralAllocationPolicy::ON_DEMAND:
					return sizeof(void*) * (elementLayout == attrib_layout::AOS ? sizeof...(Structurals) : 1);
				default:
					return 0;
			}
		}
		static constexpr uintptr_t attrib_base_offset_func() {
			switch(decoration::alloc_policy) {
				case StructuralAllocationPolicy::FULL_ALLOCATION:
					return (elementLayout == attrib_layout::AOS ? 1 : domain::extent) * excl_prefix_sum<(std::size_t) AttribNo, std::integer_sequence<uintptr_t, Structurals::size...>>::value;
				case StructuralAllocationPolicy::ON_DEMAND:
					return (elementLayout == attrib_layout::AOS ? 1 : domain::extent) * (std::size_t) AttribNo * sizeof(void*);
				default:
					return 0;
			}
		}

	   public:
		static constexpr uintptr_t element_stride_in_bytes = element_stride_in_bytes_func();

		static constexpr uintptr_t attrib_base_offset = attrib_base_offset_func();
		template<typename... Indices>
		static constexpr uintptr_t coord_offset(Indices&&... is) noexcept {
			return attrib_base_offset + domain::offset(std::forward<Indices>(is)...) * element_stride_in_bytes;
		}
		template<typename Index>
		static constexpr uintptr_t linear_offset(Index&& i) noexcept {
			return attrib_base_offset + std::forward<Index>(i) * element_stride_in_bytes;
		}
	};

	MemResource handle;

	// memory manage
	template<typename Allocator>
	void allocate_handle(Allocator allocator) {
		if(self::size != 0) {
			handle.ptr = allocator.allocate(self::size);
		} else {
			handle.ptr = nullptr;
		}
	}
	template<typename Allocator>
	void allocate_handle(Allocator allocator, std::size_t s) {
		if(s != 0) {
			handle.ptr = allocator.allocate(s);
		} else {
			handle.ptr = nullptr;
		}
	}

	template<typename Allocator>
	void deallocate(Allocator allocator) {
		allocator.deallocate(handle.ptr, self::size);
		handle.ptr = nullptr;
	}
	template<typename Allocator>
	void deallocate(Allocator allocator, std::size_t s) {
		allocator.deallocate(handle.ptr, s);
		handle.ptr = nullptr;
	}

	// value access
	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>, typename... Indices>
	constexpr auto& val(std::integral_constant<attrib_index, ChAttribNo>, Indices&&... indices) {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...));
	}
	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>>
	constexpr auto& val(std::integral_constant<attrib_index, ChAttribNo>) {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::attrib_base_offset);
	}
	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>, typename... Indices>
	constexpr const auto& val(std::integral_constant<attrib_index, ChAttribNo>, Indices&&... indices) const {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...));
	}
	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>>
	constexpr const auto& val(std::integral_constant<attrib_index, ChAttribNo>) const {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::attrib_base_offset);
	}

	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>, typename Index>
	constexpr auto& val_1d(std::integral_constant<attrib_index, ChAttribNo>, Index&& index) {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::linear_offset(std::forward<Index>(index)));
	}
	template<attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>, typename Index>
	constexpr const auto& val_1d(std::integral_constant<attrib_index, ChAttribNo>, Index&& index) const {
		return *reinterpret_cast<Type*>(handle.ptrval + Accessor<ChAttribNo>::linear_offset(std::forward<Index>(index)));
	}
};

/// root
template<typename Structural>
struct RootInstance
	: IdentityStructural
	, StructuralTraits<StructuralType::DENSE, Decorator<>, CompactDomain<int, 1>, attrib_layout::AOS, Structural> {
	/// purely for its domain and attrib info query, i.e. Structural info
	using vt = void*;
};
/// ENTITY
template<typename T>
struct StructuralEntity : IdentityStructural {
	static constexpr std::size_t size = sizeof(T);

	using vt	 = T;
	using domain = AlignedDomain<int, 1>;
};
template<>
struct StructuralEntity<void> : IdentityStructural {
	static constexpr std::size_t size = 0;

	using vt	 = void;
	using domain = AlignedDomain<int, 0>;// int[0] is allowed in c++
};
/// HASH
template<typename Decoration, typename Domain, attrib_layout Layout, typename... Structurals>
struct Structural<StructuralType::HASH, Decoration, Domain, Layout, Structurals...>
	: IdentityStructural
	, StructuralTraits<StructuralType::HASH, Decoration, Domain, Layout, Structurals...> {
	using base_t					 = StructuralTraits<StructuralType::HASH, Decoration, Domain, Layout, Structurals...>;
	using CheckPrerequisite			 = void_t<is_base_of<IdentityStructuralIndex, Domain>, is_base_of<IdentityDecorator, Decoration>, is_base_of<IdentityStructural, Structurals...>, satisfy<std::is_trivially_default_constructible, Structurals...>>;
	using key_t						 = vec<typename Domain::index_type, Domain::dim>;
	using value_t					 = typename Domain::index_type;
	static constexpr auto sentinel_v = value_t(-1);

	// data members
	value_t capacity;
	value_t* count;
	key_t* active_keys;	 //
	value_t* index_table;//

	// func members
	template<typename Allocator>
	void allocate_table(Allocator allocator, value_t capacity) {
		this->capacity	= capacity;
		count			= static_cast<value_t*>(allocator.allocate(sizeof(value_t)));
		active_keys = static_cast<key_t*>(allocator.allocate(sizeof(key_t) * capacity));
		/// lookup table
		index_table = static_cast<value_t*>(allocator.allocate(sizeof(value_t) * Domain::extent));
	}

	template<typename Allocator>
	void resize_table(Allocator allocator, std::size_t capacity) {
		allocator.deallocate(active_keys, this->capacity);
		active_keys = static_cast<key_t*>(allocator.allocate(sizeof(key_t) * capacity));
		this->capacity	= capacity;
	}

	template<typename Allocator>
	void deallocate(Allocator allocator) {
		allocator.deallocate(count, sizeof(value_t));
		allocator.deallocate(active_keys, sizeof(key_t) * capacity);
		allocator.deallocate(index_table, sizeof(value_t) * Domain::extent);
		base_t::deallocate(allocator);
		capacity	= 0;
		count			= nullptr;
		active_keys = nullptr;
		index_table = nullptr;
	}

	//TODO: Add (optional) range checks here (and maybe elsewhere too)
	template<typename... Indices>
	constexpr auto& val(Indices&&... indices) {
		return *static_cast<value_t*>(static_cast<void*>(index_table + Domain::offset(std::forward<Indices>(indices)...)));
	}

	template<std::size_t... Is>
	constexpr auto& index_impl(key_t& coord, std::index_sequence<Is...>) {
		return *static_cast<value_t*>(static_cast<void*>(index_table + Domain::offset((coord[Is])...)));
	}

	constexpr auto& index(key_t coord) {
		return index_impl(coord, std::make_index_sequence<Domain::dim> {});
	}

	template<std::size_t... Is>
	constexpr const auto& index_impl(key_t& coord, std::index_sequence<Is...>) const {
		return *static_cast<value_t*>(static_cast<void*>(index_table + Domain::offset((coord[Is])...)));
	}

	constexpr const auto& index(key_t coord) const {
		return index_impl(coord, std::make_index_sequence<Domain::dim> {});
	}
};
/// DENSE
template<typename Decoration, typename Domain, attrib_layout Layout, typename... Structurals>
struct Structural<StructuralType::DENSE, Decoration, Domain, Layout, Structurals...>
	: IdentityStructural
	, StructuralTraits<StructuralType::DENSE, Decoration, Domain, Layout, Structurals...> {
	using base_t			= StructuralTraits<StructuralType::DENSE, Decoration, Domain, Layout, Structurals...>;
	using CheckPrerequisite = void_t<is_base_of<IdentityStructuralIndex, Domain>, is_base_of<IdentityDecorator, Decoration>, is_base_of<IdentityStructural, Structurals...>, satisfy<std::is_trivially_default_constructible, Structurals...>>;
};

/// DYNAMIC
template<typename Decoration, typename Domain, attrib_layout Layout, typename... Structurals>
struct Structural<StructuralType::DYNAMIC, Decoration, Domain, Layout, Structurals...>
	: IdentityStructural
	, StructuralTraits<StructuralType::DYNAMIC, Decoration, Domain, Layout, Structurals...> {
	using base_t			= StructuralTraits<StructuralType::DYNAMIC, Decoration, Domain, Layout, Structurals...>;
	using CheckPrerequisite = void_t<is_base_of<IdentityStructuralIndex, Domain>, is_base_of<IdentityDecorator, Decoration>, is_base_of<IdentityStructural, Structurals...>, satisfy<std::is_trivially_default_constructible, Structurals...>, enable_if_t<Domain::dim == 1>>;
	using value_t			= unsigned long long int;

	std::size_t capacity;

	template<typename Allocator>
	void allocate_handle(Allocator allocator, std::size_t capacity = Domain::extent) {
		if(capacity != 0) {
			this->handle.ptr = allocator.allocate(capacity * base_t::element_storage_size);
		} else {
			this->handle.ptr = nullptr;
		}
		this->capacity = capacity;
	}

	template<typename Allocator>
	void resize(Allocator allocator, std::size_t capacity) {
		allocator.deallocate(this->handle.ptr, this->capacity);
		this->capacity		 = capacity;///< each time multiply by 2
		this->handle.ptr = allocator.allocate(capacity * base_t::element_storage_size);
	}

	template<typename Allocator>
	void deallocate(Allocator allocator) {
		allocator.deallocate(this->handle.ptr, this->capacity * base_t::element_storage_size);
		this->capacity		 = 0;
		this->handle.ptr = nullptr;
	}
};

}// namespace mn

#endif