#ifndef STRUCTURAL_H
#define STRUCTURAL_H

#include <memory>
#include <type_traits>
#include <utility>

#include "Function.h"
#include "MnBase/Meta/ControlFlow.h"
#include "MnBase/Meta/MathMeta.h"
#include "MnBase/Meta/Polymorphism.h"
#include "MnBase/Meta/TypeMeta.h"
#include "Property.h"
#include "StructuralAuxiliary.h"
#include "StructuralDeclaration.h"

namespace mn {

enum class StructuralComponentIndex : std::size_t {
	HANDLE_PTR = 0,
	PARENT_SCOPE_HANDLE,
	ACTIVE_MASK///< for on-demand allocation
};

using ci = StructuralComponentIndex;

/// cuda forbids ci as int_seq type
using orphan_signature		  = std::integer_sequence<std::size_t, static_cast<std::size_t>(StructuralComponentIndex::HANDLE_PTR)>;
using standard_signature	  = std::integer_sequence<std::size_t, static_cast<std::size_t>(StructuralComponentIndex::PARENT_SCOPE_HANDLE)>;
using sparse_orphan_signature = std::integer_sequence<std::size_t, static_cast<std::size_t>(StructuralComponentIndex::ACTIVE_MASK)>;

/// instance related declarations
template<typename ParentInstance, attrib_index AttribNo, typename Components>
struct StructuralInstance;
template<typename Structural, typename Signature = orphan_signature>
using Instance = StructuralInstance<RootInstance<Structural>, (attrib_index) 0, Signature>;

template<typename ParentInstance, attrib_index AttribNo>
struct StructuralInstanceTraits : ParentInstance::attribs::template type<(std::size_t) AttribNo> {
	using self			 = typename ParentInstance::attribs::type<(std::size_t) AttribNo>;
	using parent_indexer = typename ParentInstance::domain::index;
	using self_indexer	 = typename self::domain::index;
};

template<typename ParentInstance, attrib_index, ci>
struct StructuralInstanceComponent;
template<typename ParentInstance, attrib_index AttribNo>
struct StructuralInstanceComponent<ParentInstance, AttribNo, ci::HANDLE_PTR> {};
template<typename ParentInstance, attrib_index AttribNo>
struct StructuralInstanceComponent<ParentInstance, AttribNo, ci::PARENT_SCOPE_HANDLE> {
	typename StructuralInstanceTraits<ParentInstance, AttribNo>::ParentInstance _parent;
	typename StructuralInstanceTraits<ParentInstance, AttribNo>::parent_indexer _index;
};
template<typename ParentInstance, attrib_index AttribNo>
struct StructuralInstanceComponent<ParentInstance, AttribNo, ci::ACTIVE_MASK> {
	static_assert(sizeof(char) == 1, "size (byte) of char is not 1.");
	char* active_mask;
};

template<typename ParentInstance, attrib_index AttribNo, std::size_t... Cs>
struct StructuralInstance<ParentInstance, AttribNo, std::integer_sequence<std::size_t, Cs...>>
	: StructuralInstanceTraits<ParentInstance, AttribNo>
	, StructuralInstanceComponent<ParentInstance, AttribNo, static_cast<ci>(Cs)>... {
	using traits		= StructuralInstanceTraits<ParentInstance, AttribNo>;
	using component_seq = std::integer_sequence<std::size_t, Cs...>;
	using self_instance = StructuralInstance<ParentInstance, AttribNo, component_seq>;
	template<attrib_index ChAttribNo>
	using accessor = typename traits::template Accessor<ChAttribNo>;

	// hierarchy traverse
	template<attrib_index ChAttribNo, typename... Indices>
	constexpr auto chfull(std::integral_constant<attrib_index, ChAttribNo>, Indices&&... indices) const {
		StructuralInstance<self_instance, ChAttribNo, standard_signature> ret {};
		ret._parent = *this;
		ret._index	= traits::self_indexer(std::forward<Indices>(indices)...);
		ret.handle	= MemResource {reinterpret_cast<void*>(this->handle.ptrval + accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...))};
		return ret;
	}
	template<attrib_index ChAttribNo, typename... Indices>
	constexpr auto ch(std::integral_constant<attrib_index, ChAttribNo>, Indices&&... indices) const {
		StructuralInstance<self_instance, ChAttribNo, orphan_signature> ret {};
		ret.handle = MemResource {reinterpret_cast<void*>(this->handle.ptrval + accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...))};
		return ret;
	}
	template<attrib_index ChAttribNo, typename... Indices>
	constexpr auto chptr(std::integral_constant<attrib_index, ChAttribNo>, Indices&&... indices) const {
		StructuralInstance<self_instance, ChAttribNo, orphan_signature> ret {};
		ret.handle.ptrval = *reinterpret_cast<uintptr_t*>(this->handle.ptrval + accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...));
		return ret;
	}
};

/// initialization
template<typename Structural, typename Components, typename Allocator>
constexpr auto spawn(Allocator allocator) {
	auto ret = Instance<Structural, Components> {};
	ret.allocate_handle(allocator);
	return ret;
}
template<typename Structural, typename Components, typename Allocator>
constexpr auto spawn(Allocator allocator, std::size_t size) {
	auto ret = Instance<Structural, Components> {};
	ret.allocate_handle(allocator, size);
	return ret;
}
template<typename Structural, typename Components, typename Allocator>
constexpr void recycle(Instance<Structural, Components>& instance, Allocator allocator) {
	instance.deallocate(allocator);
}

/// common notation
using empty_ = StructuralEntity<void>;
using i64_	 = StructuralEntity<int64_t>;
using u64_	 = StructuralEntity<uint64_t>;
using f64_	 = StructuralEntity<double>;
using i32_	 = StructuralEntity<int32_t>;
using u32_	 = StructuralEntity<uint32_t>;
using f32_	 = StructuralEntity<float>;

}// namespace mn

#endif