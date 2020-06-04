#ifndef __STRUCTURAL_H_
#define __STRUCTURAL_H_

#include "Function.h"
#include "Property.h"
#include "StructuralAuxiliary.h"
#include "StructuralDeclaration.h"
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/MathMeta.h>
#include <MnBase/Meta/Polymorphism.h>
#include <MnBase/Meta/TypeMeta.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace mn {

enum class structural_component_index : std::size_t {
  handle_ptr = 0,
  parent_scope_handle,
  active_mask ///< for on-demand allocation
};

using ci = structural_component_index;

/// cuda forbids ci as int_seq type
using orphan_signature =
    std::integer_sequence<std::size_t,
                          static_cast<std::size_t>(
                              structural_component_index::handle_ptr)>;
using standard_signature =
    std::integer_sequence<std::size_t,
                          static_cast<std::size_t>(
                              structural_component_index::parent_scope_handle)>;
using sparse_orphan_signature =
    std::integer_sequence<std::size_t,
                          static_cast<std::size_t>(
                              structural_component_index::active_mask)>;

/// instance related declarations
template <typename ParentInstance, attrib_index AttribNo, typename Components>
struct structural_instance;
template <typename Structural, typename Signature = orphan_signature>
using Instance =
    structural_instance<root_instance<Structural>, (attrib_index)0, Signature>;

template <typename parent_instance, attrib_index AttribNo>
struct structural_instance_traits
    : parent_instance::attribs::template type<(std::size_t)AttribNo> {
  using self = typename parent_instance::attribs::type<(std::size_t)AttribNo>;
  using parent_indexer = typename parent_instance::domain::index;
  using self_indexer = typename self::domain::index;
};
template <typename parent_instance, attrib_index, ci>
struct structural_instance_component;
template <typename parent_instance, attrib_index AttribNo>
struct structural_instance_component<parent_instance, AttribNo,
                                     ci::handle_ptr> {};
template <typename parent_instance, attrib_index AttribNo>
struct structural_instance_component<parent_instance, AttribNo,
                                     ci::parent_scope_handle> {
  typename structural_instance_traits<parent_instance,
                                      AttribNo>::parent_instance _parent;
  typename structural_instance_traits<parent_instance, AttribNo>::parent_indexer
      _index;
};
template <typename parent_instance, attrib_index AttribNo>
struct structural_instance_component<parent_instance, AttribNo,
                                     ci::active_mask> {
  static_assert(sizeof(char) == 1, "size (byte) of char is not 1.");
  char *_activeMask;
};

template <typename ParentInstance, attrib_index AttribNo, std::size_t... Cs>
struct structural_instance<ParentInstance, AttribNo,
                           std::integer_sequence<std::size_t, Cs...>>
    : structural_instance_traits<ParentInstance, AttribNo>,
      structural_instance_component<ParentInstance, AttribNo,
                                    static_cast<ci>(Cs)>... {
  using traits = structural_instance_traits<ParentInstance, AttribNo>;
  using component_seq = std::integer_sequence<std::size_t, Cs...>;
  using self_instance =
      structural_instance<ParentInstance, AttribNo, component_seq>;
  template <attrib_index ChAttribNo>
  using accessor = typename traits::template accessor<ChAttribNo>;

  // hierarchy traverse
  template <attrib_index ChAttribNo, typename... Indices>
  constexpr auto chfull(std::integral_constant<attrib_index, ChAttribNo>,
                        Indices &&... indices) const {
    structural_instance<self_instance, ChAttribNo, standard_signature> ret{};
    ret._parent = *this;
    ret._index = traits::self_indexer(std::forward<Indices>(indices)...);
    ret._handle = MemResource{(void *)(this->_handle.ptrval +
                                       accessor<ChAttribNo>::coord_offset(
                                           std::forward<Indices>(indices)...))};
    return ret;
  }
  template <attrib_index ChAttribNo, typename... Indices>
  constexpr auto ch(std::integral_constant<attrib_index, ChAttribNo>,
                    Indices &&... indices) const {
    structural_instance<self_instance, ChAttribNo, orphan_signature> ret{};
    ret._handle = MemResource{(void *)(this->_handle.ptrval +
                                       accessor<ChAttribNo>::coord_offset(
                                           std::forward<Indices>(indices)...))};
    return ret;
  }
  template <attrib_index ChAttribNo, typename... Indices>
  constexpr auto chptr(std::integral_constant<attrib_index, ChAttribNo>,
                       Indices &&... indices) const {
    structural_instance<self_instance, ChAttribNo, orphan_signature> ret{};
    ret._handle.ptrval = *(uintptr_t *)(this->_handle.ptrval +
                                        accessor<ChAttribNo>::coord_offset(
                                            std::forward<Indices>(indices)...));
    return ret;
  }
};

/// initialization
template <typename Structural, typename Componenets, typename Allocator>
constexpr auto spawn(Allocator allocator) {
  auto ret = Instance<Structural, Componenets>{};
  ret.allocate_handle(allocator);
  return ret;
}
template <typename Structural, typename Componenets, typename Allocator>
constexpr void recycle(Instance<Structural, Componenets> &instance,
                       Allocator allocator) {
  instance.deallocate(allocator);
}

/// common notation
using empty_ = structural_entity<void>;
using i64_ = structural_entity<int64_t>;
using u64_ = structural_entity<uint64_t>;
using f64_ = structural_entity<double>;
using i32_ = structural_entity<int32_t>;
using u32_ = structural_entity<uint32_t>;
using f32_ = structural_entity<float>;

} // namespace mn

#endif