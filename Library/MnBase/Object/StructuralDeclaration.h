#ifndef __STRUCTURAL_DECLARATION_H_
#define __STRUCTURAL_DECLARATION_H_
#include <MnBase/Math/Vec.h>
#include <MnBase/Memory/MemObj.h>
#include <MnBase/Meta/MathMeta.h>
#include <type_traits>

namespace mn {

enum class structural_type : std::size_t {
  /// no child
  sentinel = 0,
  entity = 1,
  /// with child
  hash = 2,
  dense = 3,
  dynamic = 4,
  binary = 5,
  total
};
using attrib_index = place_id;

struct identity_structural {};
template <structural_type, typename Decoration, typename Domain,
          attrib_layout Layout, typename... Structurals>
struct structural;

template <structural_type NodeType, typename Decoration, typename Domain,
          attrib_layout Layout, typename... Structurals>
struct structural_traits {
  static constexpr structural_type node_type = NodeType;
  using decoration = Decoration;
  using domain = Domain;
  static constexpr attrib_layout elementLayout = Layout;
  using attribs = type_seq<Structurals...>;
  using self =
      structural<NodeType, decoration, domain, elementLayout, Structurals...>;
  using vt = void *; ///< should be hidden by structural entity
  template <attrib_index I>
  using value_type = select_indexed_type<I, typename Structurals::vt...>;
  static constexpr auto attrib_count = attribs::indices::size();
  static constexpr std::size_t element_size = select_indexed_value<
      (std::size_t)decoration::padding_policy, std::size_t,
      // compact
      integral_seq_sum<std::size_t, Structurals::size...>::value,
      // sum aligned
      next_2pow<
          std::size_t,
          integral_seq_sum<std::size_t, Structurals::size...>::value>::value,
      // max aligned
      (integral_seq_max<
          std::size_t,
          (next_2pow<std::size_t, Structurals::size>::value)...>::value) *
          sizeof...(Structurals)>::value;
  static constexpr std::size_t element_storage_size =
      select_indexed_value<(std::size_t)decoration::alloc_policy, std::size_t,
                           /// full-allocation
                           element_size,
                           /// on-demand
                           sizeof(void *) * sizeof...(Structurals)>::value;
  /// for allocation
  static constexpr std::size_t size = domain::extent * element_storage_size;
  // soa -> multipool, aos -> pool

  template <attrib_index AttribNo> struct accessor {
  private:
    static constexpr uintptr_t elementStrideInBytes() {
      switch (decoration::alloc_policy) {
      case structural_allocation_policy::full_allocation:
        return (elementLayout == attrib_layout::aos
                    ? element_storage_size
                    : attribs::template type<(std::size_t)AttribNo>::size);
      case structural_allocation_policy::on_demand:
        return sizeof(void *) * (elementLayout == attrib_layout::aos
                                     ? sizeof...(Structurals)
                                     : 1);
      default:
        return 0;
      }
    }
    static constexpr uintptr_t attribBaseOffset() {
      switch (decoration::alloc_policy) {
      case structural_allocation_policy::full_allocation:
        return (elementLayout == attrib_layout::aos ? 1 : domain::extent) *
               excl_prefix_sum<(std::size_t)AttribNo,
                               std::integer_sequence<
                                   uintptr_t, Structurals::size...>>::value;
      case structural_allocation_policy::on_demand:
        return (elementLayout == attrib_layout::aos ? 1 : domain::extent) *
               (std::size_t)AttribNo * sizeof(void *);
      default:
        return 0;
      }
    }

  public:
    static constexpr uintptr_t element_stride_in_bytes = elementStrideInBytes();

    static constexpr uintptr_t attrib_base_offset = attribBaseOffset();
    template <typename... Indices>
    static constexpr uintptr_t coord_offset(Indices &&... is) noexcept {
      return attrib_base_offset + domain::offset(std::forward<Indices>(is)...) *
                                      element_stride_in_bytes;
    }
    template <typename Index>
    static constexpr uintptr_t linear_offset(Index &&i) noexcept {
      return attrib_base_offset +
             std::forward<Index>(i) * element_stride_in_bytes;
    }
  };

  // memory manage
  template <typename Allocator> void allocate_handle(Allocator allocator) {
    if (self::size != 0)
      _handle.ptr = allocator.allocate(self::size);
    else
      _handle.ptr = nullptr;
  }
  template <typename Allocator>
  void allocate_handle(Allocator allocator, std::size_t s) {
    if (s != 0)
      _handle.ptr = allocator.allocate(s);
    else
      _handle.ptr = nullptr;
  }
  template <typename Allocator> void deallocate(Allocator allocator) {
    allocator.deallocate(_handle.ptr, self::size);
    _handle.ptr = nullptr;
  }
  template <typename Allocator>
  void deallocate(Allocator allocator, std::size_t s) {
    allocator.deallocate(_handle.ptr, s);
    _handle.ptr = nullptr;
  }
  // value access
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>,
            typename... Indices>
  constexpr auto &val(std::integral_constant<attrib_index, ChAttribNo>,
                      Indices &&... indices) {
    return *reinterpret_cast<Type *>(
        _handle.ptrval +
        accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...));
  }
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>>
  constexpr auto &val(std::integral_constant<attrib_index, ChAttribNo>) {
    return *reinterpret_cast<Type *>(_handle.ptrval +
                                     accessor<ChAttribNo>::attrib_base_offset);
  }
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>,
            typename Index>
  constexpr auto &val_1d(std::integral_constant<attrib_index, ChAttribNo>,
                         Index &&index) {
    return *reinterpret_cast<Type *>(
        _handle.ptrval +
        accessor<ChAttribNo>::linear_offset(std::forward<Index>(index)));
  }
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>,
            typename... Indices>
  constexpr const auto &val(std::integral_constant<attrib_index, ChAttribNo>,
                            Indices &&... indices) const {
    return *reinterpret_cast<Type *>(
        _handle.ptrval +
        accessor<ChAttribNo>::coord_offset(std::forward<Indices>(indices)...));
  }
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>>
  constexpr const auto &
  val(std::integral_constant<attrib_index, ChAttribNo>) const {
    return *reinterpret_cast<Type *>(_handle.ptrval +
                                     accessor<ChAttribNo>::attrib_base_offset);
  }
  template <attrib_index ChAttribNo, typename Type = value_type<ChAttribNo>,
            typename Index>
  constexpr const auto &val_1d(std::integral_constant<attrib_index, ChAttribNo>,
                               Index &&index) const {
    return *reinterpret_cast<Type *>(
        _handle.ptrval +
        accessor<ChAttribNo>::linear_offset(std::forward<Index>(index)));
  }

  MemResource _handle;
};

/// root
template <typename Structural>
struct root_instance : identity_structural,
                       structural_traits<structural_type::dense, decorator<>,
                                         compact_domain<int, 1>,
                                         attrib_layout::aos, Structural> {
  /// purely for its domain and attrib info query, i.e. structural info
  using vt = void *;
};
/// entity
template <typename T> struct structural_entity : identity_structural {
  using vt = T;
  using domain = aligned_domain<int, 1>;
  static constexpr std::size_t size = sizeof(T);
};
template <> struct structural_entity<void> : identity_structural {
  using vt = void;
  using domain = aligned_domain<int, 0>; // int[0] is allowed in c++
  static constexpr std::size_t size = 0;
};
/// hash
template <typename Decoration, typename Domain, attrib_layout Layout,
          typename... Structurals>
struct structural<structural_type::hash, Decoration, Domain, Layout,
                  Structurals...>
    : identity_structural,
      structural_traits<structural_type::hash, Decoration, Domain, Layout,
                        Structurals...> {
  using base_t = structural_traits<structural_type::hash, Decoration, Domain,
                                   Layout, Structurals...>;
  using CheckPrerequisite =
      void_t<is_base_of<identity_structural_index, Domain>,
             is_base_of<identity_decorator, Decoration>,
             is_base_of<identity_structural, Structurals...>,
             satisfy<std::is_trivially_default_constructible, Structurals...>>;
  using key_t = vec<typename Domain::index_type, Domain::dim>;
  using value_t = typename Domain::index_type;
  static constexpr auto sentinel_v = value_t(-1);
  // data members
  value_t _capacity;
  value_t *_cnt;
  key_t *_activeKeys;   //
  value_t *_indexTable; //
  // func members
  template <typename Allocator>
  void allocate_table(Allocator allocator, value_t capacity) {
    _capacity = capacity;
    _cnt = static_cast<value_t *>(allocator.allocate(sizeof(value_t)));
    _activeKeys =
        static_cast<key_t *>(allocator.allocate(sizeof(key_t) * capacity));
    /// lookup table
    _indexTable = static_cast<value_t *>(
        allocator.allocate(sizeof(value_t) * Domain::extent));
  }
  template <typename Allocator>
  void resize_table(Allocator allocator, std::size_t capacity) {
    allocator.deallocate(_activeKeys, _capacity);
    _activeKeys = (key_t *)(allocator.allocate(sizeof(key_t) * capacity));
    _capacity = capacity;
  }
  template <typename Allocator> void deallocate(Allocator allocator) {
    allocator.deallocate(_cnt, sizeof(value_t));
    allocator.deallocate(_activeKeys, sizeof(key_t) * _capacity);
    allocator.deallocate(_indexTable, sizeof(value_t) * Domain::extent);
    base_t::deallocate(allocator);
    _capacity = 0;
    _cnt = nullptr;
    _activeKeys = nullptr;
    _indexTable = nullptr;
  }
  template <typename... Indices> constexpr auto &val(Indices &&... indices) {
    return *reinterpret_cast<value_t *>(
        _indexTable + Domain::offset(std::forward<Indices>(indices)...));
  }
  template <std::size_t... Is>
  constexpr auto &index_impl(key_t &coord, std::index_sequence<Is...>) {
    return *reinterpret_cast<value_t *>(_indexTable +
                                        Domain::offset((coord[Is])...));
  }
  constexpr auto &index(key_t coord) {
    return index_impl(coord, std::make_index_sequence<Domain::dim>{});
  }
  template <std::size_t... Is>
  constexpr const auto &index_impl(key_t &coord,
                                   std::index_sequence<Is...>) const {
    return *reinterpret_cast<value_t *>(_indexTable +
                                        Domain::offset((coord[Is])...));
  }
  constexpr const auto &index(key_t coord) const {
    return index_impl(coord, std::make_index_sequence<Domain::dim>{});
  }
};
/// dense
template <typename Decoration, typename Domain, attrib_layout Layout,
          typename... Structurals>
struct structural<structural_type::dense, Decoration, Domain, Layout,
                  Structurals...>
    : identity_structural,
      structural_traits<structural_type::dense, Decoration, Domain, Layout,
                        Structurals...> {
  using base_t = structural_traits<structural_type::dense, Decoration, Domain,
                                   Layout, Structurals...>;
  using CheckPrerequisite =
      void_t<is_base_of<identity_structural_index, Domain>,
             is_base_of<identity_decorator, Decoration>,
             is_base_of<identity_structural, Structurals...>,
             satisfy<std::is_trivially_default_constructible, Structurals...>>;
};

/// dynamic
template <typename Decoration, typename Domain, attrib_layout Layout,
          typename... Structurals>
struct structural<structural_type::dynamic, Decoration, Domain, Layout,
                  Structurals...>
    : identity_structural,
      structural_traits<structural_type::dynamic, Decoration, Domain, Layout,
                        Structurals...> {
  using base_t = structural_traits<structural_type::dynamic, Decoration, Domain,
                                   Layout, Structurals...>;
  using CheckPrerequisite =
      void_t<is_base_of<identity_structural_index, Domain>,
             is_base_of<identity_decorator, Decoration>,
             is_base_of<identity_structural, Structurals...>,
             satisfy<std::is_trivially_default_constructible, Structurals...>,
             enable_if_t<Domain::dim == 1>>;
  using value_t = unsigned long long int;
  std::size_t _capacity;
  template <typename Allocator>
  void allocate_handle(Allocator allocator,
                       std::size_t capacity = Domain::extent) {
    if (capacity != 0)
      this->_handle.ptr =
          allocator.allocate(capacity * base_t::element_storage_size);
    else
      this->_handle.ptr = nullptr;
    _capacity = capacity;
  }
  template <typename Allocator>
  void resize(Allocator allocator, std::size_t capacity) {
    allocator.deallocate(this->_handle.ptr, _capacity);
    _capacity = capacity; ///< each time multiply by 2
    this->_handle.ptr =
        allocator.allocate(_capacity * base_t::element_storage_size);
  }
  template <typename Allocator> void deallocate(Allocator allocator) {
    allocator.deallocate(this->_handle.ptr,
                         _capacity * base_t::element_storage_size);
    _capacity = 0;
    this->_handle.ptr = nullptr;
  }
};

} // namespace mn

#endif