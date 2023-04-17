#ifndef HASH_CUH
#define HASH_CUH
#include "MnBase/AggregatedAttribs.cuh"
#include "MnBase/Math/Bit/Bits.h"
#include "MnSystem/Cuda/HostUtils.hpp"
///< For multiGPU_TEST Flag only
#include "MnWorld/SimulationSetting.h"

#define NO_OPT_HASH_TABLE 0

namespace mn {
#if NO_OPT_HASH_TABLE
template<typename KeyType, typename ValueType, typename HashTableAttribList = std::tuple<KeyType, ValueType>>
#	if MULTI_GPU_TEST///< HashTable on UM
struct HashTable : public CudaAttribs<HashTableAttribList, 1> {
	using Base = CudaAttribs<HashTableAttribList, 1>;
#	else
struct HashTable : public CudaAttribs<HashTableAttribList> {
	using Base = CudaAttribs<HashTableAttribList>;
#	endif
#endif
	template<typename KeyType, typename ValueType, int Opt = 1, typename HashTableAttribList = std::tuple<KeyType, ValueType>>
	struct HashTable : public CudaAttribs<HashTableAttribList, Opt> {
		using Base = CudaAttribs<HashTableAttribList, Opt>;
		struct CudaPort {
			static_assert(sizeof(KeyType) == 8 || sizeof(KeyType) == 4, "Hashtable KeyType neither 64-bit or 32-bit!\n");
			static_assert(sizeof(unsigned long long int) == 8 && sizeof(unsigned int) == 4, "\'unsigned long long int\' not 64-bit or \'unsigned int\' not 32-bit!\n");

			using KT = std::conditional_t<sizeof(KeyType) == 8, unsigned long long int, unsigned int>;
			using VT = ValueType;

			enum : uint64_t {
				HASH_KEY_OFFSET = 127
			};
			enum : KT {
				SENTINEL_VALUE = (KT) -1
			};

			std::size_t mask;
			KT* key_table;///< should be cuda compatible!
			VT* value_table;
			VT* entry_count;

			CudaPort(std::size_t mask, KeyType* key_table, ValueType* val_table, ValueType* entry_count)
				: mask {std::move(mask)}
				, key_table {static_cast<KT*>(static_cast<void*>(key_table))}
				, value_table {static_cast<VT*>(static_cast<void*>(val_table))}
				, entry_count {static_cast<VT*>(static_cast<void*>(entry_count))} {}

			template<typename KeyT = KT>
			__forceinline__ __device__ void insert(KeyT&& key) {
				KeyType hashkey = std::forward<KeyT>(key) & mask;
				KeyType ori;

				while((ori = key_table[hashkey]) != std::forward<KeyT>(key)) {
					if(ori == SENTINEL_VALUE) {
						ori = atomicCAS(key_table + hashkey, SENTINEL_VALUE, std::forward<KeyT>(key));
					}
					if(key_table[hashkey] == std::forward<KeyT>(key)) {///< found
						if(ori == SENTINEL_VALUE) {
							value_table[hashkey] = atomicAdd(entry_count, 1);///< created a record
						}
						break;
					}
					hashkey += HASH_KEY_OFFSET;///< search next entry
					if(hashkey > mask) {
						hashkey = hashkey & mask;
					}
				};
			}

			template<typename RecoverKeyOp, typename KeyT = KT>
			__forceinline__ __device__ void insert(KeyT&& key, KeyT* _records, RecoverKeyOp recover_op) {
				KeyType hashkey = std::forward<KeyT>(key) & mask;
				KeyType ori;

				while((ori = key_table[hashkey]) != std::forward<KeyT>(key)) {
					if(ori == SENTINEL_VALUE) {
						ori = atomicCAS(key_table + hashkey, SENTINEL_VALUE, std::forward<KeyT>(key));
					}
					if(key_table[hashkey] == std::forward<KeyT>(key)) {///< found
						if(ori == SENTINEL_VALUE) {
							_records[value_table[hashkey] = atomicAdd(entry_count, 1)] = recover_op(std::forward<KeyT>(key));///< created a record
						}
						break;
					}
					hashkey += HASH_KEY_OFFSET;///< search next entry
					if(hashkey > mask) {
						hashkey = hashkey & mask;
					}
				};
			}

			template<typename KeyT = KeyType>
			__forceinline__ __device__ auto query(KeyT&& key) const -> ValueType {
				KeyType hashkey = std::forward<KeyT>(key) & mask;
				KeyType ori;

				while((ori = key_table[hashkey]) != std::forward<KeyT>(key)) {
					if(ori == SENTINEL_VALUE) {
						return -1;
					}
					hashkey += HASH_KEY_OFFSET;///< search next entry
					if(hashkey > mask) {
						hashkey = hashkey & mask;
					}
				}
				return value_table[hashkey];
			}
		};

		std::size_t entry_size;
		std::size_t table_size;
		std::size_t mask;
		KeyType* key_table;
		ValueType* value_table;
		ValueType* entry_count;

		HashTable() = delete;

		template<typename Integer>
		HashTable(Integer size)
			: Base {static_cast<std::size_t>(1) << bit_count(estimate_table_size(size))}
			, entry_size {size}
			, table_size {static_cast<std::size_t>(1) << bit_count(estimate_table_size(size))} {
			static_assert(std::is_integral<ValueType>::value, "Hashtable value type is not integer.");
			mask = table_size - 1;
			//printf("required size: %llu, actual alloced size: %llu, mask: %llx(%llu)\n", size, table_size, mask, mask);
			key_table	= static_cast<KeyType*>(static_cast<void*>(Base::_attribs[0]));
			value_table = static_cast<ValueType*>(static_cast<void*>(Base::_attribs[1]));
			// #if MULTI_GPU_TEST
			// 			checkCudaErrors(cudaMallocManaged((void**)&entry_count, sizeof(int)));
			// #else
			// 			checkCudaErrors(cudaMalloc((void**)&entry_count, sizeof(int)));
			// #endif
			if(Opt == 1) {
				checkCudaErrors(cudaMallocManaged(static_cast<void**>(static_cast<void*>(&entry_count)), sizeof(int)));
			} else {
				checkCudaErrors(cudaMalloc(static_cast<void**>(static_cast<void*>(&entry_count)), sizeof(int)));
			}
		}

		~HashTable() {
			cudaFree(entry_count);
		}

		auto estimate_table_size(std::size_t entry_size) const noexcept {
			return entry_size << 5;
		}
		// auto estimate_table_size(std::size_t entry_size) const noexcept { return entry_size << 5; }

		void clear_count() {
			cudaMemset(entry_count, 0, sizeof(ValueType));
		}
		void clear_count(cudaStream_t stream) {
			cudaMemsetAsync(entry_count, 0, sizeof(ValueType), stream);
		}
		void clear() {
			cudaMemset(entry_count, 0, sizeof(ValueType));
			cudaMemset(key_table, 0xff, sizeof(KeyType) * table_size);
			cudaMemset(value_table, 0xff, sizeof(ValueType) * table_size);
		}
		void clear(cudaStream_t stream) {
			cudaMemsetAsync(entry_count, 0, sizeof(ValueType), stream);
			cudaMemsetAsync(key_table, 0xff, sizeof(KeyType) * table_size, stream);
			cudaMemsetAsync(value_table, 0xff, sizeof(ValueType) * table_size, stream);
		}
		auto get_port() {
			return CudaPort {mask, key_table, value_table, entry_count};
		}
		void retrieve_entry_count(ValueType* addr, cudaStream_t stream = cudaStreamDefault) {
			cudaMemcpyAsync(addr, entry_count, sizeof(ValueType), cudaMemcpyDeviceToHost, stream);
		}
#if MULTI_GPU_TEST
		void um_advise_access_by_init(int dev_id) {
#	if DEBUG
			printf("[DEBUG] Setting HashTable_UM_Advice AccessBy for Dev %d\n", dev_id);
#	endif
			// checkCudaErrors(cudaMemAdvise(entry_count, sizeof(int), cudaMemAdviseSetAccessedBy, dev_id));
			// checkCudaErrors(cudaMemAdvise(key_table, sizeof(KeyType) * table_size, cudaMemAdviseSetAccessedBy, dev_id));
			// checkCudaErrors(cudaMemAdvise(value_table, sizeof(ValueType) * table_size, cudaMemAdviseSetAccessedBy, dev_id));
		}
		void um_advise_preferred_loc_init(int dev_id = 0) {
#	if DEBUG
			printf("[DEBUG] Setting HashTable_UM_Advice PreferredLocation for Dev %d\n", dev_id);
#	endif
			// checkCudaErrors(cudaMemAdvise(entry_count, sizeof(int), cudaMemAdviseSetPreferredLocation, dev_id));
			// checkCudaErrors(cudaMemAdvise(key_table, sizeof(KeyType) * table_size, cudaMemAdviseSetPreferredLocation, dev_id));
			// checkCudaErrors(cudaMemAdvise(value_table, sizeof(ValueType) * table_size, cudaMemAdviseSetPreferredLocation, dev_id));
		}
#endif
	};

}// namespace mn

#endif