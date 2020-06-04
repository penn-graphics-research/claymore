#ifndef __HASH_CUH_
#define __HASH_CUH_
#include <MnBase/AggregatedAttribs.cuh>
#include <MnBase/Math/Bit/Bits.h>
#include <MnSystem/Cuda/HostUtils.hpp>
///< For multiGPU_TEST Flag only
#include <MnWorld/SimulationSetting.h>

namespace mn {
#if 0
	template<typename KeyType, typename ValueType, typename HashTableAttribList = std::tuple<KeyType, ValueType>>
#if MULTI_GPU_TEST		///< HashTable on UM
	struct HashTable : public CudaAttribs<HashTableAttribList, 1> {
		using Base = CudaAttribs<HashTableAttribList, 1>;
#else	
	struct HashTable : public CudaAttribs<HashTableAttribList> {
		using Base = CudaAttribs<HashTableAttribList>;
#endif
#endif
template<typename KeyType, typename ValueType, int opt=1, typename HashTableAttribList = std::tuple<KeyType, ValueType>>
	struct HashTable : public CudaAttribs<HashTableAttribList, opt> {
		using Base = CudaAttribs<HashTableAttribList, opt>;
		struct CudaPort {
			static_assert(sizeof(KeyType) == 8 || sizeof(KeyType) == 4, "Hashtable KeyType neither 64-bit or 32-bit!\n");
			static_assert(sizeof(unsigned long long int) == 8 && sizeof(unsigned int) == 4, 
				"\'unsigned long long int\' not 64-bit or \'unsigned int\' not 32-bit!\n");
			using KT = std::conditional_t<sizeof(KeyType) == 8, unsigned long long int, unsigned int>;
			using VT = ValueType;
			enum : uint64_t { HashKeyOffset = 127 };
			enum : KT { SentinelValue = (KT)-1 };
			std::size_t	_mask;
			KT		*_keyTable;	///< should be cuda compatible!
			VT		*_valueTable;
			VT		*_entryCount;
			CudaPort(std::size_t mask, KeyType* keyTable, ValueType* valTable, ValueType* entryCount) :
				_mask{ std::move(mask) }, _keyTable{ reinterpret_cast<KT*>(keyTable) }, _valueTable{ reinterpret_cast<VT*>(valTable) },
				_entryCount{ reinterpret_cast<VT*>(entryCount) } {}
			template<typename KeyT = KT>
			__forceinline__ __device__ void insert(KeyT && key) {
				KeyType hashkey = std::forward<KeyT>(key) & _mask, ori;

				while ((ori = _keyTable[hashkey]) != std::forward<KeyT>(key)) {
					if (ori == SentinelValue)
						ori = atomicCAS(_keyTable + hashkey, SentinelValue, std::forward<KeyT>(key));
					if (_keyTable[hashkey] == std::forward<KeyT>(key)) {	///< found
						if (ori == SentinelValue) {
							_valueTable[hashkey] = atomicAdd(_entryCount, 1); ///< created a record
						}
						break;
					}
					hashkey += HashKeyOffset; ///< search next entry
					if (hashkey > _mask)
						hashkey = hashkey & _mask;
				};
			}
			template<typename RecoverKeyOp, typename KeyT = KT>
			__forceinline__ __device__ void insert(KeyT && key, KeyT* _records, RecoverKeyOp recover_op) {
				KeyType hashkey = std::forward<KeyT>(key) & _mask, ori;

				while ((ori = _keyTable[hashkey]) != std::forward<KeyT>(key)) {
					if (ori == SentinelValue)
						ori = atomicCAS(_keyTable + hashkey, SentinelValue, std::forward<KeyT>(key));
					if (_keyTable[hashkey] == std::forward<KeyT>(key)) {	///< found
						if (ori == SentinelValue) {
							_records[_valueTable[hashkey] = atomicAdd(_entryCount, 1)] = recover_op(std::forward<KeyT>(key)); ///< created a record
						}
						break;
					}
					hashkey += HashKeyOffset; ///< search next entry
					if (hashkey > _mask)
						hashkey = hashkey & _mask;
				};
			}
			template<typename KeyT = KeyType>
			__forceinline__ __device__ auto query(KeyT && key) const ->ValueType {
				KeyType hashkey = std::forward<KeyT>(key) & _mask, ori;
				while ((ori = _keyTable[hashkey]) != std::forward<KeyT>(key)) {
					if (ori == SentinelValue) return -1;
					hashkey += HashKeyOffset; ///< search next entry
					if (hashkey > _mask)
						hashkey = hashkey & _mask;
				}
				return _valueTable[hashkey];
			}
		};

		auto estimateTableSize(std::size_t entrySize) const noexcept { return entrySize << 5; }
		// auto estimateTableSize(std::size_t entrySize) const noexcept { return entrySize << 5; }
		HashTable() = delete;
		template<typename Integer>
		HashTable(Integer size) : Base{ static_cast<std::size_t>(1) << bit_count(estimateTableSize(size)) }, 
			_entrySize{ size },
			_tableSize{ static_cast<std::size_t>(1) << bit_count(estimateTableSize(size)) } {
			static_assert(std::is_integral<ValueType>::value, "Hashtable value type is not integer.");
			_mask = _tableSize - 1;
			//printf("required size: %llu, actual alloced size: %llu, mask: %llx(%llu)\n", size, _tableSize, _mask, _mask);
			_keyTable = reinterpret_cast<KeyType*>(Base::_attribs[0]);
			_valueTable = reinterpret_cast<ValueType*>(Base::_attribs[1]);
// #if MULTI_GPU_TEST
// 			checkCudaErrors(cudaMallocManaged((void**)&_entryCount, sizeof(int)));
// #else
// 			checkCudaErrors(cudaMalloc((void**)&_entryCount, sizeof(int)));
// #endif
			if(opt == 1) {
				checkCudaErrors(cudaMallocManaged((void**)&_entryCount, sizeof(int)));
			}
			else {
				checkCudaErrors(cudaMalloc((void**)&_entryCount, sizeof(int)));				
			}

		}
		~HashTable() {
			cudaFree(_entryCount);
		}
		std::size_t		_entrySize;
		std::size_t		_tableSize, _mask;
		KeyType		*_keyTable;
		ValueType	*_valueTable;
		ValueType	*_entryCount;

		void clearCount() {
			cudaMemset(_entryCount, 0, sizeof(ValueType));
		}
		void clearCount(cudaStream_t stream) {
			cudaMemsetAsync(_entryCount, 0, sizeof(ValueType), stream);
		}
		void clear() {
			cudaMemset(_entryCount, 0, sizeof(ValueType));
			cudaMemset(_keyTable, 0xff, sizeof(KeyType) * _tableSize);
			cudaMemset(_valueTable, 0xff, sizeof(ValueType) * _tableSize);
		}
		void clear(cudaStream_t stream) {
			cudaMemsetAsync(_entryCount, 0, sizeof(ValueType), stream);
			cudaMemsetAsync(_keyTable, 0xff, sizeof(KeyType) * _tableSize, stream);
			cudaMemsetAsync(_valueTable, 0xff, sizeof(ValueType) * _tableSize, stream);
		}
		auto getPort() {
			return CudaPort{ _mask, _keyTable, _valueTable, _entryCount };
		}
		void retrieveEntryCount(ValueType* addr, cudaStream_t stream=cudaStreamDefault) {
			cudaMemcpyAsync(addr, _entryCount, sizeof(ValueType), cudaMemcpyDeviceToHost, stream);
		}
#if MULTI_GPU_TEST
		void umAdvise_AccessBy_Init(int dev_id) {
	#if DEBUG
			printf("[DEBUG] Setting HashTable_UM_Advice AccessBy for Dev %d\n", dev_id);
	#endif
			// checkCudaErrors(cudaMemAdvise(_entryCount, sizeof(int), cudaMemAdviseSetAccessedBy, dev_id));
			// checkCudaErrors(cudaMemAdvise(_keyTable, sizeof(KeyType) * _tableSize, cudaMemAdviseSetAccessedBy, dev_id));
			// checkCudaErrors(cudaMemAdvise(_valueTable, sizeof(ValueType) * _tableSize, cudaMemAdviseSetAccessedBy, dev_id));
		}
		void umAdvise_PreferredLoc_Init(int dev_id = 0) {
	#if DEBUG
			printf("[DEBUG] Setting HashTable_UM_Advice PreferredLocation for Dev %d\n", dev_id);
	#endif
			// checkCudaErrors(cudaMemAdvise(_entryCount, sizeof(int), cudaMemAdviseSetPreferredLocation, dev_id));
			// checkCudaErrors(cudaMemAdvise(_keyTable, sizeof(KeyType) * _tableSize, cudaMemAdviseSetPreferredLocation, dev_id));
			// checkCudaErrors(cudaMemAdvise(_valueTable, sizeof(ValueType) * _tableSize, cudaMemAdviseSetPreferredLocation, dev_id));
		}
#endif
	};

}

#endif