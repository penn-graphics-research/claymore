#ifndef CUDA_EXECUTION_POLICY_H
#define CUDA_EXECUTION_POLICY_H
#include <driver_types.h>

#include <string>

namespace mn {

struct LaunchConfig {
	dim3 dg {};
	dim3 db {};
	std::size_t shmem {0};
	cudaStream_t sid {cudaStreamDefault};

	template<typename IndexType0, typename IndexType1>
	LaunchConfig(IndexType0 gs, IndexType1 bs)
		: dg {static_cast<unsigned int>(gs)}
		, db {static_cast<unsigned int>(bs)}
		, shmem {0}
		, sid {cudaStreamDefault} {}

	template<typename IndexType0, typename IndexType1, typename IndexType2>
	LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem)
		: dg {static_cast<unsigned int>(gs)}
		, db {static_cast<unsigned int>(bs)}
		, shmem {static_cast<std::size_t>(mem)}
		, sid {cudaStreamDefault} {}

	template<typename IndexType0, typename IndexType1, typename IndexType2>
	LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem, cudaStream_t stream)
		: dg {static_cast<unsigned int>(gs)}
		, db {static_cast<unsigned int>(bs)}
		, shmem {static_cast<std::size_t>(mem)}
		, sid {stream} {}
};

struct LaunchInput {///< could contain more information on operation (error checking/ time recording/ etc...)
   private:
	const std::string kernel_name;
	const int num_threads;
	const std::size_t shared_mem_bytes;

   public:
	LaunchInput() = delete;
	LaunchInput(std::string kernel, int task_num, std::size_t shared_mem_bytes = 0)
		: kernel_name(kernel)
		, num_threads(task_num)
		, shared_mem_bytes(shared_mem_bytes) {}

	const std::string& name() {
		return kernel_name;
	}

	const int& threads() {
		return num_threads;
	}

	const std::size_t& mem_bytes() {
		return shared_mem_bytes;
	}
};

/// kernel launching configuration
struct ExecutionPolicy {
   private:
	int grid_size {0};
	int block_size {0};
	std::size_t shared_mem_bytes {0};
	bool sync {false};

   public:
	ExecutionPolicy() {}
	ExecutionPolicy(int gs, int bs, std::size_t memsize, bool s)
		: grid_size(gs)
		, block_size(bs)
		, shared_mem_bytes(memsize)
		, sync(s) {}

	int get_grid_size() const {
		return grid_size;
	}

	int get_block_size() const {
		return block_size;
	}

	std::size_t get_shared_mem_bytes() const {
		return shared_mem_bytes;
	}

	bool need_sync() const {
		return sync;
	}
};

}// namespace mn

#endif