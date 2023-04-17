#ifndef HOST_UTILS_HPP
#define HOST_UTILS_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <string>

namespace mn {

#define check_thrust_errors(func)                                                              \
	try {                                                                                      \
		func;                                                                                  \
	} catch(thrust::system_error & e) {                                                        \
		std::cout << std::string(__FILE__) << ":" << __LINE__ << " " << e.what() << std::endl; \
	}

inline static const char* cuda_get_error_enum(cudaError_t error) {
	return cudaGetErrorName(error);
}

#ifdef __DRIVER_TYPES_H__
#	ifndef DEVICE_RESET
#		define DEVICE_RESET cudaDeviceReset();
#	endif
#else
#	ifndef DEVICE_RESET
#		define DEVICE_RESET
#	endif
#endif

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
	if(result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cuda_get_error_enum(result), func);
		DEVICE_RESET
		// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define check_cuda_errors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define get_last_cuda_error(msg) get_last_cuda_error_impl(msg, __FILE__, __LINE__)

inline void get_last_cuda_error_impl(const char* errorMessage, const char* file, const int line) {
	cudaError_t err = cudaGetLastError();

	if(cudaSuccess != err) {
		fprintf(
			stderr,
			"%s(%i) : get_last_cuda_error() CUDA error :"
			" %s : (%d) %s.\n",
			file,
			line,
			errorMessage,
			static_cast<int>(err),
			cudaGetErrorString(err)
		);
		DEVICE_RESET
		exit(EXIT_FAILURE);
	}
}

template<class T>
__inline__ __host__ T* get_raw_ptr(thrust::device_vector<T>& V) {
	return thrust::raw_pointer_cast(V.data());
}
template<class T>
__inline__ __host__ thrust::device_ptr<T> get_device_ptr(thrust::device_vector<T>& V) {
	return thrust::device_ptr<T>(thrust::raw_pointer_cast(V.data()));
}
template<class T>
__inline__ __host__ thrust::device_ptr<T> get_device_ptr(T* V) {
	return thrust::device_ptr<T>(V);
}

inline void report_memory(std::string msg) {
	std::size_t free_byte;
	std::size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if(cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	double free_db	= (double) free_byte;
	double total_db = (double) total_byte;
	double used_db	= total_db - free_db;
	printf("GPU memory usage (%s): used = %f, free = %f MB, total = %f MB\n", msg.data(), used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

}// namespace mn

#endif