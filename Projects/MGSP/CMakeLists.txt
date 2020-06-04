if (CUDA_FOUND)

add_cuda_executable(mgsp)
target_sources(mgsp 
    PRIVATE     mgsp.cu
)
target_link_libraries(mgsp 
    PRIVATE     mnio
                fmt
)

endif()