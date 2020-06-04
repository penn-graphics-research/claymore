if (CUDA_FOUND)

add_cuda_executable(gmpm)
target_sources(gmpm
    PRIVATE     gmpm.cu
)
target_link_libraries(gmpm
    PRIVATE     mnio
                fmt
)

endif()