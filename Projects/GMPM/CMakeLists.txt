if (CUDA_FOUND)

add_cuda_executable(gmpm)
target_sources(gmpm
    PRIVATE     gmpm.cu
)
target_link_libraries(gmpm
    PRIVATE     mnio
                cxxopts
                spdlog
                fmt
                rapidjson
                range-v3
                filesystem
)

#Copy scene files
#TODO: Move them into extra directory
install(DIRECTORY
	${CMAKE_CURRENT_LIST_DIR}/ DESTINATION ${CMAKE_INSTALL_BINDIR}
	FILES_MATCHING PATTERN "*.json"
)

endif()