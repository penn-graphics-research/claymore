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
)

#Add headers as system headers
get_target_property(HEADERS_CXXOPTS cxxopts INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(HEADERS_SPDLOGS spdlog INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(HEADERS_FMT fmt INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
get_target_property(HEADERS_RAPIDJSON rapidjson INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
get_target_property(HEADERS_RANGE_V3 range-v3 INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

target_include_directories(gmpm
	SYSTEM
	PRIVATE ${CMAKE_SOURCE_DIR}/Library
	PRIVATE ${HEADERS_CXXOPTS}
	PRIVATE ${HEADERS_SPDLOGS}
	PRIVATE ${HEADERS_FMT}
	PRIVATE ${HEADERS_RAPIDJSON}
	PRIVATE ${HEADERS_RANGE_V3}
)


#Copy scene files
#TODO: Move them into extra directory
install(DIRECTORY
	${CMAKE_CURRENT_LIST_DIR}/ DESTINATION ${CMAKE_INSTALL_BINDIR}
	FILES_MATCHING PATTERN "*.json"
)

endif()