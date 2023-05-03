if (CUDA_FOUND)
	add_cuda_executable(mgsp)
	
	target_sources(mgsp 
		PRIVATE     mgsp.cu
	)
	
	target_link_libraries(mgsp 
		PRIVATE     mnio
					fmt
	)

	#Add headers as system headers
	get_target_property(HEADERS_FMT fmt INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

	target_include_directories(mgsp
		SYSTEM
		PRIVATE ${CMAKE_SOURCE_DIR}/Library
		PRIVATE ${HEADERS_FMT}
	)

endif()