include(sourcelist.cmake)

# build: PRIVATE, usage: INTERFACE, both: PUBLIC.
if (CUDA_FOUND)
	add_cuda_library(mncuda)
	
	target_include_directories(mncuda
		PUBLIC ${PROJECT_SOURCE_DIR}/Library
	)
	
	target_sources(mncuda
		PRIVATE	${CudaSystemSrc}
	)
	
	target_precompile_headers(mncuda 
		INTERFACE	Cuda/Cuda.h
					Cuda/DeviceUtils.cuh
					Cuda/HostUtils.hpp
	)

endif()

add_library(mnio)

set_target_properties(mnio
    PROPERTIES  #CXX_STANDARD 20
                #LINKER_LANGUAGE CXX
                POSITION_INDEPENDENT_CODE ON
)

target_include_directories(mnio
	PUBLIC ${PROJECT_SOURCE_DIR}/Library
)

target_compile_definitions(mnio
    INTERFACE   AssetDirPath="${PROJECT_SOURCE_DIR}/Data/"
)

target_sources(mnio 
    PRIVATE     IO/IO.cpp
)

target_precompile_headers(mnio 
    INTERFACE   IO/IO.h
)

target_link_libraries(mnio 
    PUBLIC      partio
)
