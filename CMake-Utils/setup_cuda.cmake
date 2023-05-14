message("## setup cuda")

#Find CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	message("-- cuda-compiler " ${CMAKE_CUDA_COMPILER})
else()
	message(STATUS "No CUDA support")
endif()
set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})

#Set Architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# reference: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
	get_property(old_flags
		TARGET ${EXISTING_TARGET}
		PROPERTY INTERFACE_COMPILE_OPTIONS
	)
	if(NOT "${old_flags}" STREQUAL "")
		string(REPLACE ";" "," CUDA_flags "${old_flags}")
		set_property(
			TARGET ${EXISTING_TARGET}
			PROPERTY INTERFACE_COMPILE_OPTIONS
			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
		)
	endif()
endfunction()

#Cmake does not handle linking correctly for separable compilation: https://gitlab.kitware.com/cmake/cmake/-/issues/22788
function(GET_DEVICE_LINK_PATH TARGET_NAME ret)
	cmake_path(SET DEVICE_LINK_PATH ${CMAKE_BINARY_DIR})
	cmake_path(APPEND DEVICE_LINK_PATH "CMakeFiles")
	cmake_path(APPEND DEVICE_LINK_PATH ${TARGET_NAME}.dir)
	cmake_path(APPEND DEVICE_LINK_PATH ${CMAKE_BUILD_TYPE})
	cmake_path(APPEND DEVICE_LINK_PATH "cmake_device_link.obj")
	set(${ret} ${DEVICE_LINK_PATH} PARENT_SCOPE)
endfunction()

function(add_cuda_executable binary)
	if(CUDA_FOUND)
		add_executable(${binary} ${ARGN})

		# seems not working
		target_compile_options(${binary} 
			PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g> --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
		)

		target_compile_features(${binary}
			PRIVATE cuda_std_17
		)

		set_target_properties(${binary}
			PROPERTIES	CUDA_EXTENSIONS ON
						CUDA_SEPARABLE_COMPILATION OFF
						#LINKER_LANGUAGE CUDA
						RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		)

		GET_DEVICE_LINK_PATH(${binary} DEVICE_LINK_PATH)

		target_link_libraries(${binary}
			PRIVATE mncuda
		)

		target_link_options(${binary}
			PRIVATE /NODEFAULTLIB:libcmt.lib
		)

		install(TARGETS
			${binary}
		)

		message("-- [${binary}]\tcuda executable build config")
	endif()
endfunction(add_cuda_executable)

function(add_cuda_library library)
	if(CUDA_FOUND)
	add_library(${library} ${ARGN})

	# seems not working
	target_compile_options(${library} 
		PUBLIC        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g> --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
	)

	#target_link_options(${library} 
	#  PRIVATE       $<$<LINKER_LANGUAGE:CUDA>:-arch=sm_75>
	#)

	target_compile_features(${library}
		PRIVATE cuda_std_17
	)

	set_target_properties(${library}
		PROPERTIES	CUDA_EXTENSIONS ON
					CUDA_SEPARABLE_COMPILATION OFF
					CUDA_RESOLVE_DEVICE_SYMBOLS OFF
					POSITION_INDEPENDENT_CODE ON
					#LINKER_LANGUAGE CUDA
	)

	target_compile_definitions(${library} 
		PUBLIC	CMAKE_GENERATOR_PLATFORM=x64
	)

	message("-- [${library}]\tcuda executable build config")
	endif()
endfunction(add_cuda_library)