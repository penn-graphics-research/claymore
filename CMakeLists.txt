cmake_minimum_required(VERSION 3.15)    # support relative path

message("===============================================================")

include("project_info.in")
project(Mn 
    VERSION     ${project_version}
    DESCRIPTION ${project_description}
    LANGUAGES   CXX CUDA
)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)

#Fixes https://gitlab.kitware.com/cmake/cmake/-/issues/24154 for older cmake versions
set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
include(CMake-Utils/setup_cpp.cmake)
include(CMake-Utils/setup_cuda.cmake)
include(CMake-Utils/setup_externallibs.cmake)


add_subdirectory(Externals)

# header only libraries
include_directories(
	SYSTEM
	Externals/variant
	Externals/optional
	Externals/function_ref
)

message("===============================================================")

include_directories(Library)
add_subdirectory(Library)
add_subdirectory(Projects)
