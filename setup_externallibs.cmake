#####
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
set(CPM_DOWNLOAD_VERSION 0.38.1)

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION} AND CPM_VERSION STREQUAL CPM_DOWNLOAD_VERSION))
  message(STATUS "Downloading CPM.cmake")
  file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})
#####

#Suppress warning
cmake_policy(SET CMP0077 NEW)

# rapidjson
CPMAddPackage(
  NAME rapidjson
  GIT_TAG 949c771b03de448bdedea80c44a4a5f65284bfeb
  GITHUB_REPOSITORY Tencent/rapidjson
  OPTIONS
    "RAPIDJSON_BUILD_CXX11 Off"
    "RAPIDJSON_BUILD_CXX17 On"
)
if(rapidjson_ADDED)
  add_library(rapidjson INTERFACE IMPORTED)
  target_include_directories(rapidjson SYSTEM INTERFACE ${rapidjson_SOURCE_DIR}/include)
endif()

# cxxopts
CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.0
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)

# spdlog
CPMAddPackage(
  NAME spdlog
  VERSION 1.7.0
  GITHUB_REPOSITORY gabime/spdlog
)

#Set new timestamps
cmake_policy(SET CMP0135 NEW)

set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

# ranges
CPMAddPackage(
  NAME range-v3
  URL https://github.com/ericniebler/range-v3/archive/0.10.0.zip
  VERSION 0.10.0
  # the range-v3 CMakeLists screws with configuration options
  DOWNLOAD_ONLY True
)
if(range-v3_ADDED) 
  add_library(range-v3 INTERFACE IMPORTED)
  target_include_directories(range-v3 SYSTEM INTERFACE ${range-v3_SOURCE_DIR}/include)
endif()

# fmt
CPMAddPackage(
  NAME fmt
  GIT_TAG 6.2.1
  GITHUB_REPOSITORY fmtlib/fmt
)
if(fmt_ADDED) 
  target_include_directories(fmt SYSTEM INTERFACE ${fmt_SOURCE_DIR}/include)
endif()

# glm
CPMAddPackage(
  NAME glm
  GIT_TAG 0.9.9.8
  GITHUB_REPOSITORY g-truc/glm
)

# filesystem
CPMAddPackage(
  NAME filesystem 
  GITHUB_REPOSITORY gulrak/filesystem
  VERSION 1.3.4
)
if(filesystem_ADDED) 
  add_library(filesystem INTERFACE IMPORTED)
  target_include_directories(filesystem SYSTEM INTERFACE ${filesystem_SOURCE_DIR}/include)
endif()