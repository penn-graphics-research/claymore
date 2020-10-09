#####
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
set(CPM_DOWNLOAD_VERSION 0.27.1)

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION} AND CPM_VERSION STREQUAL CPM_DOWNLOAD_VERSION))
  message(STATUS "Downloading CPM.cmake")
  file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})
#####

# rapidjson
CPMAddPackage(
  NAME rapidjson
  GIT_TAG f56928de85d56add3ca6ae7cf7f119a42ee1585b
  GITHUB_REPOSITORY Tencent/rapidjson
)
if(rapidjson_ADDED)
  add_library(rapidjson INTERFACE IMPORTED)
  target_include_directories(rapidjson INTERFACE ${rapidjson_SOURCE_DIR}/include)
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
  target_include_directories(range-v3 INTERFACE ${range-v3_SOURCE_DIR}/include)
endif()

# fmt
CPMAddPackage(
  NAME fmt
  GIT_TAG 6.2.1
  GITHUB_REPOSITORY fmtlib/fmt
)

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
  target_include_directories(filesystem INTERFACE ${filesystem_SOURCE_DIR}/include)
endif()