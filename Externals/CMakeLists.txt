find_package(Git)
include(FetchContent)
include(ExternalProject)
include(DownloadProject)

# partio
message("## building partio")
add_subdirectory(partio)

# optional
#FetchContent_Declare(
#    optional
#    GIT_REPOSITORY https://github.com/TartanLlama/optional.git
#    GIT_TAG 83a0b49e5e9571438c3ebae638e9a37502911b8c
#)
#FetchContent_MakeAvailable(optional)

# function_ref
#FetchContent_Declare(
#    function_ref
#    GIT_REPOSITORY https://github.com/TartanLlama/function_ref.git
#    GIT_TAG 556c2c3fbfc144971c01ba6d177aaab48cab6870
#)
##FetchContent_MakeAvailable(function_ref)
#if (NOT function_ref_POPULATED)
#    FetchContent_Populate(function_ref)
#endif()

# variant
#FetchContent_Declare(
#    variant
#    GIT_REPOSITORY https://github.com/mpark/variant.git
#    GIT_TAG 4988879a9f5a95d72308eca2b1779db6ed9b135d
#)
#FetchContent_MakeAvailable(variant)
