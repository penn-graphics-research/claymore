find_package(Git)
include(FetchContent)
include(ExternalProject)
include(DownloadProject)

# partio
message("## building partio")
add_subdirectory(partio)

# fmt
message("## building fmt")
#add_subdirectory(fmt-6.1.2)
#target_include_directories(fmt SYSTEM INTERFACE fmt-6.1.2/include)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 19bd751020a1f3c3363b2eb67a039852f139a8d3
)
FetchContent_MakeAvailable(fmt)

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
