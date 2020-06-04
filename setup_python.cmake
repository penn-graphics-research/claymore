function(add_python_library library)
  add_library(${library} MODULE ${ARGN})
  target_compile_options(${library} 
    INTERFACE   "${CMAKE_CXX_FLAGS} -O3 -fuse-ld=lld 
                  -fvisibility=hidden -fsized-deallocation") # -flto=thin -fsanitize=cfi
  set_target_properties(${library}
    PROPERTIES  CPP_STANDARD 14
                PREFIX "${PYTHON_MODULE_PREFIX}"
                SUFFIX "${PYTHON_MODULE_EXTENSION}"
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(${library} 
    INTERFACE   CMAKE_GENERATOR_PLATFORM=x64
  )
  target_compile_features(${library} INTERFACE cxx_std_14)
  target_link_libraries(${library}
    PRIVATE     pybind11::module
  )
  message("-- [${library}]\tpython library build config")
endfunction(add_python_library)
