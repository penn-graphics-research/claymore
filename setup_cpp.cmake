function(add_cpp_executable binary)
  add_executable(${binary} ${ARGN})
  target_compile_options(${binary} 
    INTERFACE   "${CMAKE_CXX_FLAGS} -O3 -fPIE -fuse-ld=lld -fvisibility=hidden") # -flto=thin -fsanitize=cfi 
  # -fsanitize=address memory thread
  set_target_properties(${binary}
    PROPERTIES  CXX_STANDARD 20
                LINKER_LANGUAGE CXX
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_features(${binary} INTERFACE cxx_std_20)
  target_link_libraries(${binary}
    PUBLIC      #pthread
                #stdc++
                fmt
  )
  message("-- [${binary}]\tcpp executable build config")
endfunction(add_cpp_executable)

function(add_cpp_executable_debug binary)
  add_executable(${binary} ${ARGN})
  target_compile_options(${binary} 
    INTERFACE   "${CMAKE_CXX_FLAGS} -O2 -fPIE -fuse-ld=lld -fvisibility=hidden"
                "-fsanitize=address -fsanitize=memory -fsanitize=thread -fno-sanitize-trap=cfi") # -flto=thin -fsanitize=cfi
  # -fsanitize=address memory thread
  set_target_properties(${binary}
    PROPERTIES  CXX_STANDARD 20
                LINKER_LANGUAGE CXX
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_features(${binary} INTERFACE cxx_std_20)
  target_link_libraries(${binary}
    PUBLIC      #pthread
                #stdc++
                fmt
  )
  message("-- [${binary}]\tcpp debug executable build config")
endfunction(add_cpp_executable_debug)
