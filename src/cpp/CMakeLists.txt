cmake_minimum_required(VERSION 3.28)
project(gradient_key_engines LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# One executable per source file, listed explicitly (no glob: a stray .cpp must not become an engine).
foreach(engine IN ITEMS gradient_bound gradient_enum gradient_tplus)
  add_executable(${engine} ${engine}.cpp)
endforeach()

# gradient_tplus relies on the upward rounding mode set at run time.
set_source_files_properties(gradient_tplus.cpp PROPERTIES COMPILE_OPTIONS -frounding-math)
