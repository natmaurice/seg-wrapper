cmake_minimum_required(VERSION 3.10)

project(segwrapper VERSION 0.1 LANGUAGES CXX)

set(exec segwrapper)

set(inc_dir .)
set(src_dir .)
set(lib_dir lib)

list(APPEND segwrapper_SRCs
  ${src_dir}/main.cpp
)

# optparse library does not provide CMake files
set(OPTPARSE_INCLUDE_DIRS lib/optparse/)

set(OpenCV_REQUIRED_PACKAGES "core;imgcodecs;imgproc")
FIND_PACKAGE(OpenCV REQUIRED ${OpenCV_REQUIRED_PACKAGES})


add_executable(${exec} ${segwrapper_SRCs})
include_directories(${OPTPARSE_INCLUDE_DIRS})
target_link_libraries(${exec} PRIVATE ${OpenCV_LIBS})
