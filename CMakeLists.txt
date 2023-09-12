cmake_minimum_required(VERSION 3.10)

project(TestOpenCL)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_definitions(-DCL_TARGET_OPENCL_VERSION=200)

add_executable(test-opencl src/matrix-mul.c)
target_link_libraries(test-opencl ${OpenCL_LIBRARIES})

# Copy the OpenCL kernel file to the build directory. This ensures
# that the kernel file is in the same location as the executable.

add_custom_command(TARGET test-opencl POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   -E copy
                   ${CMAKE_SOURCE_DIR}/src/matrix-mul-kernel.cl
                   $<TARGET_FILE_DIR:test-opencl>)
