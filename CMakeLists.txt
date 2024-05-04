cmake_minimum_required(VERSION 3.11)
project(cs_5220_final_project)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(OpenMP REQUIRED)

if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
endif ()

# Suppress specific warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Suppress warning for unused parameters
    add_compile_options(-diag-suppress 2464)
endif()

add_executable(serial main.cu serial.cu common.h util.h)
add_executable(gpu main.cu parallel.cu common.h util.h)
add_executable(gpu_iterative main.cu parallel_iterative.cu common.h util.h)
target_link_libraries(gpu_iterative PRIVATE OpenMP::OpenMP_CXX)

target_compile_features(gpu PRIVATE cxx_std_11)
target_compile_features(serial PRIVATE cxx_std_11)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 -extended-lambda -fopenmp")

add_compile_definitions(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)

# Copy the job scripts
configure_file(job-gpu job-gpu COPYONLY)
configure_file(job-serial job-serial COPYONLY)
configure_file(job-gpu-iterative job-gpu-iterative COPYONLY)
