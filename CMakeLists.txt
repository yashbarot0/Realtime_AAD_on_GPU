cmake_minimum_required(VERSION 3.18)
project(GPU_AAD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture for RTX 2080 Super (Turing - Compute Capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES "75")

# Source files
set(SOURCES
    src/main.cpp
    src/python_gpu_interface.cpp
)

set(CUDA_SOURCES
    include/cuda_kernels.cu
)

# Create main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Create shared library for Python integration
add_library(gpu_aad_shared SHARED ${SOURCES} ${CUDA_SOURCES})

# Set properties for CUDA files
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET gpu_aad_shared PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(gpu_aad_shared CUDA::cudart)

# Compiler flags optimized for RTX 2080 Super
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_75>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(gpu_aad_shared PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_75>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpu_aad_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set shared library output name
set_target_properties(gpu_aad_shared PROPERTIES OUTPUT_NAME "gpu_aad")
