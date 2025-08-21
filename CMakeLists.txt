cmake_minimum_required(VERSION 3.18)
project(GPU_AAD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture for RTX 2080 (Turing - Compute Capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES "75")

# Source files
set(SOURCES
    main.cpp
    RealTimePortfolioEngine.cpp
)

set(CUDA_SOURCES
    cuda_kernels.cu
)

# Create main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Create portfolio demo executable
add_executable(portfolio_demo
    portfolio_demo.cpp
    RealTimePortfolioEngine.cpp
    ${CUDA_SOURCES}
)

# Set properties for CUDA files
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET portfolio_demo PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(portfolio_demo CUDA::cudart)

# Compiler flags optimized for RTX 2080
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_75>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_compile_options(portfolio_demo PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_75>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(portfolio_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# User-space installation support
if(DEFINED CMAKE_INSTALL_PREFIX AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    install(TARGETS ${PROJECT_NAME} portfolio_demo
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()
