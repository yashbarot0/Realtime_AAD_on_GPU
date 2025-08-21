cmake_minimum_required(VERSION 3.18)
project(GPU_AAD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to disable CUDA and use CPU-only version
option(USE_CUDA "Use CUDA for GPU acceleration" ON)
option(CPU_ONLY "Force CPU-only build" OFF)

# Find required packages
find_package(OpenMP REQUIRED)

# Check for CUDA availability
if(USE_CUDA AND NOT CPU_ONLY)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        message(STATUS "CUDA found - GPU acceleration enabled")
        
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # Check CUDA and GCC compatibility
        if(CUDAToolkit_VERSION VERSION_LESS "11.0")
            # For CUDA < 11.0, check GCC compatibility
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "8.0")
                    message(WARNING "CUDA ${CUDAToolkit_VERSION} may not support GCC ${CMAKE_CXX_COMPILER_VERSION}")
                    message(STATUS "Attempting to use compatibility flags...")
                    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
                endif()
            endif()
        endif()
        
        set(CUDA_AVAILABLE TRUE)
    else()
        message(STATUS "CUDA not found - using CPU-only version")
        set(CUDA_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "CUDA disabled - using CPU-only version")
    set(CUDA_AVAILABLE FALSE)
endif()

# Set compile definitions
if(NOT CUDA_AVAILABLE OR CPU_ONLY)
    add_definitions(-DCPU_ONLY)
endif()

# Source files
set(SOURCES
    main.cpp
    RealTimePortfolioEngine.cpp
)

if(CUDA_AVAILABLE AND NOT CPU_ONLY)
    # CUDA build
    set(CUDA_SOURCES
        cuda_kernels.cu
    )
    
    # Set CUDA architecture (try to detect or use common ones)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;70;75")
    endif()
    
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
    
    # Set CUDA flags with compatibility
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            ${CMAKE_CUDA_FLAGS}
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    
    target_compile_options(portfolio_demo PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            ${CMAKE_CUDA_FLAGS}
        >
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
else()
    # CPU-only build
    add_executable(${PROJECT_NAME} ${SOURCES})
    add_executable(portfolio_demo
        portfolio_demo.cpp
        RealTimePortfolioEngine.cpp
    )
    
    # Create a CPU implementation source file
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels_cpu.cpp")
        file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels_cpu.cpp"
            "#include \"cpu_fallback.h\"\n"
            "// CPU-only implementation - all functions defined in cpu_fallback.h\n"
        )
    endif()
    target_sources(${PROJECT_NAME} PRIVATE cuda_kernels_cpu.cpp)
    target_sources(portfolio_demo PRIVATE cuda_kernels_cpu.cpp)
endif()

# Link OpenMP for all targets
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(portfolio_demo OpenMP::OpenMP_CXX)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(portfolio_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiler-specific options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_compile_options(portfolio_demo PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# User-space installation support
if(DEFINED CMAKE_INSTALL_PREFIX AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    install(TARGETS ${PROJECT_NAME} portfolio_demo
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()

# Print build configuration
message(STATUS "Build configuration:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CUDA_AVAILABLE AND NOT CPU_ONLY)
    message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  CPU Only Mode: ${CPU_ONLY}")
message(STATUS "  CUDA Available: ${CUDA_AVAILABLE}")
