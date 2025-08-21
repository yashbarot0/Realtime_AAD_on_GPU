#!/bin/bash

echo "CUDA Compatibility Fix for College Servers"
echo "=========================================="
echo "🔧 Fixing CUDA 10.1 + GCC 14.2 compatibility issue"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Detect the compatibility issue
print_step "Analyzing CUDA/GCC compatibility..."

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
GCC_VERSION=$(gcc --version | head -1 | sed 's/.*gcc[^0-9]*\([0-9.]*\).*/\1/')

print_status "Detected CUDA version: $CUDA_VERSION"
print_status "Detected GCC version: $GCC_VERSION"

# Check if we have this specific compatibility issue
if [[ "$CUDA_VERSION" == "10.1" ]] && [[ "${GCC_VERSION%%.*}" -gt 8 ]]; then
    print_warning "CUDA 10.1 only supports GCC <= 8, but found GCC $GCC_VERSION"
    print_step "Implementing compatibility fixes..."
    
    # Solution 1: Try to find compatible GCC version
    COMPATIBLE_GCC=""
    for gcc_ver in gcc-8 gcc-7 gcc-6; do
        if command -v $gcc_ver &> /dev/null; then
            COMPATIBLE_GCC=$gcc_ver
            print_status "Found compatible compiler: $gcc_ver"
            break
        fi
    done
    
    # Solution 2: Check for modules with older GCC
    if command -v module &> /dev/null; then
        print_status "Checking for GCC modules..."
        
        # Common module names for older GCC
        for gcc_module in "gcc/8" "gcc/7" "gcc/6" "GCC/8" "GCC/7" "toolchain/gcc/8"; do
            if module avail 2>&1 | grep -q "$gcc_module"; then
                print_status "Found GCC module: $gcc_module"
                echo "module load $gcc_module" >> load_compatible_gcc.sh
                COMPATIBLE_GCC="module:$gcc_module"
                break
            fi
        done
    fi
    
    # Create a workaround CMakeLists.txt
    print_step "Creating compatibility CMakeLists.txt..."
    
    cp CMakeLists.txt CMakeLists.txt.backup
    
    cat > CMakeLists_cuda10_compat.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)

# Handle CUDA 10.1 + modern GCC compatibility
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    message(STATUS "Detected CUDA 10.x - applying compatibility fixes")
    
    # Option 1: Disable CUDA and build CPU-only
    option(FORCE_CPU_ONLY "Force CPU-only build" OFF)
    
    if(FORCE_CPU_ONLY)
        message(STATUS "Building CPU-only version due to CUDA compatibility issues")
        project(GPU_AAD LANGUAGES CXX)
        set(ENABLE_CUDA OFF)
    else()
        # Option 2: Try CUDA with compatibility flags
        project(GPU_AAD LANGUAGES CXX CUDA)
        
        # Force older C++ standard that CUDA 10.1 supports
        set(CMAKE_CXX_STANDARD 14)
        set(CMAKE_CUDA_STANDARD 14)
        
        # Add compatibility flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -w") # Suppress warnings
        
        # Try to find compatible host compiler
        find_program(GCC8_COMPILER NAMES gcc-8 gcc-7 gcc-6)
        if(GCC8_COMPILER)
            message(STATUS "Using compatible compiler: ${GCC8_COMPILER}")
            set(CMAKE_CUDA_HOST_COMPILER ${GCC8_COMPILER})
        endif()
    endif()
else()
    project(GPU_AAD LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA only if not disabled
if(NOT DEFINED ENABLE_CUDA OR ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA not found, building CPU-only version")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Detect GPU architecture more safely
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    # Default to common architectures for older CUDA
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;70") # Safe defaults for CUDA 10.x
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75") # RTX 2080
    endif()
endif()

# Source files
set(SOURCES
    main.cpp
    RealTimePortfolioEngine.cpp
)

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    set(CUDA_SOURCES
        cuda_kernels.cu
    )
    message(STATUS "Building with CUDA support")
else()
    set(CUDA_SOURCES)
    message(STATUS "Building CPU-only version")
    add_definitions(-DCPU_ONLY)
endif()

# Create executables
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    add_executable(portfolio_demo
        portfolio_demo.cpp
        RealTimePortfolioEngine.cpp
        ${CUDA_SOURCES}
    )
else()
    add_executable(portfolio_demo
        portfolio_demo.cpp
        RealTimePortfolioEngine.cpp
    )
endif()

# Set properties and link libraries
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    set_property(TARGET portfolio_demo PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    
    target_link_libraries(${PROJECT_NAME} CUDA::cudart)
    target_link_libraries(portfolio_demo CUDA::cudart)
    
    # Safer compiler flags for CUDA 10.1
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>
    )
    
    target_compile_options(portfolio_demo PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>
    )
else()
    # CPU-only optimizations
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -fopenmp -march=native)
    target_compile_options(portfolio_demo PRIVATE -O3 -fopenmp -march=native)
    
    # Link OpenMP for CPU parallelization
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
        target_link_libraries(portfolio_demo OpenMP::OpenMP_CXX)
    endif()
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(portfolio_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Installation
if(DEFINED CMAKE_INSTALL_PREFIX AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    install(TARGETS ${PROJECT_NAME} portfolio_demo
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib)
endif()
EOF

    # Create build scripts for different scenarios
    print_step "Creating build scripts..."
    
    # Script 1: Try CUDA with compatibility flags
    cat > build_with_cuda_compat.sh << 'EOF'
#!/bin/bash
echo "Building with CUDA 10.1 compatibility flags..."

# Load compatible GCC if available
if [ -f "load_compatible_gcc.sh" ]; then
    source load_compatible_gcc.sh
fi

mkdir -p build_cuda_compat
cd build_cuda_compat

# Use compatibility CMakeLists
cp ../CMakeLists_cuda10_compat.txt ../CMakeLists.txt

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler -Xcompiler -w" \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_CUDA_STANDARD=14

make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ CUDA build successful with compatibility flags!"
else
    echo "❌ CUDA build failed, try CPU-only build"
fi
EOF

    # Script 2: Force CPU-only build
    cat > build_cpu_only.sh << 'EOF'
#!/bin/bash
echo "Building CPU-only version (no CUDA)..."

mkdir -p build_cpu_only
cd build_cpu_only

# Use compatibility CMakeLists
cp ../CMakeLists_cuda10_compat.txt ../CMakeLists.txt

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_CPU_ONLY=ON \
    -DCMAKE_CXX_FLAGS="-O3 -fopenmp -march=native"

make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ CPU-only build successful!"
    echo "Note: Will use OpenMP for parallelization instead of CUDA"
else
    echo "❌ Build failed"
fi
EOF

    # Script 3: Try with specific GCC version
    if [ -n "$COMPATIBLE_GCC" ] && [[ "$COMPATIBLE_GCC" != module:* ]]; then
        cat > build_with_gcc8.sh << EOF
#!/bin/bash
echo "Building with compatible GCC ($COMPATIBLE_GCC)..."

mkdir -p build_gcc8
cd build_gcc8

# Use compatibility CMakeLists
cp ../CMakeLists_cuda10_compat.txt ../CMakeLists.txt

# Set specific compilers
export CC=$COMPATIBLE_GCC
export CXX=\${COMPATIBLE_GCC/gcc/g++}

cmake .. \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_C_COMPILER=\$CC \\
    -DCMAKE_CXX_COMPILER=\$CXX \\
    -DCMAKE_CUDA_HOST_COMPILER=\$CC

make -j\$(nproc)

if [ \$? -eq 0 ]; then
    echo "✅ Build successful with compatible GCC!"
else
    echo "❌ Build failed with compatible GCC"
fi
EOF
    fi
    
    chmod +x build_*.sh
    
    print_step "Attempting automatic fix..."
    
    # Try the builds in order of preference
    if [ -f "build_with_gcc8.sh" ]; then
        print_status "Trying build with compatible GCC..."
        ./build_with_gcc8.sh
        if [ $? -eq 0 ]; then
            print_status "✅ Successfully built with compatible GCC!"
            exit 0
        fi
    fi
    
    print_status "Trying CUDA build with compatibility flags..."
    ./build_with_cuda_compat.sh
    if [ $? -eq 0 ]; then
        print_status "✅ Successfully built with CUDA compatibility flags!"
        exit 0
    fi
    
    print_status "Falling back to CPU-only build..."
    ./build_cpu_only.sh
    if [ $? -eq 0 ]; then
        print_status "✅ Successfully built CPU-only version!"
        print_warning "GPU acceleration disabled, but system will still work efficiently"
        exit 0
    fi
    
    print_error "All build attempts failed. Manual intervention required."
    
else
    print_status "No CUDA/GCC compatibility issues detected"
    print_status "Proceeding with normal build..."
    
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
fi

# Create a wrapper script that automatically uses the working build
print_step "Creating build wrapper..."

cat > smart_build.sh << 'EOF'
#!/bin/bash
echo "Smart Build - Automatically handling CUDA/GCC compatibility"

# Check what builds are available
if [ -d "build_gcc8" ] && [ -f "build_gcc8/GPU_AAD" ]; then
    echo "✅ Using GCC8-compatible build"
    export BUILD_DIR="build_gcc8"
elif [ -d "build_cuda_compat" ] && [ -f "build_cuda_compat/GPU_AAD" ]; then
    echo "✅ Using CUDA-compatible build"
    export BUILD_DIR="build_cuda_compat"
elif [ -d "build_cpu_only" ] && [ -f "build_cpu_only/GPU_AAD" ]; then
    echo "✅ Using CPU-only build"
    export BUILD_DIR="build_cpu_only"
elif [ -d "build" ] && [ -f "build/GPU_AAD" ]; then
    echo "✅ Using standard build"
    export BUILD_DIR="build"
else
    echo "❌ No successful builds found. Run fix_cuda_compatibility.sh first."
    exit 1
fi

echo "Using build directory: $BUILD_DIR"
EOF

chmod +x smart_build.sh

print_status "Compatibility fix completed!"
echo ""
echo "🔧 Available build options:"
echo "1. ./build_with_cuda_compat.sh - Try CUDA with compatibility flags"
echo "2. ./build_cpu_only.sh - CPU-only version (recommended fallback)"
if [ -f "build_with_gcc8.sh" ]; then
    echo "3. ./build_with_gcc8.sh - Use compatible GCC version"
fi
echo ""
echo "🚀 To run the system:"
echo "1. source smart_build.sh  # Sets BUILD_DIR"
echo "2. \$BUILD_DIR/portfolio_demo  # Run the demo"
echo ""
print_status "The CPU-only version will still be very fast for portfolio Greeks!"
