#!/bin/bash

# Simple compilation test script
echo "=== Compilation Test ==="

# Test basic C++ compilation
echo "Testing basic C++ compilation..."
cat > test_basic.cpp << 'EOF'
#include <iostream>
#include <cstring>
int main() {
    char test[10];
    strcpy(test, "hello");
    std::cout << "Basic C++ works: " << test << std::endl;
    return 0;
}
EOF

if g++ -std=c++17 -o test_basic test_basic.cpp; then
    echo "✓ Basic C++ compilation works"
    ./test_basic
    rm -f test_basic test_basic.cpp
else
    echo "✗ Basic C++ compilation failed"
    exit 1
fi

echo ""
echo "Testing OpenMP..."
cat > test_openmp.cpp << 'EOF'
#include <iostream>
#include <omp.h>
int main() {
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            std::cout << "OpenMP works with " << omp_get_num_threads() << " threads" << std::endl;
        }
    }
    return 0;
}
EOF

if g++ -std=c++17 -fopenmp -o test_openmp test_openmp.cpp; then
    echo "✓ OpenMP compilation works"
    ./test_openmp
    rm -f test_openmp test_openmp.cpp
else
    echo "✗ OpenMP compilation failed"
    echo "Try: sudo apt install libomp-dev (Ubuntu) or yum install libgomp-devel (CentOS)"
fi

echo ""
echo "Testing CUDA availability..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(nvcc --version | grep release)"
    
    # Test basic CUDA compilation
    cat > test_cuda.cu << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

__global__ void test_kernel() {
    printf("CUDA kernel works!\n");
}

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess) {
        std::cout << "CUDA devices found: " << deviceCount << std::endl;
        if (deviceCount > 0) {
            test_kernel<<<1,1>>>();
            cudaDeviceSynchronize();
        }
    } else {
        std::cout << "CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
    }
    return 0;
}
EOF

    if nvcc -std=c++17 -o test_cuda test_cuda.cu 2>/dev/null; then
        echo "✓ CUDA compilation works"
        ./test_cuda 2>/dev/null || echo "CUDA runtime not available (expected on login nodes)"
        rm -f test_cuda test_cuda.cu
    else
        echo "✗ CUDA compilation failed (headers missing or version incompatibility)"
        rm -f test_cuda.cu
    fi
else
    echo "- nvcc not found"
fi

echo ""
echo "System information:"
echo "OS: $(uname -a)"
echo "GCC: $(gcc --version 2>/dev/null | head -1 || echo 'Not found')"
echo "CMake: $(cmake --version 2>/dev/null | head -1 || echo 'Not found')"

echo ""
echo "Recommendations:"
echo "1. If basic C++ failed: install build tools"
echo "2. If OpenMP failed: install OpenMP development libraries"
echo "3. If CUDA failed: run ./setup_cuda_env.sh or use --cpu-only"
echo "4. To build: ./build_compatible.sh [--cpu-only]"
