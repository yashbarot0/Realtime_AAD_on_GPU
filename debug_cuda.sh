#!/bin/bash

# CUDA compilation debugging script
echo "=== CUDA Compilation Debug ==="

# Create a simple test file to isolate the issue
cat > cuda_debug.cu << 'EOF'
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Custom atomicAdd for double
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__device__ void safe_atomic_add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    atomicAddDouble(address, val);
#endif
}

__global__ void test_kernel(double* data) {
    int idx = threadIdx.x;
    if (idx < 10) {
        safe_atomic_add(&data[0], 1.0);
    }
}

int main() {
    double *d_data;
    cudaMalloc(&d_data, sizeof(double));
    test_kernel<<<1, 32>>>(d_data);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    return 0;
}
EOF

echo "Testing basic CUDA compilation with atomicAdd..."
if nvcc -std=c++17 -arch=sm_35 -o cuda_debug cuda_debug.cu 2>cuda_error.log; then
    echo "✓ Basic CUDA test compiled successfully"
    rm -f cuda_debug cuda_debug.cu
else
    echo "✗ Basic CUDA test failed"
    echo "Error log:"
    cat cuda_error.log
fi

echo ""
echo "Testing with different compute capabilities..."
for arch in sm_35 sm_50 sm_60 sm_70 sm_75; do
    echo -n "Testing $arch: "
    if nvcc -std=c++17 -arch=$arch -o cuda_debug cuda_debug.cu 2>/dev/null; then
        echo "✓"
    else
        echo "✗"
    fi
done

rm -f cuda_debug cuda_debug.cu cuda_error.log

echo ""
echo "Testing compilation flags..."
echo "Available CUDA architectures on this system:"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "NVCC version details:"
nvcc --version

echo ""
echo "Recommended actions:"
echo "1. Try building with specific architecture: cmake -DCMAKE_CUDA_ARCHITECTURES=\"60;70;75\" .."
echo "2. Or use broader compatibility: cmake -DCMAKE_CUDA_ARCHITECTURES=\"35;50;60;70;75\" .."
echo "3. Check if building with CPU-only works: ./build_compatible.sh --cpu-only"
