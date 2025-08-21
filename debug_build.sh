#!/bin/bash

# Simple GPU test to debug the issue
echo "=== GPU Test Debug ==="

# Clean build first
rm -rf build
mkdir -p build
cd build

echo "Building with debug flags..."
if cmake -DUSE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CUDA_ARCHITECTURES="75" \
         -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -g" \
         -DCMAKE_CXX_FLAGS="-g -fsanitize=address" \
         .. && make -j4; then
    
    echo "✓ Debug build successful!"
    echo ""
    echo "Testing with AddressSanitizer..."
    echo ""
    
    # Test basic GPU info only
    echo "=== GPU Info Test ==="
    timeout 10s ./GPU_AAD || echo "GPU_AAD test failed or timed out"
    
    echo ""
    echo "=== Portfolio Demo Test ==="
    timeout 10s ./portfolio_demo || echo "portfolio_demo test failed or timed out"
    
else
    echo "✗ Debug build failed"
    echo "Falling back to CPU-only debug build..."
    
    rm -rf *
    if cmake -DCPU_ONLY=ON \
             -DCMAKE_BUILD_TYPE=Debug \
             -DCMAKE_CXX_FLAGS="-g -fsanitize=address" \
             .. && make -j4; then
        
        echo "✓ CPU-only debug build successful!"
        echo ""
        echo "Testing CPU-only mode..."
        timeout 10s ./GPU_AAD || echo "CPU test failed"
        timeout 10s ./portfolio_demo || echo "Portfolio demo failed"
    else
        echo "✗ Even CPU-only build failed"
    fi
fi
