#!/bin/bash

echo "=== Testing CUDA Struct Sizes ==="

# Compile with real CUDA headers
echo "Compiling with nvcc..."
if nvcc -o struct_test struct_test.cpp; then
    echo "Running struct test..."
    ./struct_test
    rm -f struct_test
else
    echo "nvcc compilation failed"
fi

echo ""
echo "The issue is likely that cudaDeviceProp struct in CUDA headers"
echo "is much larger than our simplified version, causing memory corruption."
echo ""
echo "Fix: Use actual CUDA headers or create CPU-only simplified build."
