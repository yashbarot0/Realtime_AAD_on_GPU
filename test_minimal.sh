#!/bin/bash

echo "=== Minimal CUDA Test ==="

# Test with direct nvcc compilation
echo "Compiling minimal test with nvcc..."
if nvcc -std=c++17 -arch=sm_75 -o minimal_test minimal_test.cpp; then
    echo "✓ Compilation successful"
    
    echo "Running minimal test..."
    ./minimal_test
    
    echo "Exit code: $?"
    rm -f minimal_test
else
    echo "✗ Compilation failed"
    
    echo "Trying with g++ (CPU-only)..."
    if g++ -std=c++17 -o minimal_test_cpu minimal_test.cpp; then
        echo "✓ CPU compilation successful"
        ./minimal_test_cpu
        rm -f minimal_test_cpu
    else
        echo "✗ CPU compilation also failed"
    fi
fi

echo ""
echo "If the minimal test works, the issue is in the complex AAD code."
echo "If it fails, the issue is with CUDA setup or struct definitions."
