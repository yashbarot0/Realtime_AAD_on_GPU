#!/bin/bash

# Quick test script for CUDA detection
echo "=== CUDA Detection Test ==="

echo "1. Checking nvcc availability:"
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found at: $(which nvcc)"
    nvcc --version | head -4
else
    echo "✗ nvcc not found"
fi

echo ""
echo "2. Checking CUDA headers:"
CUDA_HEADER_PATHS=(
    "/usr/local/cuda/include/cuda_runtime.h"
    "/usr/local/cuda-12.9/include/cuda_runtime.h"
    "/usr/local/cuda-12/include/cuda_runtime.h"
    "/opt/cuda/include/cuda_runtime.h"
    "$CUDA_HOME/include/cuda_runtime.h"
    "$CUDA_PATH/include/cuda_runtime.h"
)

HEADER_FOUND=false
for header_path in "${CUDA_HEADER_PATHS[@]}"; do
    if [[ -f "$header_path" ]]; then
        echo "✓ Found CUDA headers at: $(dirname "$header_path")"
        HEADER_FOUND=true
        break
    fi
done

if [[ "$HEADER_FOUND" == "false" ]]; then
    echo "✗ CUDA headers not found in standard locations"
    echo "Searching for cuda_runtime.h..."
    find /usr /opt -name "cuda_runtime.h" 2>/dev/null | head -3
fi

echo ""
echo "3. Environment variables:"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "CUDA_PATH: ${CUDA_PATH:-not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"

echo ""
echo "4. Checking for module system:"
if command -v module &> /dev/null; then
    echo "✓ Module system available"
    echo "Checking for CUDA modules..."
    module avail 2>&1 | grep -i cuda | head -3
else
    echo "- No module system found"
fi

echo ""
echo "5. Recommended actions:"
if [[ "$HEADER_FOUND" == "true" ]]; then
    echo "✓ CUDA setup looks good. Try building with:"
    echo "  ./build_compatible.sh"
else
    echo "⚠ CUDA headers not found. Try:"
    echo "  1. Run: ./setup_cuda_env.sh"
    echo "  2. Or load CUDA module: module load cuda"
    echo "  3. Or set manually: export CUDA_HOME=/path/to/cuda"
    echo "  4. Or build CPU-only: ./build_compatible.sh --cpu-only"
fi
