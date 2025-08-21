#!/bin/bash

# Quick build script optimized for your system (CUDA 12.9 + sm_75 GPU)
echo "=== Quick Build for CUDA 12.9 + RTX GPU ==="

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean and build
print_status "Cleaning previous build..."
rm -rf build
mkdir -p build
cd build

print_status "Building for your RTX GPU (sm_75) with CUDA 12.9..."

if cmake -DUSE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="75" \
         -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
         .. && make -j$(nproc 2>/dev/null || echo 4); then
    
    print_status "✓ Build successful!"
    echo ""
    echo "🚀 GPU-accelerated portfolio system ready!"
    echo ""
    echo "Executables created:"
    echo "  - GPU_AAD (main application)"
    echo "  - portfolio_demo (real-time portfolio demonstration)"
    echo ""
    echo "To test:"
    echo "  ./GPU_AAD"
    echo "  ./portfolio_demo"
    echo ""
    echo "Your system specs:"
    echo "  - CUDA 12.9"
    echo "  - GPU compute capability: 7.5"
    echo "  - Architecture: sm_75 (RTX 2080/2070 series)"
    
else
    print_error "Build failed!"
    echo ""
    echo "Fallback options:"
    echo "1. Try CPU-only build: ../build_compatible.sh --cpu-only"
    echo "2. Try full compatibility build: ../build_compatible.sh"
    echo "3. Check build log for details"
    
    exit 1
fi
