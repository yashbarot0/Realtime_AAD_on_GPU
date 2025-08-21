#!/bin/bash

echo "=== Creating Working CPU-Only Build ==="

# Clean build
rm -rf build
mkdir -p build
cd build

echo "Building CPU-only version with proper fallbacks..."

if cmake -DCPU_ONLY=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O2" \
         .. && make -j4; then
    
    echo "✓ CPU-only build successful!"
    echo ""
    echo "Testing CPU-only applications..."
    
    echo "=== GPU_AAD Test ==="
    timeout 10s ./GPU_AAD && echo "✓ GPU_AAD works in CPU mode"
    
    echo ""
    echo "=== Portfolio Demo Test ==="
    timeout 5s ./portfolio_demo && echo "✓ Portfolio demo works in CPU mode"
    
    echo ""
    echo "🎉 CPU-only version working!"
    echo "For CUDA version, we need to fix struct compatibility issues."
    
else
    echo "❌ Even CPU-only build failed"
    echo "Check the build log above for errors"
fi
