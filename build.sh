#!/bin/bash

# Build script for RTX 2080 setup
echo "Building GPU AAD for RTX 2080..."

# Clean previous builds
rm -rf build
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo "Build completed!"
echo "Run with: ./GPU_AAD"
