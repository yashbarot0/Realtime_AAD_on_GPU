#!/bin/bash

# Enhanced build script for college server compatibility
echo "=== GPU AAD Portfolio Build Script ==="
echo "Attempting to build with CUDA compatibility fixes..."

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean previous build
print_status "Cleaning previous build..."
rm -rf build
mkdir -p build
cd build

# Check system info
print_status "System Information:"
echo "  GCC Version: $(gcc --version | head -n1)"
echo "  CMake Version: $(cmake --version | head -n1)"

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "  NVCC Version: $(nvcc --version | grep release)"
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "  CUDA Version: $CUDA_VERSION"
    
    # Check CUDA-GCC compatibility
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    echo "  GCC Major Version: $GCC_VERSION"
    
    if (( $(echo "$CUDA_VERSION < 11.0" | bc -l) )) && (( GCC_VERSION > 8 )); then
        print_warning "CUDA $CUDA_VERSION may not support GCC $GCC_VERSION"
        print_warning "Will attempt compatibility build..."
        FORCE_COMPAT=true
    fi
else
    print_warning "NVCC not found - will build CPU-only version"
fi

# Build strategy 1: Try CUDA with compatibility flags
if [[ "$FORCE_COMPAT" == "true" ]] || [[ -n "$1" && "$1" == "--force-compat" ]]; then
    print_status "Strategy 1: CUDA build with compatibility flags..."
    
    if cmake -DUSE_CUDA=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
             .. && make -j$(nproc); then
        print_status "✓ CUDA build with compatibility flags succeeded!"
        echo ""
        echo "Build completed successfully!"
        echo "Executables created:"
        echo "  - GPU_AAD (main application)"
        echo "  - portfolio_demo (portfolio demonstration)"
        echo ""
        echo "To run:"
        echo "  ./GPU_AAD"
        echo "  ./portfolio_demo"
        exit 0
    else
        print_error "CUDA build with compatibility flags failed"
    fi
fi

# Build strategy 2: Try standard CUDA build
if [[ "$FORCE_COMPAT" != "true" ]]; then
    print_status "Strategy 2: Standard CUDA build..."
    
    rm -rf *
    if cmake -DUSE_CUDA=ON \
             -DCMAKE_BUILD_TYPE=Release \
             .. && make -j$(nproc); then
        print_status "✓ Standard CUDA build succeeded!"
        echo ""
        echo "Build completed successfully!"
        echo "Executables created:"
        echo "  - GPU_AAD (main application)"
        echo "  - portfolio_demo (portfolio demonstration)"
        echo ""
        echo "To run:"
        echo "  ./GPU_AAD"
        echo "  ./portfolio_demo"
        exit 0
    else
        print_error "Standard CUDA build failed"
    fi
fi

# Build strategy 3: CPU-only fallback
print_status "Strategy 3: CPU-only fallback build..."

rm -rf *
if cmake -DCPU_ONLY=ON \
         -DCMAKE_BUILD_TYPE=Release \
         .. && make -j$(nproc); then
    print_status "✓ CPU-only build succeeded!"
    print_warning "Note: Running in CPU-only mode (no GPU acceleration)"
    echo ""
    echo "Build completed successfully!"
    echo "Executables created:"
    echo "  - GPU_AAD (main application, CPU-only)"
    echo "  - portfolio_demo (portfolio demonstration, CPU-only)"
    echo ""
    echo "To run:"
    echo "  ./GPU_AAD"
    echo "  ./portfolio_demo"
    echo ""
    echo "Performance note: CPU-only mode uses OpenMP for parallelization"
    echo "but will be slower than GPU acceleration."
    exit 0
else
    print_error "CPU-only build failed"
fi

# Build strategy 4: Minimal fallback
print_status "Strategy 4: Minimal fallback build..."

rm -rf *
if cmake -DUSE_CUDA=OFF \
         -DCPU_ONLY=ON \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-std=c++11" \
         .. && make; then
    print_status "✓ Minimal fallback build succeeded!"
    print_warning "Note: Running in minimal mode with reduced features"
    echo ""
    echo "Build completed successfully!"
    echo "Executables created:"
    echo "  - GPU_AAD (minimal version)"
    echo "  - portfolio_demo (minimal version)"
    echo ""
    echo "To run:"
    echo "  ./GPU_AAD"
    echo "  ./portfolio_demo"
    exit 0
else
    print_error "Minimal fallback build failed"
fi

# If we get here, all strategies failed
print_error "All build strategies failed!"
echo ""
echo "Troubleshooting suggestions:"
echo "1. Check that you have a C++ compiler installed:"
echo "   gcc --version"
echo ""
echo "2. Check CMake version (need >= 3.18):"
echo "   cmake --version"
echo ""
echo "3. Try installing missing dependencies:"
echo "   # On Ubuntu/Debian:"
echo "   sudo apt update && sudo apt install build-essential cmake"
echo "   # On CentOS/RHEL:"
echo "   sudo yum groupinstall 'Development Tools' && sudo yum install cmake3"
echo ""
echo "4. For CUDA issues, check if CUDA is properly installed:"
echo "   nvcc --version"
echo "   nvidia-smi"
echo ""
echo "5. If still failing, try manual build:"
echo "   cd build"
echo "   cmake -DCPU_ONLY=ON .."
echo "   make VERBOSE=1"

exit 1
