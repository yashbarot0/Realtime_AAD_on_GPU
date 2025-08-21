#!/bin/bash

# Enhanced build script for college server compatibility
echo "=== GPU AAD Portfolio Build Script ==="

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

# Parse command line arguments
CPU_ONLY_FLAG=false
FORCE_COMPAT=false

for arg in "$@"; do
    case $arg in
        --cpu-only)
            CPU_ONLY_FLAG=true
            shift
            ;;
        --force-compat)
            FORCE_COMPAT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --cpu-only      Force CPU-only build (no CUDA)"
            echo "  --force-compat  Force CUDA compatibility mode"
            echo "  --help          Show this help"
            exit 0
            ;;
    esac
done

# Clean previous build
print_status "Cleaning previous build..."
rm -rf build
mkdir -p build
cd build

# Check system info
print_status "System Information:"
echo "  GCC Version: $(gcc --version 2>/dev/null | head -n1 || echo 'Not found')"
echo "  CMake Version: $(cmake --version 2>/dev/null | head -n1 || echo 'Not found')"

# Check for CUDA
if [[ "$CPU_ONLY_FLAG" == "true" ]]; then
    print_status "CPU-only mode requested - skipping CUDA detection"
else
    if command -v nvcc &> /dev/null; then
        echo "  NVCC Version: $(nvcc --version | grep release)"
        CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "  CUDA Version: $CUDA_VERSION"
        
        # Check for CUDA headers
        if [[ -f "$CUDA_HOME/include/cuda_runtime.h" ]] || [[ -f "/usr/local/cuda/include/cuda_runtime.h" ]]; then
            echo "  CUDA Headers: Found"
        else
            print_warning "CUDA headers not found - may need to run ./setup_cuda_env.sh"
        fi
    else
        print_warning "NVCC not found - will build CPU-only version"
        CPU_ONLY_FLAG=true
    fi
fi

if [[ "$CPU_ONLY_FLAG" == "true" ]]; then
    print_status "Building CPU-only version..."
    
    if cmake -DCPU_ONLY=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-std=c++17" \
             .. && make -j$(nproc 2>/dev/null || echo 4) 2>&1 | tee build.log; then
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
        echo "Build log saved to build/build.log"
        echo ""
        echo "Common issues and solutions:"
        echo "1. Missing C++ compiler: install build-essential"
        echo "2. Missing CMake: install cmake"
        echo "3. Missing OpenMP: install libomp-dev"
        exit 1
    fi
fi

# Build strategy 1: Try standard CUDA build
if [[ "$FORCE_COMPAT" != "true" ]]; then
    print_status "Strategy 1: Standard CUDA build..."
    
    if cmake -DUSE_CUDA=ON \
             -DCMAKE_BUILD_TYPE=Release \
             .. && make -j$(nproc 2>/dev/null || echo 4) 2>&1 | tee build.log; then
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
        echo "Build log saved to build/build.log"
        echo "Trying compatibility mode..."
    fi
fi

# Build strategy 2: Try CUDA with compatibility flags  
print_status "Strategy 2: CUDA build with compatibility flags..."

rm -rf *
if cmake -DUSE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
         -DCMAKE_CUDA_ARCHITECTURES="60;70;75" \
         .. && make -j$(nproc 2>/dev/null || echo 4) 2>&1 | tee build.log; then
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
    echo "Build log saved to build/build.log"
fi

# Build strategy 3: Try CUDA with minimal architecture support
print_status "Strategy 3: CUDA build with minimal architecture..."

rm -rf *
if cmake -DUSE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="60" \
         .. && make -j$(nproc 2>/dev/null || echo 4) 2>&1 | tee build.log; then
    print_status "✓ CUDA build with minimal architecture succeeded!"
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
    print_error "CUDA build with minimal architecture failed"
    echo "Build log saved to build/build.log"
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
