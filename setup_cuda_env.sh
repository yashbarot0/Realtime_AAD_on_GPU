#!/bin/bash

# CUDA Environment Detection and Setup Script
echo "=== CUDA Environment Detection ==="

# Function to check if a path exists and is readable
check_path() {
    if [[ -d "$1" && -r "$1" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to find CUDA installation
find_cuda() {
    echo "Searching for CUDA installation..."
    
    # Common CUDA installation paths
    CUDA_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.9"
        "/usr/local/cuda-12"
        "/opt/cuda"
        "/usr/cuda"
        "$HOME/cuda"
        "/sw/cuda"
        "/apps/cuda"
    )
    
    # Also check environment variables
    if [[ -n "$CUDA_HOME" ]]; then
        CUDA_PATHS=("$CUDA_HOME" "${CUDA_PATHS[@]}")
    fi
    
    if [[ -n "$CUDA_PATH" ]]; then
        CUDA_PATHS=("$CUDA_PATH" "${CUDA_PATHS[@]}")
    fi
    
    # Try to find nvcc and derive CUDA path from it
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        NVCC_DIR=$(dirname "$NVCC_PATH")
        CUDA_FROM_NVCC=$(dirname "$NVCC_DIR")
        CUDA_PATHS=("$CUDA_FROM_NVCC" "${CUDA_PATHS[@]}")
        echo "Found nvcc at: $NVCC_PATH"
        echo "Derived CUDA path: $CUDA_FROM_NVCC"
    fi
    
    # Check each path
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if check_path "$cuda_path/include" && check_path "$cuda_path/lib64"; then
            echo "✓ Found CUDA installation at: $cuda_path"
            echo "  Include path: $cuda_path/include"
            echo "  Library path: $cuda_path/lib64"
            
            # Check for key headers
            if [[ -f "$cuda_path/include/cuda_runtime.h" ]]; then
                echo "  ✓ cuda_runtime.h found"
            else
                echo "  ✗ cuda_runtime.h missing"
                continue
            fi
            
            if [[ -f "$cuda_path/include/device_launch_parameters.h" ]]; then
                echo "  ✓ device_launch_parameters.h found"
            else
                echo "  ✗ device_launch_parameters.h missing"
                continue
            fi
            
            # Export environment variables
            export CUDA_HOME="$cuda_path"
            export CUDA_PATH="$cuda_path"
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:${LD_LIBRARY_PATH}"
            export CPATH="$cuda_path/include:${CPATH}"
            export LIBRARY_PATH="$cuda_path/lib64:${LIBRARY_PATH}"
            
            echo "Environment variables set:"
            echo "  CUDA_HOME=$CUDA_HOME"
            echo "  CUDA_PATH=$CUDA_PATH"
            echo "  Updated PATH, LD_LIBRARY_PATH, CPATH, LIBRARY_PATH"
            
            return 0
        fi
    done
    
    echo "✗ No valid CUDA installation found"
    return 1
}

# Function to check module system (common in HPC environments)
check_modules() {
    if command -v module &> /dev/null; then
        echo ""
        echo "=== Module System Detected ==="
        echo "Available CUDA modules:"
        module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found"
        
        echo ""
        echo "To load CUDA module, try one of:"
        echo "  module load cuda"
        echo "  module load cuda/12.9"
        echo "  module load CUDA"
        echo ""
        echo "Currently loaded modules:"
        module list 2>&1 | head -10
        
        # Try to auto-load a CUDA module
        if module avail cuda 2>&1 | grep -q "cuda/"; then
            CUDA_MODULE=$(module avail cuda 2>&1 | grep "cuda/" | head -1 | awk '{print $1}')
            echo "Attempting to load: $CUDA_MODULE"
            if module load "$CUDA_MODULE" 2>/dev/null; then
                echo "✓ Successfully loaded $CUDA_MODULE"
                return 0
            fi
        fi
    fi
    return 1
}

# Main execution
echo "CUDA Compiler Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH"
fi

echo ""
echo "Current environment:"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  CUDA_PATH: ${CUDA_PATH:-not set}"
echo "  PATH: $(echo $PATH | tr ':' '\n' | grep -i cuda | head -3)"

echo ""
check_modules

echo ""
if find_cuda; then
    echo ""
    echo "=== CUDA Setup Complete ==="
    echo "You can now run the build script:"
    echo "  ./build_compatible.sh"
    echo ""
    echo "Or set these environment variables in your shell:"
    echo "  export CUDA_HOME=$CUDA_HOME"
    echo "  export CUDA_PATH=$CUDA_PATH"
    echo "  export PATH=$CUDA_PATH/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
    echo "  export CPATH=$CUDA_PATH/include:\$CPATH"
else
    echo ""
    echo "=== CUDA Not Found - CPU Fallback Mode ==="
    echo "Will build in CPU-only mode. Run:"
    echo "  ./build_compatible.sh"
    echo ""
    echo "Or try manually setting CUDA paths if you know where it's installed:"
    echo "  export CUDA_HOME=/path/to/cuda"
    echo "  ./setup_cuda_env.sh"
fi
