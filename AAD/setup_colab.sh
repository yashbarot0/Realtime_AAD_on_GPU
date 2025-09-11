#!/bin/bash
# Setup script for Google Colab

echo "Setting up AAD GPU project on Google Colab..."

# Check CUDA installation
echo "Checking CUDA installation..."
nvcc --version
nvidia-smi

# Install build tools
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y build-essential

# Create project structure
echo "Creating project structure..."
mkdir -p include src

# Set environment variables
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo "Environment setup complete!"
echo "CUDA_PATH: $CUDA_PATH"
echo "You can now run: make clean && make -j4"
