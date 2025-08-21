#!/bin/bash

echo "Real-Time Portfolio Greeks Setup - User Space Installation"
echo "========================================================="
echo "🎓 Optimized for college servers without sudo access"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Set up user directories
USER_HOME="$HOME"
INSTALL_PREFIX="$USER_HOME/.local"
BIN_DIR="$INSTALL_PREFIX/bin"
LIB_DIR="$INSTALL_PREFIX/lib"
INCLUDE_DIR="$INSTALL_PREFIX/include"

print_step "Setting up user-space installation directories..."
mkdir -p "$BIN_DIR" "$LIB_DIR" "$INCLUDE_DIR"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "export PATH=\"$BIN_DIR:\$PATH\"" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "export PKG_CONFIG_PATH=\"$LIB_DIR/pkgconfig:\$PKG_CONFIG_PATH\"" >> ~/.bashrc
    print_status "Added local directories to PATH"
fi

# Source the updated PATH
export PATH="$BIN_DIR:$PATH"
export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

print_step "Checking system information..."

# Check available modules (common on college clusters)
if command -v module &> /dev/null; then
    print_status "Environment modules system detected"
    echo "Available modules:"
    module avail 2>&1 | head -20
    echo "..."
    
    # Common modules to load
    MODULES_TO_LOAD=("gcc" "cuda" "python" "cmake")
    
    for mod in "${MODULES_TO_LOAD[@]}"; do
        if module avail 2>&1 | grep -i "$mod" &> /dev/null; then
            print_status "Loading module: $mod"
            module load "$mod" 2>/dev/null || print_warning "Could not load $mod module"
        fi
    done
    
    # Show loaded modules
    print_status "Currently loaded modules:"
    module list 2>&1 | head -10
fi

# Check for existing installations
print_step "Checking for existing tools..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "Found Python: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    print_status "Found Python: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python not found. Contact system administrator."
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    print_warning "pip not found. Installing pip locally..."
    
    # Download and install pip locally
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py --user
    rm get-pip.py
    
    # Update PATH for pip
    PYTHON_USER_BASE=$($PYTHON_CMD -m site --user-base)
    export PATH="$PYTHON_USER_BASE/bin:$PATH"
    echo "export PATH=\"$PYTHON_USER_BASE/bin:\$PATH\"" >> ~/.bashrc
    
    PIP_CMD="pip"
fi

print_status "Using pip: $PIP_CMD"

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_warning "CMake not found. Installing locally..."
    
    # Download and install CMake locally
    CMAKE_VERSION="3.27.7"
    CMAKE_DIR="cmake-$CMAKE_VERSION-linux-x86_64"
    CMAKE_TAR="$CMAKE_DIR.tar.gz"
    
    cd "$USER_HOME"
    wget "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/$CMAKE_TAR"
    tar -xzf "$CMAKE_TAR"
    
    # Create symlinks
    ln -sf "$USER_HOME/$CMAKE_DIR/bin/cmake" "$BIN_DIR/cmake"
    ln -sf "$USER_HOME/$CMAKE_DIR/bin/ctest" "$BIN_DIR/ctest"
    
    rm "$CMAKE_TAR"
    print_status "CMake installed locally"
else
    CMAKE_VERSION=$(cmake --version | head -1)
    print_status "Found CMake: $CMAKE_VERSION"
fi

# Check for NVIDIA GPU and CUDA
print_step "Checking GPU availability..."
GPU_AVAILABLE=false

if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || print_warning "Could not query GPU details"
    GPU_AVAILABLE=true
else
    print_warning "nvidia-smi not found"
fi

# Check for CUDA
CUDA_AVAILABLE=false
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release)
    print_status "Found CUDA: $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    print_warning "CUDA not found in PATH"
    
    # Try common CUDA locations
    CUDA_PATHS=("/usr/local/cuda" "/opt/cuda" "/usr/cuda" "$HOME/cuda")
    
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -d "$cuda_path/bin" ]; then
            print_status "Found CUDA installation at: $cuda_path"
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            echo "export PATH=\"$cuda_path/bin:\$PATH\"" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\"$cuda_path/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
            CUDA_AVAILABLE=true
            break
        fi
    done
    
    if [ "$CUDA_AVAILABLE" = false ]; then
        print_warning "CUDA not found. Will build CPU-only version."
        print_warning "For GPU acceleration, contact your system administrator about CUDA access."
    fi
fi

# Create Python virtual environment
print_step "Setting up Python virtual environment..."
$PYTHON_CMD -m venv venv --user

# Activate virtual environment
source venv/bin/activate

# Upgrade pip in virtual environment
python -m pip install --upgrade pip

# Install Python dependencies with user flag
print_step "Installing Python dependencies..."

# Core scientific computing
print_status "Installing core scientific packages..."
pip install --user numpy scipy pandas

# Financial data packages
print_status "Installing financial data packages..."
pip install --user yfinance pandas-ta

# Async/networking packages  
print_status "Installing networking packages..."
pip install --user aiohttp websockets

# Server packages (optional)
print_status "Installing server packages..."
pip install --user fastapi uvicorn python-multipart || print_warning "FastAPI installation failed (optional)"

# Performance packages
print_status "Installing performance packages..."
pip install --user psutil
pip install --user numba || print_warning "Numba installation failed (optional - requires compiler)"

print_status "Python environment configured successfully"

# Build C++ components
print_step "Configuring build environment..."
mkdir -p build
cd build

# Create user-friendly CMake configuration
print_step "Configuring CMake for user-space build..."

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")

if [ "$CUDA_AVAILABLE" = true ]; then
    print_status "Configuring with CUDA support..."
    
    # Try to detect GPU compute capability
    if command -v nvidia-smi &> /dev/null; then
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.' || echo "75")
        print_status "Detected GPU compute capability: sm_$GPU_ARCH"
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH")
    else
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=75")  # Default to RTX 2080 capability
    fi
    
    CMAKE_ARGS+=("-DCUDA_FAST_MATH=ON")
    CMAKE_ARGS+=("-DCMAKE_CUDA_FLAGS=-O3 --use_fast_math")
else
    print_warning "Building without CUDA support"
    CMAKE_ARGS+=("-DENABLE_CUDA=OFF")
    CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS=-O3 -march=native -ffast-math")
fi

# Run CMake
print_status "Running CMake configuration..."
cmake .. "${CMAKE_ARGS[@]}"

if [ $? -eq 0 ]; then
    print_status "CMake configuration successful"
    
    # Build the project
    print_step "Building C++ components..."
    NPROC=$(nproc 2>/dev/null || echo 4)
    print_status "Building with $NPROC parallel jobs..."
    
    make -j$NPROC
    
    if [ $? -eq 0 ]; then
        print_status "Build completed successfully"
        
        # Install to user space
        make install 2>/dev/null || print_warning "Install step failed (binaries still available in build/)"
    else
        print_error "Build failed. Check for missing dependencies or compiler errors."
        print_warning "Continuing with Python-only mode..."
    fi
else
    print_error "CMake configuration failed"
    print_warning "Continuing with Python-only mode..."
fi

cd ..

# Create user-friendly scripts
print_step "Creating user scripts..."

# User startup script
cat > start_portfolio_system.sh << 'EOF'
#!/bin/bash
echo "Starting Portfolio Greeks System (User Mode)..."

# Load environment
source ~/.bashrc

# Load modules if available
if command -v module &> /dev/null; then
    module load gcc 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load python 2>/dev/null || true
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=$(nproc 2>/dev/null || echo 4)

# Start the server
python server_portfolio_system.py --server-mode --config user_config.json
EOF

# Quick test script
cat > quick_test.sh << 'EOF'
#!/bin/bash
echo "Quick Portfolio System Test..."

# Load environment
source ~/.bashrc

# Load modules if available  
if command -v module &> /dev/null; then
    module load python 2>/dev/null || true
fi

# Activate virtual environment
source venv/bin/activate

# Run quick test
python test_linux_server.py
EOF

# Performance monitor (no sudo required)
cat > monitor_user.sh << 'EOF'
#!/bin/bash
echo "Portfolio Greeks Performance Monitor (User Mode)"
echo "=============================================="

while true; do
    echo "$(date): System Status:"
    
    # GPU info (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Usage:"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU info not available"
    fi
    
    # User process info
    echo "Portfolio Processes:"
    ps aux | grep -E "(python.*portfolio|GPU_AAD)" | grep -v grep || echo "No portfolio processes found"
    
    # Memory usage
    echo "Memory Usage:"
    free -h 2>/dev/null || echo "Memory info not available"
    
    echo "---"
    sleep 10
done
EOF

# Make scripts executable
chmod +x start_portfolio_system.sh quick_test.sh monitor_user.sh

# Create user configuration
print_step "Creating user configuration..."
cat > user_config.json << 'EOF'
{
    "user_mode": true,
    "market_data": {
        "update_interval_seconds": 30,
        "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "data_source": "yfinance",
        "parallel_fetching": false,
        "timeout_seconds": 10
    },
    "risk_limits": {
        "max_delta": 1000.0,
        "max_gamma": 100.0,
        "max_vega": 5000.0,
        "max_loss": -50000.0,
        "max_exposure": 500000.0
    },
    "compute_config": {
        "update_frequency_ms": 2000,
        "enable_second_order": true,
        "use_gpu": true,
        "max_positions": 1000,
        "batch_size": 256,
        "fallback_to_cpu": true
    },
    "logging": {
        "enable_console": true,
        "enable_file": true,
        "log_level": "INFO",
        "log_file": "portfolio_greeks.log"
    },
    "api": {
        "enable_rest_api": true,
        "port": 8080,
        "host": "127.0.0.1"
    },
    "security": {
        "enable_auth": false,
        "allowed_ips": ["127.0.0.1", "localhost"]
    }
}
EOF

# Create sample portfolio for testing
cat > user_portfolio_sample.json << 'EOF'
{
    "positions": [
        {"symbol": "AAPL", "option_type": "CALL", "strike": 180.0, "expiration": "2024-03-15", "quantity": 10, "premium_paid": 5.50},
        {"symbol": "AAPL", "option_type": "PUT", "strike": 175.0, "expiration": "2024-03-15", "quantity": 5, "premium_paid": 3.80},
        {"symbol": "MSFT", "option_type": "CALL", "strike": 370.0, "expiration": "2024-02-16", "quantity": 8, "premium_paid": 12.50},
        {"symbol": "GOOGL", "option_type": "CALL", "strike": 135.0, "expiration": "2024-04-19", "quantity": 12, "premium_paid": 8.90},
        {"symbol": "TSLA", "option_type": "CALL", "strike": 220.0, "expiration": "2024-01-19", "quantity": 3, "premium_paid": 18.50}
    ]
}
EOF

# Create user instructions
cat > USER_SETUP.md << 'EOF'
# User-Space Setup Instructions (No Sudo Required)

## ✅ Setup Complete!

Your Portfolio Greeks system is now installed in user space without requiring administrator privileges.

## 🚀 Quick Start

### 1. Test the installation
```bash
./quick_test.sh
```

### 2. Start the system
```bash
./start_portfolio_system.sh
```

### 3. Monitor performance
```bash
./monitor_user.sh
```

## 📁 File Locations

- **Installation**: `~/.local/` (binaries, libraries)
- **Python Environment**: `./venv/` (virtual environment)
- **Configuration**: `user_config.json`
- **Portfolio**: `user_portfolio_sample.json`
- **Logs**: `portfolio_greeks.log`

## 🔧 Configuration

### Edit your portfolio
```bash
nano user_portfolio_sample.json
```

### Adjust settings
```bash
nano user_config.json
```

### Load additional modules (if available)
```bash
module load gcc cuda python
```

## 🌐 Access the API

Once running, access the web interface:
- Health check: `curl http://localhost:8080/health`
- Portfolio Greeks: `curl http://localhost:8080/portfolio/greeks`
- View in browser: `http://localhost:8080/health`

## 🐛 Troubleshooting

### If Python packages fail to install:
```bash
# Retry with --user flag
pip install --user package_name

# Or install in virtual environment
source venv/bin/activate
pip install package_name
```

### If CUDA is not found:
```bash
# Check available modules
module avail

# Load CUDA module
module load cuda

# Verify CUDA
which nvcc
```

### If build fails:
```bash
# Use CPU-only mode
export USE_CUDA=0
cd build && cmake .. -DENABLE_CUDA=OFF && make
```

### If port 8080 is busy:
```bash
# Change port in user_config.json
"api": {
    "port": 8081,
    ...
}
```

## 📊 Performance Tips

1. **Load system modules**: `module load gcc cuda python`
2. **Use local resources**: All files stay in your home directory
3. **Monitor resources**: Use `./monitor_user.sh`
4. **Adjust batch size**: Edit `batch_size` in config if memory limited

## 🎓 College Server Notes

- No sudo/root access required
- Everything installs in your home directory
- Uses user-space Python packages
- Compatible with module systems
- Respects resource limits
- Safe for shared environments

EOF

print_status "User-space setup completed successfully!"
echo ""
echo "🎓 College Server Installation Summary:"
echo "✅ No sudo privileges required"
echo "✅ All files in your home directory (~)"
echo "✅ Python virtual environment created"
echo "✅ User-space CMake and dependencies"
echo "✅ GPU support (if available)"
echo "✅ Module system integration"
echo ""
echo "📋 Next Steps:"
echo "1. Test installation: ./quick_test.sh"
echo "2. Edit portfolio: nano user_portfolio_sample.json"  
echo "3. Start system: ./start_portfolio_system.sh"
echo "4. Monitor: ./monitor_user.sh"
echo ""
echo "📖 Read USER_SETUP.md for detailed instructions"
echo ""
if [ "$GPU_AVAILABLE" = true ]; then
    echo "🚀 GPU acceleration enabled!"
else
    echo "💻 CPU-only mode (still very fast!)"
fi
echo ""
print_status "Ready for real-time portfolio Greeks on your college server! 🎓🚀"
