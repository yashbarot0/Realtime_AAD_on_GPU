# 🎓 College Server Deployment Guide

## 📋 **No-Sudo Installation for College Servers**

Perfect for **restricted university environments** where you don't have administrator privileges!

---

## 🚀 **Quick Setup (5 Minutes)**

### **Step 1: Upload to Your College Server**
```bash
# From your local machine
scp -r "/Users/yash/Desktop/ GPU"/* username@college-server:~/portfolio-greeks/

# SSH into your server
ssh username@college-server
cd ~/portfolio-greeks/
```

### **Step 2: Run User-Space Setup**
```bash
# Make setup script executable
chmod +x setup_user_space.sh

# Run automated setup (no sudo required!)
./setup_user_space.sh
```

### **Step 3: Test Installation**
```bash
# Quick system test
./quick_test.sh

# Comprehensive test
python3 test_user_space.py
```

### **Step 4: Start Trading!**
```bash
# Start the portfolio system
./start_portfolio_system.sh
```

---

## 🎯 **What's Different for College Servers**

### ✅ **No Admin Rights Needed**
- Everything installs in your home directory (`~/`)
- Uses Python virtual environments
- Downloads dependencies to `~/.local/`
- No system-wide changes

### ✅ **Module System Integration**
- Automatically detects and loads: `gcc`, `cuda`, `python`, `cmake`
- Works with popular HPC module systems
- Falls back gracefully if modules unavailable

### ✅ **Resource-Aware**
- Respects memory and CPU limits
- Configurable batch sizes
- Efficient for shared environments
- Minimal disk usage

### ✅ **Network-Friendly**
- Uses free Yahoo Finance API
- No firewall issues (port 8080 local only)
- Configurable timeouts
- Handles connection interruptions

---

## 📊 **College Server Optimizations**

### **File Locations**
```
~/portfolio-greeks/           # Main directory
├── ~/.local/bin/            # User binaries (CMake, etc.)
├── ~/.local/lib/            # User libraries  
├── venv/                    # Python virtual environment
├── user_config.json        # Your configuration
├── user_portfolio_sample.json  # Your positions
├── portfolio_greeks.log    # System logs
└── build/                   # Compiled programs
```

### **Resource Usage**
- **Memory**: ~500MB-2GB (depending on portfolio size)
- **Disk**: ~1GB for full installation
- **CPU**: Uses available cores efficiently
- **GPU**: Optional, falls back to CPU gracefully

### **Network Requirements**
- **Outbound HTTPS**: For Yahoo Finance API
- **Local Port**: 8080 (configurable)
- **Bandwidth**: ~10KB/min for market data

---

## 🔧 **Common College Server Scenarios**

### **Scenario 1: Shared Login Nodes**
```bash
# Use lower resource settings
nano user_config.json

# Edit these values:
"compute_config": {
    "batch_size": 128,           # Smaller batches
    "update_frequency_ms": 5000, # Less frequent updates
    "max_positions": 500         # Fewer positions
}
```

### **Scenario 2: GPU Cluster Access**
```bash
# Check for GPU access
nvidia-smi

# If available, load CUDA module
module load cuda

# GPU will be detected automatically
./start_portfolio_system.sh
```

### **Scenario 3: CPU-Only Environment**
```bash
# Force CPU-only mode
export USE_CUDA=0

# Rebuild if needed
cd build
cmake .. -DENABLE_CUDA=OFF
make

# CPU mode is still very fast!
```

### **Scenario 4: Limited Internet**
```bash
# Increase timeouts in config
"market_data": {
    "timeout_seconds": 30,
    "retry_attempts": 3
}
```

---

## 📱 **Monitoring on College Servers**

### **Check System Status**
```bash
# Monitor your processes
./monitor_user.sh

# Check resource usage
ps aux | grep $(whoami)

# View logs
tail -f portfolio_greeks.log
```

### **API Access**
```bash
# Health check
curl http://localhost:8080/health

# Get portfolio Greeks
curl http://localhost:8080/portfolio/greeks

# Via browser (if X11 forwarding enabled)
firefox http://localhost:8080/health
```

### **Performance Tuning**
```bash
# Check CPU usage
top -u $(whoami)

# Check memory usage
free -h

# Check disk usage
du -sh ~/portfolio-greeks/
```

---

## 🚨 **Troubleshooting College Servers**

### **"Permission Denied" Errors**
```bash
# Make sure you're in your home directory
cd ~/portfolio-greeks/

# Fix script permissions
chmod +x *.sh

# Check file ownership
ls -la
```

### **"Module Not Found" Errors**
```bash
# Load required modules
module load python gcc

# Or check available modules
module avail

# Update your .bashrc
echo "module load python gcc" >> ~/.bashrc
```

### **"Port Already in Use"**
```bash
# Change port in config
nano user_config.json

# Find another user's process
lsof -i :8080

# Use different port
"api": {"port": 8081}
```

### **"Out of Memory" Errors**
```bash
# Reduce batch size
nano user_config.json

"compute_config": {
    "batch_size": 64,
    "max_positions": 100
}
```

### **"CUDA Not Found"**
```bash
# Check for CUDA module
module avail | grep -i cuda

# Load CUDA module
module load cuda/11.8

# Verify
which nvcc
```

---

## 🎓 **Best Practices for College Servers**

### **Resource Etiquette**
- Monitor your CPU/memory usage
- Use reasonable update frequencies
- Clean up old log files
- Respect shared resources

### **Security**
- Keep API local (127.0.0.1)
- Don't share API keys in code
- Use secure file permissions
- Protect your portfolio data

### **Performance**
```bash
# Optimize for your environment
./quick_test.sh

# Adjust configuration based on results
nano user_config.json
```

### **Collaboration**
```bash
# Share read-only access
chmod 644 user_portfolio_sample.json

# Demo to classmates
curl http://localhost:8080/portfolio/greeks | python -m json.tool
```

---

## 📚 **Academic Use Cases**

### **Research Projects**
- Options pricing model validation
- Greek hedging strategy backtesting
- High-frequency risk management
- GPU vs CPU performance analysis

### **Class Demonstrations**
- Real-time financial computing
- CUDA programming examples
- REST API development
- Risk management systems

### **Portfolio Management**
- Paper trading portfolios
- Risk analysis projects
- Performance attribution
- Scenario analysis

---

## 🎯 **Example Usage**

### **Add Your Portfolio**
```bash
# Edit the sample portfolio
nano user_portfolio_sample.json

# Add your positions:
{
  "positions": [
    {
      "symbol": "AAPL",
      "option_type": "CALL",
      "strike": 190.0,
      "expiration": "2024-03-15",
      "quantity": 10,
      "premium_paid": 5.50
    }
  ]
}
```

### **Monitor Live Greeks**
```bash
# Start the system
./start_portfolio_system.sh

# In another terminal, watch updates
watch -n 5 'curl -s http://localhost:8080/portfolio/greeks | python -m json.tool'
```

### **Generate Reports**
```bash
# Get current status
curl http://localhost:8080/portfolio/greeks > portfolio_report_$(date +%Y%m%d).json

# Performance stats
curl http://localhost:8080/stats
```

---

## ✅ **Success Checklist**

- [ ] Files uploaded to college server
- [ ] `./setup_user_space.sh` completed successfully
- [ ] `./quick_test.sh` shows "READY" status
- [ ] Portfolio loaded in `user_portfolio_sample.json`
- [ ] System starts with `./start_portfolio_system.sh`
- [ ] API responds at `http://localhost:8080/health`
- [ ] Greeks updating every 30 seconds
- [ ] Logs writing to `portfolio_greeks.log`

---

## 🎉 **You're Ready!**

Your **real-time portfolio Greeks system** is now running on your college server with:

- ✅ **No sudo privileges required**
- ✅ **Live options data from Yahoo Finance**
- ✅ **GPU acceleration** (if available)
- ✅ **Real-time risk monitoring**
- ✅ **REST API interface**
- ✅ **Professional-grade performance**

**Perfect for academic research, class projects, and learning quantitative finance!** 🎓📈🚀
