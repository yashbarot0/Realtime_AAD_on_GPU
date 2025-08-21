#!/usr/bin/env python3
"""
User-Space Portfolio System Test
Optimized for college servers without sudo access
"""

import asyncio
import json
import subprocess
import sys
import time
import os
from pathlib import Path

class UserSpaceTest:
    """Test suite for user-space installation"""
    
    def __init__(self):
        self.results = {
            'environment': {},
            'dependencies': {},
            'build_system': {},
            'portfolio_system': {},
            'overall_status': 'UNKNOWN'
        }
        self.user_home = Path.home()
        self.local_prefix = self.user_home / '.local'
    
    def print_header(self, title):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print('='*60)
    
    def print_status(self, test_name, status, details=""):
        """Print test status"""
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test_name:<40} {details}")
    
    def test_user_environment(self):
        """Test user environment setup"""
        self.print_header("User Environment")
        
        all_good = True
        
        # Check home directory permissions
        home_writable = os.access(self.user_home, os.W_OK)
        self.print_status("Home directory writable", home_writable)
        all_good = all_good and home_writable
        
        # Check .local directory
        local_exists = self.local_prefix.exists()
        if not local_exists:
            try:
                self.local_prefix.mkdir(parents=True, exist_ok=True)
                local_exists = True
            except:
                pass
        
        self.print_status("User .local directory", local_exists)
        all_good = all_good and local_exists
        
        # Check virtual environment
        venv_path = Path('venv')
        venv_exists = venv_path.exists()
        self.print_status("Python virtual environment", venv_exists)
        
        if venv_exists:
            # Check if venv is properly activated
            python_path = venv_path / 'bin' / 'python'
            venv_python = python_path.exists()
            self.print_status("Virtual environment Python", venv_python)
            all_good = all_good and venv_python
        
        # Check for module system
        module_available = subprocess.run(['which', 'module'], 
                                        capture_output=True).returncode == 0
        self.print_status("Module system available", module_available)
        
        if module_available:
            try:
                result = subprocess.run(['module', 'avail'], 
                                      capture_output=True, text=True, timeout=10)
                available_modules = result.stderr  # module output goes to stderr
                
                useful_modules = []
                for module in ['gcc', 'cuda', 'python', 'cmake']:
                    if module in available_modules.lower():
                        useful_modules.append(module)
                
                self.print_status("Useful modules found", len(useful_modules) > 0, 
                                f"{', '.join(useful_modules)}")
                
            except Exception as e:
                self.print_status("Module query", False, str(e))
        
        return all_good
    
    def test_python_setup(self):
        """Test Python installation and packages"""
        self.print_header("Python Environment")
        
        all_good = True
        
        # Test Python version
        try:
            python_version = sys.version_info
            version_ok = python_version >= (3, 7)
            self.print_status("Python version", version_ok, 
                            f"{python_version.major}.{python_version.minor}.{python_version.micro}")
            all_good = all_good and version_ok
        except:
            self.print_status("Python version check", False)
            all_good = False
        
        # Test pip
        try:
            import pip
            self.print_status("pip available", True)
        except ImportError:
            self.print_status("pip available", False)
            all_good = False
        
        # Test required packages
        required_packages = [
            'numpy', 'pandas', 'scipy', 'yfinance'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_status(f"Package: {package}", True)
            except ImportError:
                self.print_status(f"Package: {package}", False)
                all_good = False
        
        # Test optional packages
        optional_packages = {
            'fastapi': 'Web API',
            'aiohttp': 'Async HTTP',
            'psutil': 'System monitoring',
            'numba': 'JIT compilation'
        }
        
        for package, description in optional_packages.items():
            try:
                __import__(package)
                self.print_status(f"Optional: {package}", True, description)
            except ImportError:
                self.print_status(f"Optional: {package}", False, f"{description} (optional)")
        
        return all_good
    
    def test_gpu_environment(self):
        """Test GPU and CUDA without requiring sudo"""
        self.print_header("GPU Environment")
        
        gpu_available = False
        cuda_available = False
        
        # Test nvidia-smi (user accessible)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 2:
                    self.print_status("GPU detected", True, f"{gpu_info[0]}")
                    self.print_status("GPU memory", True, f"{gpu_info[1]} MB")
                    gpu_available = True
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.print_status("nvidia-smi access", False)
        
        # Test CUDA compiler
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        cuda_version = line.strip()
                        self.print_status("CUDA compiler", True, cuda_version)
                        cuda_available = True
                        break
                        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Try alternative CUDA paths
            cuda_paths = ['/usr/local/cuda/bin/nvcc', '/opt/cuda/bin/nvcc']
            for cuda_path in cuda_paths:
                if Path(cuda_path).exists():
                    self.print_status("CUDA found", True, cuda_path)
                    cuda_available = True
                    break
            
            if not cuda_available:
                self.print_status("CUDA compiler", False)
        
        # Test GPU programming libraries
        try:
            import torch
            if torch.cuda.is_available():
                self.print_status("PyTorch CUDA", True, f"GPU count: {torch.cuda.device_count()}")
            else:
                self.print_status("PyTorch CUDA", False, "PyTorch available but no CUDA")
        except ImportError:
            self.print_status("PyTorch CUDA", False, "PyTorch not installed")
        
        return gpu_available or cuda_available
    
    def test_build_system(self):
        """Test build system without sudo"""
        self.print_header("Build System")
        
        build_ok = True
        
        # Test CMake
        cmake_available = False
        try:
            result = subprocess.run(['cmake', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.print_status("CMake available", True, version_line)
                cmake_available = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Check user-installed CMake
            user_cmake = self.local_prefix / 'bin' / 'cmake'
            if user_cmake.exists():
                self.print_status("User CMake", True, str(user_cmake))
                cmake_available = True
            else:
                self.print_status("CMake available", False)
        
        build_ok = build_ok and cmake_available
        
        # Test compiler
        compilers = ['g++', 'gcc', 'clang++']
        compiler_found = False
        
        for compiler in compilers:
            try:
                result = subprocess.run([compiler, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    self.print_status(f"Compiler: {compiler}", True, version_line)
                    compiler_found = True
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        if not compiler_found:
            self.print_status("C++ compiler", False)
            build_ok = False
        
        # Check if build was successful
        build_dir = Path('build')
        if build_dir.exists():
            executables = list(build_dir.glob('*'))
            executable_count = len([e for e in executables if e.is_file() and os.access(e, os.X_OK)])
            self.print_status("Build directory", True, f"{executable_count} executables")
            
            # Check specific executables
            expected_exes = ['GPU_AAD', 'portfolio_demo']
            for exe in expected_exes:
                exe_path = build_dir / exe
                if exe_path.exists():
                    self.print_status(f"Executable: {exe}", True)
                else:
                    self.print_status(f"Executable: {exe}", False)
        else:
            self.print_status("Build directory", False)
            build_ok = False
        
        return build_ok
    
    async def test_market_connectivity(self):
        """Test market data connectivity"""
        self.print_header("Market Data Connectivity")
        
        try:
            import yfinance as yf
            
            # Test basic connection with timeout
            print("Testing Yahoo Finance connection...")
            ticker = yf.Ticker("AAPL")
            
            # Use a shorter period for faster testing
            hist = ticker.history(period="1d", interval="1h")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                self.print_status("Stock data fetch", True, f"AAPL: ${current_price:.2f}")
                
                # Quick options test
                try:
                    exp_dates = ticker.options
                    if exp_dates:
                        self.print_status("Options expiration dates", True, f"{len(exp_dates)} dates")
                        
                        # Test one option chain (quickly)
                        opt_chain = ticker.option_chain(exp_dates[0])
                        calls = len(opt_chain.calls) if hasattr(opt_chain, 'calls') else 0
                        puts = len(opt_chain.puts) if hasattr(opt_chain, 'puts') else 0
                        self.print_status("Options chain data", calls > 0 or puts > 0, 
                                        f"{calls} calls, {puts} puts")
                        return True
                    else:
                        self.print_status("Options data", False, "No expiration dates")
                        return False
                except Exception as e:
                    self.print_status("Options data", False, str(e))
                    return True  # Stock data works, that's enough
                    
            else:
                self.print_status("Stock data fetch", False)
                return False
                
        except Exception as e:
            self.print_status("Market data test", False, str(e))
            return False
    
    async def test_portfolio_functionality(self):
        """Test portfolio system functionality"""
        self.print_header("Portfolio System")
        
        try:
            # Test basic import
            from realtime_portfolio_system import RealTimePortfolioSystem
            self.print_status("Import portfolio system", True)
            
            # Test system creation
            system = RealTimePortfolioSystem(['AAPL'], update_interval=60)
            self.print_status("Create system instance", True)
            
            # Test position management
            pos_id = system.add_position('AAPL', 'CALL', 190.0, '2024-03-15', 10, 5.50)
            positions = system.portfolio_manager.get_positions()
            self.print_status("Position management", len(positions) > 0, 
                            f"Added position {pos_id}")
            
            # Test configuration loading
            config_file = 'user_config.json'
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    self.print_status("Configuration loading", True, f"Loaded {config_file}")
                except Exception as e:
                    self.print_status("Configuration loading", False, str(e))
            else:
                self.print_status("Configuration file", False, f"{config_file} not found")
            
            return True
            
        except Exception as e:
            self.print_status("Portfolio system test", False, str(e))
            return False
    
    def test_resource_limits(self):
        """Test system resource constraints"""
        self.print_header("Resource Limits (College Server)")
        
        try:
            import psutil
            
            # Memory info
            memory = psutil.virtual_memory()
            available_gb = memory.available // (1024**3)
            self.print_status("Available memory", available_gb >= 2, f"{available_gb} GB")
            
            # CPU info
            cpu_count = psutil.cpu_count()
            self.print_status("CPU cores", cpu_count > 0, f"{cpu_count} cores")
            
            # Disk space
            disk = psutil.disk_usage(str(self.user_home))
            free_gb = disk.free // (1024**3)
            self.print_status("Home directory space", free_gb >= 1, f"{free_gb} GB free")
            
            # Process limits (approximate)
            try:
                import resource
                max_processes = resource.getrlimit(resource.RLIMIT_NPROC)[0]
                self.print_status("Process limit", max_processes > 100, f"{max_processes} processes")
            except:
                self.print_status("Process limit", True, "Unable to check")
            
            return True
            
        except Exception as e:
            self.print_status("Resource check", False, str(e))
            return False
    
    def save_results(self):
        """Save test results"""
        results_file = 'user_test_results.json'
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n📄 Test results saved to {results_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    async def run_all_tests(self):
        """Run all user-space tests"""
        print("🧪 User-Space Portfolio System Test")
        print("🎓 Optimized for college servers without sudo")
        print()
        
        test_results = []
        
        # Run all tests
        test_results.append(self.test_user_environment())
        test_results.append(self.test_python_setup())
        test_results.append(self.test_gpu_environment())
        test_results.append(self.test_build_system())
        test_results.append(await self.test_market_connectivity())
        test_results.append(await self.test_portfolio_functionality())
        test_results.append(self.test_resource_limits())
        
        # Calculate results
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        # Determine overall status
        if passed_tests >= 5:  # Core functionality
            self.results['overall_status'] = 'READY'
        elif passed_tests >= 3:  # Basic functionality
            self.results['overall_status'] = 'PARTIAL'
        else:
            self.results['overall_status'] = 'NEEDS_SETUP'
        
        # Print summary
        self.print_header("User-Space Test Summary")
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Overall Status: {self.results['overall_status']}")
        
        if self.results['overall_status'] == 'READY':
            print("\n🎉 System ready for use!")
            print("\n🚀 Next steps:")
            print("1. Edit portfolio: nano user_portfolio_sample.json")
            print("2. Start system: ./start_portfolio_system.sh")
            print("3. Monitor: ./monitor_user.sh")
            print("4. Access API: http://localhost:8080/health")
            
        elif self.results['overall_status'] == 'PARTIAL':
            print("\n⚠️  Core functionality available with limitations.")
            print("\n🔧 Recommendations:")
            print("- Some features may be disabled")
            print("- CPU-only mode if GPU unavailable")
            print("- Limited to basic portfolio operations")
            
        else:
            print("\n❌ Setup incomplete. Please run:")
            print("./setup_user_space.sh")
        
        # College server specific notes
        print(f"\n🎓 College Server Notes:")
        print("- All files in your home directory")
        print("- No admin privileges required")
        print("- Respects system resource limits")
        print("- Compatible with module systems")
        
        self.save_results()
        return self.results['overall_status'] == 'READY'

async def main():
    """Main test function"""
    tester = UserSpaceTest()
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
