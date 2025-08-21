#!/usr/bin/env python3
"""
Quick test script to verify the system works
"""

import asyncio
import json
from datetime import datetime, timezone

async def test_basic_functionality():
    """Test basic system functionality without real market data"""
    print("🧪 Testing Real-Time Portfolio Greeks System")
    print("=" * 50)
    
    try:
        # Import our system
        from realtime_portfolio_system import RealTimePortfolioSystem
        
        print("✅ Successfully imported RealTimePortfolioSystem")
        
        # Create a test system
        system = RealTimePortfolioSystem(['AAPL'], update_interval=5)
        print("✅ Created system instance")
        
        # Add test positions
        pos_id1 = system.add_position('AAPL', 'CALL', 190.0, '2024-03-15', 10, 5.50)
        pos_id2 = system.add_position('AAPL', 'PUT', 180.0, '2024-03-15', 5, 3.20)
        
        print(f"✅ Added test positions: {pos_id1}, {pos_id2}")
        
        # Test portfolio manager
        positions = system.portfolio_manager.get_positions()
        print(f"✅ Retrieved {len(positions)} positions")
        
        # Test callbacks
        updates_received = 0
        alerts_received = 0
        
        async def test_callback(greeks):
            nonlocal updates_received
            updates_received += 1
            print(f"📊 Received Greeks update #{updates_received}")
            print(f"   Delta: {greeks.get('total_delta', 0):.2f}")
            print(f"   Positions: {greeks.get('num_positions', 0)}")
        
        async def alert_callback(alert):
            nonlocal alerts_received
            alerts_received += 1
            print(f"⚠️  Received alert: {alert.get('message', 'Unknown')}")
        
        system.set_greeks_callback(test_callback)
        system.set_alert_callback(alert_callback)
        print("✅ Set up callbacks")
        
        # Test for a short time
        print("\n🏃 Running quick test (30 seconds)...")
        
        # Start system (will run for 30 seconds then timeout)
        try:
            await asyncio.wait_for(system.start(), timeout=30)
        except asyncio.TimeoutError:
            print("⏰ Test timeout reached")
        
        await system.stop()
        
        print(f"\n📈 Test Results:")
        print(f"   Updates received: {updates_received}")
        print(f"   Alerts received: {alerts_received}")
        
        if updates_received > 0:
            print("✅ System is working correctly!")
        else:
            print("⚠️  No updates received - check network/API access")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure to install required packages:")
        print("   pip install yfinance pandas numpy scipy")
        return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

async def test_market_data_connection():
    """Test connection to market data"""
    print("\n🌐 Testing market data connection...")
    
    try:
        import yfinance as yf
        
        # Test basic ticker fetch
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d", interval="1m")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            print(f"✅ Successfully fetched AAPL price: ${current_price:.2f}")
            
            # Test options data
            exp_dates = ticker.options
            if exp_dates:
                print(f"✅ Found {len(exp_dates)} expiration dates")
                
                # Try to get option chain
                opt_chain = ticker.option_chain(exp_dates[0])
                calls = len(opt_chain.calls)
                puts = len(opt_chain.puts)
                print(f"✅ Retrieved option chain: {calls} calls, {puts} puts")
                
                return True
            else:
                print("⚠️  No options data available for AAPL")
                return False
        else:
            print("❌ Could not fetch stock price")
            return False
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
        return False

def test_gpu_availability():
    """Test if CUDA/GPU is available"""
    print("\n🖥️  Testing GPU availability...")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("⚠️  nvidia-smi not found - GPU may not be available")
            return False
            
    except FileNotFoundError:
        print("⚠️  NVIDIA drivers not found")
        return False
    except Exception as e:
        print(f"❌ GPU test error: {e}")
        return False

def create_test_config():
    """Create a test configuration file"""
    print("\n📝 Creating test configuration...")
    
    config = {
        "test_mode": True,
        "market_data": {
            "update_interval_seconds": 5,
            "symbols": ["AAPL", "MSFT"],
            "data_source": "yfinance"
        },
        "risk_limits": {
            "max_delta": 100.0,  # Lower for testing
            "max_gamma": 10.0,
            "max_vega": 500.0,
            "max_loss": -5000.0,
            "max_exposure": 50000.0
        },
        "compute_config": {
            "update_frequency_ms": 2000,
            "enable_second_order": False,  # Disable for testing
            "use_gpu": False,  # CPU mode for testing
            "max_positions": 10
        }
    }
    
    try:
        with open('test_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Created test_config.json")
        return True
    except Exception as e:
        print(f"❌ Error creating config: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Real-Time Portfolio Greeks - System Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test GPU
    gpu_ok = test_gpu_availability()
    
    # Test market data
    market_ok = await test_market_data_connection()
    all_passed = all_passed and market_ok
    
    # Create test config
    config_ok = create_test_config()
    all_passed = all_passed and config_ok
    
    # Test main functionality
    if market_ok:
        system_ok = await test_basic_functionality()
        all_passed = all_passed and system_ok
    else:
        print("⏭️  Skipping system test due to market data issues")
        system_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   GPU Available: {'✅' if gpu_ok else '❌'}")
    print(f"   Market Data: {'✅' if market_ok else '❌'}")
    print(f"   Configuration: {'✅' if config_ok else '❌'}")
    print(f"   System Test: {'✅' if system_ok else '❌'}")
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run ./setup.sh to complete installation")
        print("2. Edit sample_portfolio.json with your positions")
        print("3. Run ./run_python_demo.sh for full demo")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("\nTroubleshooting:")
        if not market_ok:
            print("- Check internet connection")
            print("- Install: pip install yfinance pandas numpy scipy")
        if not gpu_ok:
            print("- GPU is optional for basic functionality")
            print("- Install NVIDIA drivers for GPU acceleration")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())
