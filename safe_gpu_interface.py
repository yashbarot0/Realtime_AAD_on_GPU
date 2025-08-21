import ctypes
import os
from typing import Dict, List
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import traceback

@dataclass
class PortfolioGreeks:
    total_delta: float
    total_vega: float  
    total_gamma: float
    total_theta: float
    total_rho: float
    total_pnl: float
    timestamp: datetime

class SafeGPUInterface:
    def __init__(self):
        """Safe GPU interface with comprehensive error handling"""
        self.use_gpu = False
        self.lib = None
        self.manager = None
        
        # Portfolio positions
        self.portfolio_positions = {
            'AAPL': {'quantity': 1000, 'entry_price': 200.0},
            'MSFT': {'quantity': 500, 'entry_price': 450.0}, 
            'GOOGL': {'quantity': 200, 'entry_price': 180.0}
        }
        
        self.current_greeks = PortfolioGreeks(0, 0, 0, 0, 0, 0, datetime.now())
        
        # Try to initialize GPU safely
        self._safe_gpu_init()
    
    def _safe_gpu_init(self):
        """Safely attempt GPU initialization with detailed error reporting"""
        try:
            print("🔍 Checking GPU library availability...")
            
            # Check multiple possible paths
            possible_paths = [
                "./build/libgpu_aad.so",
                "./build/libgpu_aad_realtime.so", 
                "build/libgpu_aad.so",
                "libgpu_aad.so"
            ]
            
            lib_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    print(f"✅ Found GPU library at: {path}")
                    break
            
            if not lib_path:
                print("❌ GPU library not found at any expected location")
                print("📁 Available files:")
                try:
                    files = os.listdir(".")
                    for f in files:
                        if f.endswith('.so'):
                            print(f"   {f}")
                    if os.path.exists("build"):
                        build_files = os.listdir("build")
                        for f in build_files:
                            if f.endswith('.so'):
                                print(f"   build/{f}")
                except:
                    pass
                print("🔄 Using CPU fallback mode")
                return
            
            print(f"📚 Loading library: {lib_path}")
            
            # Load library with error checking
            self.lib = ctypes.CDLL(lib_path)
            print("✅ Library loaded successfully")
            
            # Setup function signatures safely
            self._setup_function_signatures()
            print("✅ Function signatures configured")
            
            # Create manager safely
            print("🏗️ Creating portfolio manager...")
            self.manager = self.lib.create_portfolio_manager()
            
            if not self.manager:
                print("❌ Failed to create portfolio manager")
                return
            
            print("✅ Portfolio manager created")
            
            # Start processing safely
            print("▶️ Starting processing...")
            self.lib.start_processing(self.manager)
            print("✅ Processing started")
            
            self.use_gpu = True
            print("🚀 GPU mode activated successfully!")
            
        except Exception as e:
            print(f"❌ GPU initialization failed: {e}")
            print(f"🔍 Error details:")
            traceback.print_exc()
            print("🔄 Falling back to CPU mode")
            self.use_gpu = False
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        try:
            # Set return types
            self.lib.create_portfolio_manager.restype = ctypes.c_void_p
            
            # Set argument types  
            self.lib.destroy_portfolio_manager.argtypes = [ctypes.c_void_p]
            self.lib.start_processing.argtypes = [ctypes.c_void_p]
            self.lib.stop_processing.argtypes = [ctypes.c_void_p]
            
            print("✅ Basic function signatures set")
            
        except Exception as e:
            print(f"❌ Function signature setup failed: {e}")
            raise
    
    def calculate_cpu_greeks(self, S, K, T, r, sigma, is_call=True):
        """CPU Greeks calculation as fallback"""
        try:
            from math import log, sqrt, exp
            from scipy.stats import norm
            
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0}
            
            d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
            
            if is_call:
                delta = norm.cdf(d1)
                rho = K*T*exp(-r*T)*norm.cdf(d2)
            else:
                delta = norm.cdf(d1) - 1
                rho = -K*T*exp(-r*T)*norm.cdf(-d2)
            
            gamma = norm.pdf(d1) / (S*sigma*sqrt(T))
            vega = S*norm.pdf(d1)*sqrt(T) / 100.0  # Vega per 1% vol change
            
            if is_call:
                theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
            else:
                theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2)
            
            theta /= 365.0  # Theta per day
            
            return {
                'delta': delta,
                'vega': vega, 
                'gamma': gamma,
                'theta': theta,
                'rho': rho
            }
        except Exception as e:
            print(f"Error in CPU Greeks: {e}")
            return {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0}
    
    def process_portfolio_options(self, options_data: List[Dict], market_data: Dict):
        """Process options and calculate portfolio Greeks"""
        total_greeks = {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0, 'pnl': 0}
        
        # Calculate P&L
        for symbol, data in market_data.items():
            if symbol in self.portfolio_positions:
                position = self.portfolio_positions[symbol]
                spot_price = data.get('spot_price', 0)
                pnl = (spot_price - position['entry_price']) * position['quantity']
                total_greeks['pnl'] += pnl
        
        # Process options
        processed_count = 0
        for option in options_data:
            symbol = option['symbol']
            
            if symbol not in self.portfolio_positions:
                continue
            
            position = self.portfolio_positions[symbol]
            
            # Calculate Greeks (CPU for now)
            greeks = self.calculate_cpu_greeks(
                S=option['spot_price'],
                K=option['strike'],
                T=option['time_to_expiry'], 
                r=option['risk_free_rate'],
                sigma=option['implied_volatility'],
                is_call=option['is_call']
            )
            
            # Weight by position
            position_weight = position['quantity'] / 100.0
            
            total_greeks['delta'] += greeks['delta'] * position_weight
            total_greeks['vega'] += greeks['vega'] * position_weight  
            total_greeks['gamma'] += greeks['gamma'] * position_weight
            total_greeks['theta'] += greeks['theta'] * position_weight
            total_greeks['rho'] += greeks['rho'] * position_weight
            
            processed_count += 1
        
        # Update current Greeks
        self.current_greeks = PortfolioGreeks(
            total_delta=total_greeks['delta'],
            total_vega=total_greeks['vega'],
            total_gamma=total_greeks['gamma'], 
            total_theta=total_greeks['theta'],
            total_rho=total_greeks['rho'],
            total_pnl=total_greeks['pnl'],
            timestamp=datetime.now()
        )
        
        return processed_count
    
    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks"""
        return self.current_greeks
    
    def get_positions(self):
        """Get current positions"""
        return self.portfolio_positions
    
    def __del__(self):
        """Safe cleanup"""
        if self.use_gpu and self.manager and self.lib:
            try:
                self.lib.stop_processing(self.manager)
                self.lib.destroy_portfolio_manager(self.manager)
                print("✅ GPU resources cleaned up")
            except:
                pass

# Test the safe interface
if __name__ == "__main__":
    print("🧪 Testing Safe GPU Portfolio Interface")
    print("=" * 50)
    
    try:
        # Create safe interface
        interface = SafeGPUInterface()
        
        # Test data
        options_data = [
            {
                'symbol': 'AAPL',
                'strike': 225.0,
                'spot_price': 225.74,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.25,
                'is_call': True,
                'market_price': 8.50
            },
            {
                'symbol': 'MSFT', 
                'strike': 505.0,
                'spot_price': 504.07,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.30,
                'is_call': True,
                'market_price': 15.20
            }
        ]
        
        market_data = {
            'AAPL': {'spot_price': 225.74},
            'MSFT': {'spot_price': 504.07},
            'GOOGL': {'spot_price': 200.54}
        }
        
        print(f"\n📊 Portfolio Positions:")
        for symbol, pos in interface.get_positions().items():
            print(f"  {symbol}: {pos['quantity']:,} @ ${pos['entry_price']:.2f}")
        
        print(f"\n⚙️ Processing {len(options_data)} options...")
        processed = interface.process_portfolio_options(options_data, market_data)
        
        greeks = interface.get_portfolio_greeks()
        
        print(f"\n💰 Portfolio Greeks (Processed {processed} options):")
        print(f"  Delta:  {greeks.total_delta:>12.2f}")
        print(f"  Vega:   {greeks.total_vega:>12.2f}")  
        print(f"  Gamma:  {greeks.total_gamma:>12.6f}")
        print(f"  Theta:  {greeks.total_theta:>12.2f}")
        print(f"  Rho:    {greeks.total_rho:>12.2f}")
        print(f"  P&L:    ${greeks.total_pnl:>11,.2f}")
        
        print(f"\n🖥️ Compute Mode: {'GPU' if interface.use_gpu else 'CPU'}")
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
 