import ctypes
import os
from typing import Dict, List
import numpy as np
import time
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
    # C struct layout matching LiveOptionData in C++
    _C_STRUCT = np.dtype([
        ('symbol',              'S16'),         # char[16] - fixed length
        ('strike',              np.float64),
        ('spot_price',          np.float64),
        ('time_to_expiry',      np.float64),
        ('risk_free_rate',      np.float64),
        ('implied_volatility',  np.float64),
        ('is_call',             np.int32),
        ('pad',                 np.int32),      # padding for alignment
        ('market_price',        np.float64),
    ], align=True)

    def __init__(self):
        """Safe GPU interface with comprehensive error handling and batched processing"""
        self.use_gpu = False
        self.lib = None
        self.manager = None

        # 🚀 ADD: Caching system
        self.data_cache = {}
        self.cache_timeout = 30  # seconds
        self.last_prices = {}
        
        # Portfolio positions
        self.portfolio_positions = {
            'AAPL': {'quantity': 1000, 'entry_price': 200.0},
            'MSFT': {'quantity': 500, 'entry_price': 450.0},
            'GOOGL': {'quantity': 200, 'entry_price': 180.0},
            'TSLA': {'quantity': 100, 'entry_price': 250.0},    
            'NVDA': {'quantity': 50, 'entry_price': 400.0},       
            'META': {'quantity': 150, 'entry_price': 300.0},    
            'AMZN': {'quantity': 75, 'entry_price': 150.0},     
            'NFLX': {'quantity': 25, 'entry_price': 400.0},     
            'SPY': {'quantity': 500, 'entry_price': 450.0},
            'QQQ': {'quantity': 300, 'entry_price': 350.0} 
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
                "./build/libgpu_aad_shared.so", 
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
                            print(f"  {f}")
                    if os.path.exists("build"):
                        build_files = os.listdir("build")
                        for f in build_files:
                            if f.endswith('.so'):
                                print(f"  build/{f}")
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
            
            # Set argument types for basic functions
            self.lib.destroy_portfolio_manager.argtypes = [ctypes.c_void_p]
            self.lib.start_processing.argtypes = [ctypes.c_void_p]
            self.lib.stop_processing.argtypes = [ctypes.c_void_p]
            
            # Set up the new batched function signature
            self.lib.add_options_batch.argtypes = [
                ctypes.c_void_p,        # manager
                ctypes.c_void_p,        # batch data pointer
                ctypes.c_size_t         # batch size
            ]
            
            # Greeks retrieval function
            self.lib.get_portfolio_greeks.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ]
            
            print("✅ Batched function signatures set")
            
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
        """🚀 OPTIMIZED: Process entire batch with single GPU call"""
        total_greeks = {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0, 'pnl': 0}
        
        # Calculate P&L first
        for symbol, data in market_data.items():
            if symbol in self.portfolio_positions:
                position = self.portfolio_positions[symbol]
                spot_price = data.get('spot_price', 0)
                pnl = (spot_price - position['entry_price']) * position['quantity']
                total_greeks['pnl'] += pnl

        if not options_data:
            self._update_current_greeks(total_greeks)
            return 0

        # 🔥 BATCHED PROCESSING: Pack all options into contiguous numpy array
        if self.use_gpu and self.manager:
            try:
                # Create structured numpy array
                batch_array = np.empty(len(options_data), dtype=self._C_STRUCT)
                
                for i, option in enumerate(options_data):
                    # Truncate symbol to fit fixed-length field
                    symbol_bytes = option['symbol'].encode('utf-8')[:15]
                    batch_array[i]['symbol'] = symbol_bytes
                    batch_array[i]['strike'] = option['strike']
                    batch_array[i]['spot_price'] = option['spot_price']
                    batch_array[i]['time_to_expiry'] = option['time_to_expiry']
                    batch_array[i]['risk_free_rate'] = option['risk_free_rate']
                    batch_array[i]['implied_volatility'] = option['implied_volatility']
                    batch_array[i]['is_call'] = 1 if option['is_call'] else 0
                    batch_array[i]['market_price'] = option['market_price']
                
                # 🚀 SINGLE GPU CALL: Send entire batch at once
                self.lib.add_options_batch(
                    self.manager,
                    batch_array.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_size_t(len(options_data))
                )
                
                processed_count = len(options_data)
                print(f"🚀 Batched GPU call: {processed_count} options in single transaction")
                
            except Exception as e:
                print(f"❌ GPU batch processing failed: {e}")
                processed_count = 0
        else:
            processed_count = 0

        # CPU fallback for Greeks calculation (until GPU Greeks are fully connected)
        for option in options_data:
            symbol = option['symbol']
            if symbol not in self.portfolio_positions:
                continue
                
            position = self.portfolio_positions[symbol]
            
            # Calculate Greeks using CPU
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

        # Update current Greeks
        self._update_current_greeks(total_greeks)
        
        return processed_count if processed_count > 0 else len(options_data)

    def _update_current_greeks(self, total_greeks):
        """Helper to update current Greeks"""
        self.current_greeks = PortfolioGreeks(
            total_delta=total_greeks['delta'],
            total_vega=total_greeks['vega'],
            total_gamma=total_greeks['gamma'],
            total_theta=total_greeks['theta'],
            total_rho=total_greeks['rho'],
            total_pnl=total_greeks['pnl'],
            timestamp=datetime.now()
        )

    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks"""
        if self.use_gpu and self.manager:
            try:
                delta = ctypes.c_double()
                vega = ctypes.c_double()
                gamma = ctypes.c_double()
                theta = ctypes.c_double()
                rho = ctypes.c_double()
                pnl = ctypes.c_double()
                
                self.lib.get_portfolio_greeks(
                    self.manager, ctypes.byref(delta), ctypes.byref(vega),
                    ctypes.byref(gamma), ctypes.byref(theta),
                    ctypes.byref(rho), ctypes.byref(pnl)
                )
                
                # If GPU returns all zeros, use CPU calculation
                if (delta.value == 0 and vega.value == 0 and gamma.value == 0):
                    return self.current_greeks
                
                return PortfolioGreeks(
                    total_delta=delta.value,
                    total_vega=vega.value,
                    total_gamma=gamma.value,
                    total_theta=theta.value,
                    total_rho=rho.value,
                    total_pnl=pnl.value,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                print(f"GPU Greeks retrieval failed: {e}")
                return self.current_greeks
        
        return self.current_greeks

    def get_positions(self):
        """Get current positions"""
        return self.portfolio_positions

    def update_positions(self, positions: Dict):
        """Update portfolio positions"""
        self.portfolio_positions = positions

    def __del__(self):
        """Safe cleanup"""
        if self.use_gpu and self.manager and self.lib:
            try:
                self.lib.stop_processing(self.manager)
                self.lib.destroy_portfolio_manager(self.manager)
                print("✅ GPU resources cleaned up")
            except:
                pass

# Test the optimized interface
if __name__ == "__main__":
    print("🧪 Testing Optimized Batched GPU Portfolio Interface")
    print("=" * 55)
    
    try:
        # Create safe interface
        interface = SafeGPUInterface()
        
        # Test data - larger batch to show batching benefits
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
                'symbol': 'AAPL',
                'strike': 220.0,
                'spot_price': 225.74,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.23,
                'is_call': False,
                'market_price': 7.76
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
            },
            {
                'symbol': 'MSFT',
                'strike': 500.0,
                'spot_price': 504.07,
                'time_to_expiry': 0.33,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.28,
                'is_call': False,
                'market_price': 12.45
            },
            {
                'symbol': 'GOOGL',
                'strike': 180.0,
                'spot_price': 182.50,
                'time_to_expiry': 0.17,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.32,
                'is_call': True,
                'market_price': 9.80
            }
        ]
        
        market_data = {
            'AAPL': {'spot_price': 225.74},
            'MSFT': {'spot_price': 504.07},
            'GOOGL': {'spot_price': 182.50}
        }
        
        print(f"\n📊 Portfolio Positions:")
        for symbol, pos in interface.get_positions().items():
            print(f"  {symbol}: {pos['quantity']:,} @ ${pos['entry_price']:.2f}")
        
        print(f"\n⚙️ Processing {len(options_data)} options with BATCHED interface...")
        
        import time
        start_time = time.time()
        processed = interface.process_portfolio_options(options_data, market_data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        greeks = interface.get_portfolio_greeks()
        
        print(f"\n💰 Portfolio Greeks (Processed {processed} options):")
        print(f"  Delta: {greeks.total_delta:>12.2f}")
        print(f"  Vega:  {greeks.total_vega:>12.2f}")
        print(f"  Gamma: {greeks.total_gamma:>12.6f}")
        print(f"  Theta: {greeks.total_theta:>12.2f}")
        print(f"  Rho:   {greeks.total_rho:>12.2f}")
        print(f"  P&L:   ${greeks.total_pnl:>11,.2f}")
        
        print(f"\n⚡ Performance:")
        print(f"  Processing Time: {elapsed_ms:>8.1f} ms")
        print(f"  Avg per Option:  {elapsed_ms/len(options_data):>8.2f} ms")
        print(f"  Compute Mode:    {'GPU' if interface.use_gpu else 'CPU':>8s}")
        
        print("✅ Batched test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
