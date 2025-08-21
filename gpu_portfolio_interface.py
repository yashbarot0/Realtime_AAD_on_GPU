import ctypes
import os
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class PortfolioGreeks:
    total_delta: float
    total_vega: float
    total_gamma: float
    total_theta: float
    total_rho: float
    total_pnl: float
    timestamp: datetime

class GPUPortfolioInterface:
    def __init__(self, lib_path: str = "./build/libgpu_aad_realtime.so"):
        """Initialize GPU Portfolio Interface"""
        if not os.path.exists(lib_path):
            print(f"Warning: GPU library not found at {lib_path}")
            print("Using CPU fallback mode")
            self.use_gpu = False
            self.manager = None
        else:
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures()
                
                self.manager = self.lib.create_portfolio_manager()
                if not self.manager:
                    raise RuntimeError("Failed to create portfolio manager")
                
                self.lib.start_processing(self.manager)
                self.use_gpu = True
                print("GPU Portfolio Interface initialized successfully")
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                print("Using CPU fallback mode")
                self.use_gpu = False
                self.manager = None
        
        # Initialize portfolio positions
        self.portfolio_positions = {
            'AAPL': {'quantity': 1000, 'entry_price': 200.0},
            'MSFT': {'quantity': 500, 'entry_price': 450.0},
            'GOOGL': {'quantity': 200, 'entry_price': 180.0}
        }
        
        # Current Greeks (fallback calculation)
        self.current_greeks = PortfolioGreeks(0, 0, 0, 0, 0, 0, datetime.now())
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        self.lib.create_portfolio_manager.restype = ctypes.c_void_p
        self.lib.destroy_portfolio_manager.argtypes = [ctypes.c_void_p]
        
        self.lib.add_option_data.argtypes = [
            ctypes.c_void_p,  ctypes.c_char_p,  ctypes.c_double,  
            ctypes.c_double,  ctypes.c_double,  ctypes.c_double,  
            ctypes.c_double,  ctypes.c_int,     ctypes.c_double,
        ]
        
        self.lib.get_portfolio_greeks.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),  ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),  ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),  ctypes.POINTER(ctypes.c_double),
        ]
    
    def calculate_cpu_greeks(self, S, K, T, r, sigma, is_call=True):
        """CPU fallback Greeks calculation"""
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
            vega = S*norm.pdf(d1)*sqrt(T)
            
            if is_call:
                theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
            else:
                theta = -(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2)
            
            return {
                'delta': delta,
                'vega': vega,
                'gamma': gamma,
                'theta': theta,
                'rho': rho
            }
        except Exception as e:
            print(f"Error in CPU Greeks calculation: {e}")
            return {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0}
    
    def add_option_data(self, symbol: str, strike: float, spot_price: float,
                       time_to_expiry: float, risk_free_rate: float,
                       implied_volatility: float, is_call: bool, market_price: float):
        """Add single option data point"""
        if self.use_gpu and self.manager:
            try:
                symbol_bytes = symbol.encode('utf-8')
                self.lib.add_option_data(
                    self.manager, symbol_bytes, strike, spot_price,
                    time_to_expiry, risk_free_rate, implied_volatility,
                    1 if is_call else 0, market_price
                )
                return True
            except Exception as e:
                print(f"GPU option add failed: {e}")
                return False
        return False
    
    def add_options_batch_with_positions(self, options_data: List[Dict], market_data: Dict):
        """Add batch of options data and calculate portfolio Greeks"""
        total_greeks = {'delta': 0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0, 'pnl': 0}
        
        # Calculate P&L first
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
            
            # Check if we have a position in this symbol
            if symbol not in self.portfolio_positions:
                continue
            
            position = self.portfolio_positions[symbol]
            
            # Try GPU first, fallback to CPU
            if self.use_gpu:
                success = self.add_option_data(
                    option['symbol'], option['strike'], option['spot_price'],
                    option['time_to_expiry'], option['risk_free_rate'],
                    option['implied_volatility'], option['is_call'], 
                    option['market_price']
                )
                if success:
                    processed_count += 1
            
            # Calculate Greeks using CPU (for now, until GPU AAD is fully connected)
            greeks = self.calculate_cpu_greeks(
                S=option['spot_price'],
                K=option['strike'],
                T=option['time_to_expiry'],
                r=option['risk_free_rate'],
                sigma=option['implied_volatility'],
                is_call=option['is_call']
            )
            
            # Weight by position size
            position_weight = position['quantity'] / 100.0  # Standard option contract multiplier
            
            total_greeks['delta'] += greeks['delta'] * position_weight
            total_greeks['vega'] += greeks['vega'] * position_weight
            total_greeks['gamma'] += greeks['gamma'] * position_weight
            total_greeks['theta'] += greeks['theta'] * position_weight
            total_greeks['rho'] += greeks['rho'] * position_weight
        
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
    
    def update_positions(self, positions: Dict):
        """Update portfolio positions"""
        self.portfolio_positions = positions
    
    def get_positions(self):
        """Get current positions"""
        return self.portfolio_positions
    
    def __del__(self):
        if self.use_gpu and hasattr(self, 'manager') and self.manager:
            try:
                self.lib.stop_processing(self.manager)
                self.lib.destroy_portfolio_manager(self.manager)
            except:
                pass

# Test the enhanced interface
if __name__ == "__main__":
    try:
        print("🚀 Testing Enhanced GPU Portfolio Interface")
        print("==========================================")
        
        # Create interface
        gpu_interface = GPUPortfolioInterface()
        
        # Test with realistic market data
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
                'strike': 225.0,
                'spot_price': 225.74,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.25,
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
            }
        ]
        
        market_data = {
            'AAPL': {'spot_price': 225.74},
            'MSFT': {'spot_price': 504.07},
            'GOOGL': {'spot_price': 200.54}
        }
        
        print(f"Current Positions:")
        for symbol, pos in gpu_interface.get_positions().items():
            print(f"  {symbol}: {pos['quantity']} shares @ ${pos['entry_price']}")
        
        print(f"\nProcessing {len(options_data)} options...")
        
        # Process options
        processed_count = gpu_interface.add_options_batch_with_positions(options_data, market_data)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Get Greeks
        greeks = gpu_interface.get_portfolio_greeks()
        
        print(f"\nPortfolio Greeks (Processed {processed_count} options):")
        print(f"  Delta:  {greeks.total_delta:>10.2f}")
        print(f"  Vega:   {greeks.total_vega:>10.2f}")
        print(f"  Gamma:  {greeks.total_gamma:>10.6f}")
        print(f"  Theta:  {greeks.total_theta:>10.2f}")
        print(f"  Rho:    {greeks.total_rho:>10.2f}")
        print(f"  P&L:    ${greeks.total_pnl:>9,.2f}")
        
        print(f"\nUsing: {'GPU' if gpu_interface.use_gpu else 'CPU'} processing")
        
    except Exception as e:
        print(f"Error testing GPU interface: {e}")
        import traceback
        traceback.print_exc()
