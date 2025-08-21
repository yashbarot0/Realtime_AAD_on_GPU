import ctypes
import os
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import threading
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
    def __init__(self, lib_path: str = "./build/libgpu_aad.so"):
        """Initialize GPU Portfolio Interface"""
        # Load the shared library
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"GPU library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._setup_function_signatures()
        
        # Create portfolio manager
        self.manager = self.lib.create_portfolio_manager()
        if not self.manager:
            raise RuntimeError("Failed to create portfolio manager")
        
        # Start processing
        self.lib.start_processing(self.manager)
        
        print("GPU Portfolio Interface initialized successfully")
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        # create_portfolio_manager
        self.lib.create_portfolio_manager.restype = ctypes.c_void_p
        
        # destroy_portfolio_manager
        self.lib.destroy_portfolio_manager.argtypes = [ctypes.c_void_p]
        
        # add_option_data
        self.lib.add_option_data.argtypes = [
            ctypes.c_void_p,  # manager
            ctypes.c_char_p,  # symbol
            ctypes.c_double,  # strike
            ctypes.c_double,  # spot_price
            ctypes.c_double,  # time_to_expiry
            ctypes.c_double,  # risk_free_rate
            ctypes.c_double,  # implied_volatility
            ctypes.c_int,     # is_call
            ctypes.c_double,  # market_price
        ]
        
        # get_portfolio_greeks
        self.lib.get_portfolio_greeks.argtypes = [
            ctypes.c_void_p,     # manager
            ctypes.POINTER(ctypes.c_double),  # total_delta
            ctypes.POINTER(ctypes.c_double),  # total_vega
            ctypes.POINTER(ctypes.c_double),  # total_gamma
            ctypes.POINTER(ctypes.c_double),  # total_theta
            ctypes.POINTER(ctypes.c_double),  # total_rho
            ctypes.POINTER(ctypes.c_double),  # total_pnl
        ]
        
        # start_processing
        self.lib.start_processing.argtypes = [ctypes.c_void_p]
        
        # stop_processing
        self.lib.stop_processing.argtypes = [ctypes.c_void_p]
    
    def add_option_data(self, symbol: str, strike: float, spot_price: float,
                       time_to_expiry: float, risk_free_rate: float,
                       implied_volatility: float, is_call: bool, market_price: float):
        """Add single option data point"""
        symbol_bytes = symbol.encode('utf-8')
        self.lib.add_option_data(
            self.manager,
            symbol_bytes,
            strike,
            spot_price,
            time_to_expiry,
            risk_free_rate,
            implied_volatility,
            1 if is_call else 0,
            market_price
        )
    
    def add_options_batch(self, options_data: List[Dict]):
        """Add batch of options data"""
        for option in options_data:
            self.add_option_data(
                option['symbol'],
                option['strike'],
                option['spot_price'],
                option['time_to_expiry'],
                option['risk_free_rate'],
                option['implied_volatility'],
                option['is_call'],
                option['market_price']
            )
    
    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks"""
        delta = ctypes.c_double()
        vega = ctypes.c_double()
        gamma = ctypes.c_double()
        theta = ctypes.c_double()
        rho = ctypes.c_double()
        pnl = ctypes.c_double()
        
        self.lib.get_portfolio_greeks(
            self.manager,
            ctypes.byref(delta),
            ctypes.byref(vega),
            ctypes.byref(gamma),
            ctypes.byref(theta),
            ctypes.byref(rho),
            ctypes.byref(pnl)
        )
        
        return PortfolioGreeks(
            total_delta=delta.value,
            total_vega=vega.value,
            total_gamma=gamma.value,
            total_theta=theta.value,
            total_rho=rho.value,
            total_pnl=pnl.value,
            timestamp=datetime.now()
        )
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'manager') and self.manager:
            self.lib.stop_processing(self.manager)
            self.lib.destroy_portfolio_manager(self.manager)

# Test the interface
def test_gpu_interface():
    """Test the GPU portfolio interface"""
    try:
        # Create interface
        gpu_interface = GPUPortfolioInterface()
        
        # Add some test data
        test_options = [
            {
                'symbol': 'AAPL',
                'strike': 150.0,
                'spot_price': 155.0,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.25,
                'is_call': True,
                'market_price': 8.50
            },
            {
                'symbol': 'AAPL',
                'strike': 150.0,
                'spot_price': 155.0,
                'time_to_expiry': 0.25,
                'risk_free_rate': 0.05,
                'implied_volatility': 0.25,
                'is_call': False,
                'market_price': 3.25
            }
        ]
        
        # Add batch
        gpu_interface.add_options_batch(test_options)
        
        # Wait for processing
        time.sleep(1)
        
        # Get Greeks
        greeks = gpu_interface.get_portfolio_greeks()
        
        print("Portfolio Greeks:")
        print(f"  Delta: {greeks.total_delta:.4f}")
        print(f"  Vega: {greeks.total_vega:.4f}")
        print(f"  Gamma: {greeks.total_gamma:.4f}")
        print(f"  Theta: {greeks.total_theta:.4f}")
        print(f"  Rho: {greeks.total_rho:.4f}")
        print(f"  P&L: ${greeks.total_pnl:.2f}")
        
    except Exception as e:
        print(f"Error testing GPU interface: {e}")

if __name__ == "__main__":
    test_gpu_interface()
