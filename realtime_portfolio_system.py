#!/usr/bin/env python3
"""
Python-C++ bridge for the Real-Time Portfolio Engine
Connects live options data feed to GPU-accelerated Greeks computation
"""

import ctypes
import json
import asyncio
import numpy as np
from typing import Dict, List, Optional
from options_data_feed import OptionsDataFeed, PortfolioManager
from datetime import datetime, timezone
import time
import threading

# C++ Structure definitions (must match C++ structs exactly)
class BlackScholesParams(ctypes.Structure):
    _fields_ = [
        ('spot', ctypes.c_double),
        ('strike', ctypes.c_double),
        ('time', ctypes.c_double),
        ('rate', ctypes.c_double),
        ('volatility', ctypes.c_double),
        ('is_call', ctypes.c_bool)
    ]

class OptionResults(ctypes.Structure):
    _fields_ = [
        ('price', ctypes.c_double),
        ('delta', ctypes.c_double),
        ('vega', ctypes.c_double),
        ('gamma', ctypes.c_double),
        ('theta', ctypes.c_double),
        ('rho', ctypes.c_double)
    ]

class OptionPosition(ctypes.Structure):
    _fields_ = [
        ('position_id', ctypes.c_int),
        ('symbol', ctypes.c_char * 16),
        ('strike', ctypes.c_double),
        ('expiration_time', ctypes.c_double),
        ('quantity', ctypes.c_int),
        ('is_call', ctypes.c_bool),
        ('premium_paid', ctypes.c_double),
        ('entry_spot', ctypes.c_double)
    ]

class PortfolioGreeks(ctypes.Structure):
    _fields_ = [
        ('total_delta', ctypes.c_double),
        ('total_gamma', ctypes.c_double),
        ('total_vega', ctypes.c_double),
        ('total_theta', ctypes.c_double),
        ('total_rho', ctypes.c_double),
        ('total_pnl', ctypes.c_double),
        ('total_exposure', ctypes.c_double),
        ('num_positions', ctypes.c_int),
        ('timestamp', ctypes.c_longlong)
    ]

class RiskAlert(ctypes.Structure):
    _fields_ = [
        ('metric_name', ctypes.c_char * 32),
        ('current_value', ctypes.c_double),
        ('limit_value', ctypes.c_double),
        ('is_breach', ctypes.c_bool),
        ('timestamp', ctypes.c_longlong)
    ]


class RealTimePortfolioSystem:
    """Main system that combines data feed with GPU computation"""
    
    def __init__(self, symbols: List[str], update_interval: float = 30.0):
        self.symbols = symbols
        self.update_interval = update_interval
        
        # Components
        self.portfolio_manager = PortfolioManager()
        self.data_feed = None
        self.cpp_engine = None
        
        # State
        self.is_running = False
        self.last_greeks = None
        self.performance_stats = {
            'computations': 0,
            'avg_latency_ms': 0.0,
            'data_updates': 0,
            'start_time': None
        }
        
        # Callbacks
        self.greeks_callback = None
        self.alert_callback = None
        
        # Threading
        self.main_loop_task = None
        
    def add_position(self, symbol: str, option_type: str, strike: float, 
                    expiration: str, quantity: int, premium_paid: float = 0.0) -> int:
        """Add a position to the portfolio"""
        # Add to Python portfolio manager
        pos_id = self.portfolio_manager.add_position(
            symbol, option_type, strike, expiration, quantity, premium_paid
        )
        
        # TODO: Add to C++ engine when it's loaded
        # For now, we'll sync during computation
        
        return pos_id
    
    def remove_position(self, position_id: int) -> bool:
        """Remove a position from the portfolio"""
        return self.portfolio_manager.remove_position(position_id)
    
    def set_greeks_callback(self, callback):
        """Set callback for Greeks updates"""
        self.greeks_callback = callback
    
    def set_alert_callback(self, callback):
        """Set callback for risk alerts"""
        self.alert_callback = callback
    
    async def start(self):
        """Start the real-time system"""
        if self.is_running:
            return
            
        print("Starting Real-Time Portfolio Greeks System...")
        self.performance_stats['start_time'] = time.time()
        
        # Get symbols from portfolio
        portfolio_symbols = self.portfolio_manager.get_symbols()
        all_symbols = list(set(self.symbols + portfolio_symbols))
        
        if not all_symbols:
            print("Warning: No symbols to track")
            return
        
        # Initialize data feed
        self.data_feed = OptionsDataFeed(all_symbols, self.update_interval)
        
        # Start data streaming
        self.is_running = True
        self.main_loop_task = asyncio.create_task(self._main_loop())
        
        print(f"System started, tracking {len(all_symbols)} symbols: {all_symbols}")
    
    async def stop(self):
        """Stop the real-time system"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.data_feed:
            self.data_feed.stop_streaming()
        
        if self.main_loop_task:
            self.main_loop_task.cancel()
            try:
                await self.main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Print final statistics
        self._print_performance_stats()
        print("Real-Time Portfolio Greeks System stopped")
    
    async def _main_loop(self):
        """Main processing loop"""
        async def data_update_callback(market_data):
            """Handle new market data"""
            self.performance_stats['data_updates'] += 1
            
            # Compute portfolio Greeks
            greeks_result = await self._compute_portfolio_greeks(market_data)
            
            if greeks_result:
                self.last_greeks = greeks_result
                self.performance_stats['computations'] += 1
                
                # Update average latency
                if 'computation_time_ms' in greeks_result:
                    current_avg = self.performance_stats['avg_latency_ms']
                    count = self.performance_stats['computations']
                    new_latency = greeks_result['computation_time_ms']
                    
                    if count == 1:
                        self.performance_stats['avg_latency_ms'] = new_latency
                    else:
                        self.performance_stats['avg_latency_ms'] = \
                            (current_avg * (count - 1) + new_latency) / count
                
                # Call user callback
                if self.greeks_callback:
                    await self._safe_callback(self.greeks_callback, greeks_result)
                
                # Check for alerts
                await self._check_alerts(greeks_result)
        
        # Start data feed with callback
        await self.data_feed.start_streaming(data_update_callback)
    
    async def _compute_portfolio_greeks(self, market_data: Dict) -> Optional[Dict]:
        """Compute portfolio Greeks using current positions and market data"""
        start_time = time.time()
        
        try:
            positions = self.portfolio_manager.get_positions()
            
            if not positions:
                return None
            
            # Convert to computation format
            portfolio_greeks = {
                'total_delta': 0.0,
                'total_gamma': 0.0,
                'total_vega': 0.0,
                'total_theta': 0.0,
                'total_rho': 0.0,
                'total_pnl': 0.0,
                'total_exposure': 0.0,
                'num_positions': len(positions),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'position_details': []
            }
            
            # For each position, compute Greeks
            for position in positions:
                symbol = position['symbol']
                
                if symbol not in market_data:
                    print(f"Warning: No market data for {symbol}")
                    continue
                
                market = market_data[symbol]
                
                # Calculate time to expiration (simplified)
                # In real implementation, parse expiration date properly
                time_to_exp = 0.25  # 3 months default
                
                # Find closest strike price in market data
                option_data = None
                option_list = market['calls'] if position['option_type'] == 'CALL' else market['puts']
                
                if option_list:
                    # Find closest strike
                    closest_option = min(option_list, 
                                        key=lambda x: abs(x['strike'] - position['strike']))
                    option_data = closest_option
                
                if option_data:
                    # Use Black-Scholes approximation for Greeks
                    # (In production, this would call the GPU kernel)
                    greeks = self._approximate_greeks(
                        spot=market['spot_price'],
                        strike=position['strike'],
                        time=time_to_exp,
                        rate=market['risk_free_rate'],
                        volatility=option_data.get('impliedVolatility', 0.25),
                        is_call=position['option_type'] == 'CALL',
                        quantity=position['quantity']
                    )
                    
                    # Aggregate to portfolio level
                    portfolio_greeks['total_delta'] += greeks['delta']
                    portfolio_greeks['total_gamma'] += greeks['gamma']
                    portfolio_greeks['total_vega'] += greeks['vega']
                    portfolio_greeks['total_theta'] += greeks['theta']
                    portfolio_greeks['total_rho'] += greeks['rho']
                    portfolio_greeks['total_exposure'] += abs(greeks['price'])
                    
                    # Add position detail
                    position_detail = {
                        'position_id': position['id'],
                        'symbol': symbol,
                        'greeks': greeks,
                        'market_price': option_data.get('lastPrice', 0.0)
                    }
                    portfolio_greeks['position_details'].append(position_detail)
            
            computation_time = (time.time() - start_time) * 1000  # Convert to ms
            portfolio_greeks['computation_time_ms'] = computation_time
            
            return portfolio_greeks
            
        except Exception as e:
            print(f"Error computing portfolio Greeks: {e}")
            return None
    
    def _approximate_greeks(self, spot: float, strike: float, time: float, 
                           rate: float, volatility: float, is_call: bool, quantity: int) -> Dict:
        """Simplified Black-Scholes Greeks calculation"""
        import math
        from scipy.stats import norm
        
        # Black-Scholes Greeks (simplified implementation)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (volatility * math.sqrt(time))
        d2 = d1 - volatility * math.sqrt(time)
        
        if is_call:
            price = spot * norm.cdf(d1) - strike * math.exp(-rate * time) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = strike * math.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time))
        vega = spot * norm.pdf(d1) * math.sqrt(time) / 100  # Per 1% vol change
        theta = (-spot * norm.pdf(d1) * volatility / (2 * math.sqrt(time)) 
                - rate * strike * math.exp(-rate * time) * 
                (norm.cdf(d2) if is_call else norm.cdf(-d2))) / 365  # Per day
        rho = (strike * time * math.exp(-rate * time) * 
               (norm.cdf(d2) if is_call else norm.cdf(-d2))) / 100  # Per 1% rate change
        
        # Scale by quantity
        return {
            'price': price * quantity,
            'delta': delta * quantity,
            'gamma': gamma * quantity,
            'vega': vega * quantity,
            'theta': theta * quantity,
            'rho': rho * quantity
        }
    
    async def _check_alerts(self, greeks_result: Dict):
        """Check for risk alerts"""
        # Simple alert thresholds (configurable in production)
        alerts = []
        
        if abs(greeks_result['total_delta']) > 1000:
            alerts.append({
                'metric': 'Delta',
                'value': greeks_result['total_delta'],
                'threshold': 1000,
                'message': f"High delta exposure: {greeks_result['total_delta']:.2f}"
            })
        
        if abs(greeks_result['total_gamma']) > 100:
            alerts.append({
                'metric': 'Gamma',
                'value': greeks_result['total_gamma'],
                'threshold': 100,
                'message': f"High gamma exposure: {greeks_result['total_gamma']:.2f}"
            })
        
        if greeks_result['total_exposure'] > 500000:  # $500k
            alerts.append({
                'metric': 'Exposure',
                'value': greeks_result['total_exposure'],
                'threshold': 500000,
                'message': f"High portfolio exposure: ${greeks_result['total_exposure']:,.2f}"
            })
        
        # Send alerts
        for alert in alerts:
            if self.alert_callback:
                await self._safe_callback(self.alert_callback, alert)
            print(f"ALERT: {alert['message']}")
    
    async def _safe_callback(self, callback, data):
        """Safely call user callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            print(f"Error in user callback: {e}")
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if self.performance_stats['start_time']:
            runtime = time.time() - self.performance_stats['start_time']
            print(f"\n=== Performance Statistics ===")
            print(f"Runtime: {runtime:.1f} seconds")
            print(f"Data updates: {self.performance_stats['data_updates']}")
            print(f"Computations: {self.performance_stats['computations']}")
            print(f"Average latency: {self.performance_stats['avg_latency_ms']:.2f} ms")
            if runtime > 0:
                print(f"Update rate: {self.performance_stats['data_updates']/runtime:.2f} updates/sec")
    
    def get_current_greeks(self) -> Optional[Dict]:
        """Get the most recent Greeks computation"""
        return self.last_greeks
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        if stats['start_time']:
            stats['runtime'] = time.time() - stats['start_time']
        return stats


# Demo application
async def demo_realtime_system():
    """Demonstrate the real-time portfolio Greeks system"""
    
    print("Real-Time Portfolio Greeks Demo")
    print("==============================")
    
    # Create system
    system = RealTimePortfolioSystem(['AAPL', 'MSFT', 'GOOGL'], update_interval=30)
    
    # Add some positions
    system.add_position('AAPL', 'CALL', 190.0, '2024-01-19', 10, 5.50)
    system.add_position('AAPL', 'PUT', 185.0, '2024-01-19', -5, 3.20)
    system.add_position('MSFT', 'CALL', 380.0, '2024-01-19', 8, 12.80)
    system.add_position('GOOGL', 'CALL', 140.0, '2024-01-19', 15, 8.90)
    
    # Set callbacks
    async def greeks_update(greeks):
        print(f"\n=== Portfolio Update ===")
        print(f"Total Delta: {greeks['total_delta']:.2f}")
        print(f"Total Gamma: {greeks['total_gamma']:.2f}")
        print(f"Total Vega: {greeks['total_vega']:.2f}")
        print(f"Total Theta: {greeks['total_theta']:.2f}")
        print(f"Total Exposure: ${greeks['total_exposure']:,.2f}")
        print(f"Computation Time: {greeks['computation_time_ms']:.2f} ms")
        print(f"Positions: {greeks['num_positions']}")
    
    async def risk_alert(alert):
        print(f"🚨 RISK ALERT: {alert['message']}")
    
    system.set_greeks_callback(greeks_update)
    system.set_alert_callback(risk_alert)
    
    # Run demo
    try:
        await system.start()
        await asyncio.sleep(300)  # Run for 5 minutes
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        await system.stop()


if __name__ == "__main__":
    # Install required packages first
    print("Make sure to install required packages:")
    print("pip install yfinance pandas numpy scipy asyncio aiohttp")
    print()
    
    asyncio.run(demo_realtime_system())
