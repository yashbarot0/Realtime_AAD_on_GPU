#!/usr/bin/env python3
"""
Live Options Data Feed using yfinance and other free APIs
Supports real-time data streaming for portfolio Greeks computation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsDataFeed:
    """Real-time options data feed manager"""
    
    def __init__(self, symbols: List[str], update_interval: float = 1.0):
        self.symbols = symbols
        self.update_interval = update_interval
        self.current_data = {}
        self.is_running = False
        
    async def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """Fetch option chain for a symbol using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                logger.warning(f"No price data for {symbol}")
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            # Get option expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                logger.warning(f"No options data for {symbol}")
                return None
            
            # Use nearest expiration (typically weekly options)
            nearest_exp = exp_dates[0]
            
            # Get option chain
            opt_chain = ticker.option_chain(nearest_exp)
            
            # Process calls and puts
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Filter for liquid options (volume > 0, bid > 0)
            calls = calls[(calls['volume'] > 0) & (calls['bid'] > 0)]
            puts = puts[(puts['volume'] > 0) & (puts['bid'] > 0)]
            
            return {
                'symbol': symbol,
                'spot_price': float(current_price),
                'expiration': nearest_exp,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'calls': calls.to_dict('records'),
                'puts': puts.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return None
    
    async def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year treasury)"""
        try:
            # Use yfinance to get 10-year treasury rate
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1]) / 100.0  # Convert percentage to decimal
            else:
                logger.warning("Could not fetch risk-free rate, using default 5%")
                return 0.05
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            return 0.05  # Default to 5%
    
    async def update_data(self):
        """Update all symbols data"""
        logger.info("Updating options data...")
        
        # Get risk-free rate
        risk_free_rate = await self.get_risk_free_rate()
        
        # Fetch data for all symbols
        tasks = [self.fetch_option_chain(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error updating {symbol}: {result}")
                continue
                
            if result:
                result['risk_free_rate'] = risk_free_rate
                self.current_data[symbol] = result
                logger.info(f"Updated {symbol}: spot=${result['spot_price']:.2f}, "
                           f"calls={len(result['calls'])}, puts={len(result['puts'])}")
    
    async def start_streaming(self, callback=None):
        """Start the data streaming loop"""
        self.is_running = True
        logger.info(f"Starting options data stream for {self.symbols}")
        
        while self.is_running:
            try:
                await self.update_data()
                
                if callback:
                    await callback(self.current_data)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop_streaming(self):
        """Stop the data streaming"""
        self.is_running = False
        logger.info("Stopping options data stream")
    
    def get_current_data(self) -> Dict:
        """Get current market data"""
        return self.current_data.copy()


class PortfolioManager:
    """Manage options portfolio positions"""
    
    def __init__(self):
        self.positions = []
    
    def add_position(self, symbol: str, option_type: str, strike: float, 
                    expiration: str, quantity: int, premium_paid: float = 0.0):
        """Add an options position"""
        position = {
            'id': len(self.positions),
            'symbol': symbol,
            'option_type': option_type.upper(),  # 'CALL' or 'PUT'
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'premium_paid': premium_paid,
            'entry_time': datetime.now(timezone.utc).isoformat()
        }
        
        self.positions.append(position)
        logger.info(f"Added position: {quantity} {symbol} {strike} {option_type}")
        return position['id']
    
    def remove_position(self, position_id: int) -> bool:
        """Remove a position by ID"""
        for i, pos in enumerate(self.positions):
            if pos['id'] == position_id:
                self.positions.pop(i)
                logger.info(f"Removed position ID {position_id}")
                return True
        return False
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_symbols(self) -> List[str]:
        """Get unique symbols in portfolio"""
        return list(set(pos['symbol'] for pos in self.positions))


# Demo usage
async def demo_live_feed():
    """Demonstrate live options data feed"""
    
    # Create portfolio with some positions
    portfolio = PortfolioManager()
    portfolio.add_position('AAPL', 'CALL', 190.0, '2024-01-19', 10)
    portfolio.add_position('AAPL', 'PUT', 185.0, '2024-01-19', -5)
    portfolio.add_position('MSFT', 'CALL', 380.0, '2024-01-19', 8)
    
    # Get symbols from portfolio
    symbols = portfolio.get_symbols()
    
    # Create data feed
    feed = OptionsDataFeed(symbols, update_interval=30.0)  # Update every 30 seconds
    
    async def data_callback(data):
        """Process updated market data"""
        print(f"\n=== Market Update at {datetime.now()} ===")
        for symbol, market_data in data.items():
            print(f"{symbol}: ${market_data['spot_price']:.2f} "
                  f"(Risk-free: {market_data['risk_free_rate']:.2%})")
            print(f"  Calls: {len(market_data['calls'])}, Puts: {len(market_data['puts'])}")
    
    # Start streaming (run for demo)
    try:
        await asyncio.wait_for(feed.start_streaming(data_callback), timeout=300)  # 5 minutes demo
    except asyncio.TimeoutError:
        print("Demo completed")
    finally:
        feed.stop_streaming()


if __name__ == "__main__":
    print("Live Options Data Feed Demo")
    print("==========================")
    asyncio.run(demo_live_feed())
