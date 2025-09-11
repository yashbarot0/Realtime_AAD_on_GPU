import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class AsyncMarketDataFetcher:
    def __init__(self, symbols):
        self.symbols = symbols
        self.price_cache = {}
        self.last_update = {}
    
    async def fetch_single_price(self, symbol):
        """Fetch latest price for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            # Get recent data asynchronously
            hist = await asyncio.to_thread(
                ticker.history, 
                period="1d", 
                interval="1m"
            )
            
            if not hist.empty:
                latest_price = float(hist['Close'].iloc[-1])
                self.price_cache[symbol] = latest_price
                self.last_update[symbol] = datetime.now()
                return symbol, latest_price
            else:
                # Use cached price if available
                return symbol, self.price_cache.get(symbol, 100.0)
                
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            # Return cached price or default
            return symbol, self.price_cache.get(symbol, 100.0)
    
    async def fetch_all_prices(self):
        """Fetch all symbols' prices concurrently"""
        start_time = asyncio.get_event_loop().time()
        
        # Create concurrent tasks
        tasks = [
            asyncio.create_task(self.fetch_single_price(symbol)) 
            for symbol in self.symbols
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        price_dict = {}
        for result in results:
            if isinstance(result, tuple):
                symbol, price = result
                price_dict[symbol] = price
            else:
                print(f"Exception in fetch: {result}")
        
        fetch_time = asyncio.get_event_loop().time() - start_time
        
        return price_dict, fetch_time
    
    def generate_option_parameters(self, price_dict, options_per_symbol=10):
        """Generate option parameters for portfolio"""
        options = []
        
        for symbol in self.symbols:
            spot = price_dict.get(symbol, 100.0)
            
            # Generate diverse option strikes around current price
            for i in range(options_per_symbol):
                strike_multiplier = 0.85 + (i / (options_per_symbol - 1)) * 0.30  # 85% to 115%
                strike = spot * strike_multiplier
                
                # Vary time to expiration (1 day to 30 days)
                time_to_expiry = (1 + (i % 30)) / 365.0
                
                # Vary volatility slightly
                base_vol = 0.25
                vol_adjustment = (i % 5 - 2) * 0.02  # ±4% volatility
                volatility = max(0.1, base_vol + vol_adjustment)
                
                option = {
                    'symbol': symbol,
                    'spot': spot,
                    'strike': strike,
                    'time': time_to_expiry,
                    'rate': 0.02,  # 2% risk-free rate
                    'volatility': volatility,
                    'is_call': i % 2 == 0  # Alternate calls and puts
                }
                options.append(option)
        
        return options
