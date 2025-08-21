import yfinance as yf
import requests
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

@dataclass
class OptionContract:
    """Live option contract data"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    timestamp: datetime

@dataclass
class MarketData:
    """Current market data for underlying"""
    symbol: str
    spot_price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime

class LiveOptionsDataFetcher:
    def __init__(self):
        self.setup_logging()
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Multiple data sources for redundancy
        self.data_sources = {
            'yfinance': self._fetch_yfinance_data,
            'alpha_vantage': self._fetch_alpha_vantage_data,
            'marketdata': self._fetch_marketdata_api
        }
        
        # API keys (get free keys from respective providers)
        self.alpha_vantage_key = "YOUR_ALPHA_VANTAGE_KEY"
        self.marketdata_key = "YOUR_MARKETDATA_KEY"  # 100 free requests/day
        
        self.active_symbols = set()
        self.portfolio_positions = {}
        
    def setup_logging(self):
        """Setup logging for monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_options.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _fetch_yfinance_data(self, symbol: str) -> Dict:
        """Fetch options data using yFinance"""
        try:
            self._respect_rate_limit()
            
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            # Get options expirations
            expirations = ticker.options
            if not expirations:
                return None
            
            # Focus on nearest expiration for real-time trading
            nearest_expiry = expirations[0]
            
            # Get options chain
            opt_chain = ticker.option_chain(nearest_expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Process calls and puts
            options_data = []
            
            # Process calls
            for _, row in calls.iterrows():
                if pd.notna(row['lastPrice']) and row['volume'] > 0:
                    option = OptionContract(
                        symbol=symbol,
                        strike=row['strike'],
                        expiry=nearest_expiry,
                        option_type='call',
                        bid=row['bid'] if pd.notna(row['bid']) else 0,
                        ask=row['ask'] if pd.notna(row['ask']) else 0,
                        last=row['lastPrice'],
                        volume=row['volume'] if pd.notna(row['volume']) else 0,
                        open_interest=row['openInterest'] if pd.notna(row['openInterest']) else 0,
                        implied_volatility=row['impliedVolatility'] if pd.notna(row['impliedVolatility']) else 0.2,
                        timestamp=datetime.now()
                    )
                    options_data.append(option)
            
            # Process puts
            for _, row in puts.iterrows():
                if pd.notna(row['lastPrice']) and row['volume'] > 0:
                    option = OptionContract(
                        symbol=symbol,
                        strike=row['strike'],
                        expiry=nearest_expiry,
                        option_type='put',
                        bid=row['bid'] if pd.notna(row['bid']) else 0,
                        ask=row['ask'] if pd.notna(row['ask']) else 0,
                        last=row['lastPrice'],
                        volume=row['volume'] if pd.notna(row['volume']) else 0,
                        open_interest=row['openInterest'] if pd.notna(row['openInterest']) else 0,
                        implied_volatility=row['impliedVolatility'] if pd.notna(row['impliedVolatility']) else 0.2,
                        timestamp=datetime.now()
                    )
                    options_data.append(option)
            
            market_data = MarketData(
                symbol=symbol,
                spot_price=current_price,
                bid=current_price,  # yFinance doesn't provide bid/ask easily
                ask=current_price,
                volume=int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                timestamp=datetime.now()
            )
            
            return {
                'market_data': market_data,
                'options': options_data,
                'source': 'yfinance'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching yFinance data for {symbol}: {e}")
            return None
    
    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Alpha Vantage (backup source)"""
        try:
            # Alpha Vantage has limited free options data
            # Mainly for stock prices
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            self._respect_rate_limit()
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                market_data = MarketData(
                    symbol=symbol,
                    spot_price=float(quote['05. price']),
                    bid=float(quote['05. price']) * 0.999,  # Approximate bid/ask
                    ask=float(quote['05. price']) * 1.001,
                    volume=int(quote['06. volume']),
                    timestamp=datetime.now()
                )
                
                return {
                    'market_data': market_data,
                    'options': [],  # Alpha Vantage free tier doesn't have options
                    'source': 'alpha_vantage'
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    def _fetch_marketdata_api(self, symbol: str) -> Optional[Dict]:
        """Fetch from MarketData.app (100 free requests/day)"""
        try:
            # Get current stock price
            self._respect_rate_limit()
            stock_url = f"https://api.marketdata.app/v1/stocks/quotes/{symbol}/"
            headers = {'Authorization': f'Token {self.marketdata_key}'} if self.marketdata_key != "YOUR_MARKETDATA_KEY" else {}
            
            response = requests.get(stock_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
                
            stock_data = response.json()
            if stock_data['s'] != 'ok':
                return None
            
            # Get options chain
            self._respect_rate_limit()
            options_url = f"https://api.marketdata.app/v1/options/chain/{symbol}/"
            
            response = requests.get(options_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
                
            options_data = response.json()
            
            # Process the data (MarketData.app specific format)
            # This is a simplified version - you'd need to adapt based on their exact API response
            
            return None  # Placeholder - implement based on MarketData.app API docs
            
        except Exception as e:
            self.logger.error(f"Error fetching MarketData.app data for {symbol}: {e}")
            return None
    
    def fetch_live_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch live data for multiple symbols"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Try multiple sources for redundancy
            futures = {}
            
            for symbol in symbols:
                # Try yFinance first (most reliable for free data)
                future = executor.submit(self._fetch_yfinance_data, symbol)
                futures[future] = (symbol, 'yfinance')
            
            # Collect results
            for future in futures:
                symbol, source = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        results[symbol] = result
                        self.logger.info(f"Successfully fetched {symbol} from {source}")
                    else:
                        self.logger.warning(f"No data for {symbol} from {source}")
                except Exception as e:
                    self.logger.error(f"Error fetching {symbol} from {source}: {e}")
        
        return results
    
    def get_portfolio_positions(self) -> Dict:
        """Get current portfolio positions (placeholder)"""
        # In a real system, this would connect to your broker API
        # For demo purposes, we'll create some sample positions
        return {
            'AAPL': {'quantity': 100, 'entry_price': 150.0},
            'MSFT': {'quantity': 50, 'entry_price': 300.0},
            'GOOGL': {'quantity': 25, 'entry_price': 120.0}
        }

async def main():
    """Test the live data fetcher"""
    fetcher = LiveOptionsDataFetcher()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"Fetching live data at {datetime.now()}")
            print(f"{'='*60}")
            
            # Fetch live data
            data = fetcher.fetch_live_data(symbols)
            
            for symbol, symbol_data in data.items():
                market_data = symbol_data['market_data']
                options = symbol_data['options']
                
                print(f"\n{symbol}:")
                print(f"  Spot: ${market_data.spot_price:.2f}")
                print(f"  Options available: {len(options)}")
                
                if options:
                    calls = [opt for opt in options if opt.option_type == 'call']
                    puts = [opt for opt in options if opt.option_type == 'put']
                    print(f"  Calls: {len(calls)}, Puts: {len(puts)}")
                    
                    # Show ATM options
                    atm_calls = [opt for opt in calls if abs(opt.strike - market_data.spot_price) <= 5]
                    atm_puts = [opt for opt in puts if abs(opt.strike - market_data.spot_price) <= 5]
                    
                    if atm_calls:
                        best_call = min(atm_calls, key=lambda x: abs(x.strike - market_data.spot_price))
                        print(f"  ATM Call: ${best_call.strike} -> ${best_call.last:.2f} (IV: {best_call.implied_volatility:.1%})")
                    
                    if atm_puts:
                        best_put = min(atm_puts, key=lambda x: abs(x.strike - market_data.spot_price))
                        print(f"  ATM Put: ${best_put.strike} -> ${best_put.last:.2f} (IV: {best_put.implied_volatility:.1%})")
            
            # Wait before next update (respect rate limits)
            print(f"\nWaiting 60 seconds for next update...")
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            print("\nStopping live data feed...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
