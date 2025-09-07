import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor

# Import WORKING components (using correct names)
from safe_gpu_interface import SafeGPUInterface
from live_options_fetcher import LiveOptionsDataFetcher

class RealtimePortfolioSystemMaximized:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with CORRECT class names
        self.data_fetcher = LiveOptionsDataFetcher()
        self.gpu_interface = SafeGPUInterface()
        
        # 🚀 MAXIMIZED Configuration for 16GB GPU
        self.tracked_symbols = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            # Financial & Industrial  
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
            # Consumer & Healthcare
            'PG', 'JNJ', 'UNH', 'PFE', 'MRK', 'KO', 'PEP', 'WMT',
            # Energy & Materials
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'FCX', 'NEM', 'AA',
            # ETFs for diversification
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT', 'XLF', 'XLE',
            # Additional symbols for maximum scale
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'CSCO', 'IBM'
        ]  # 48 symbols total
        
        # 🚀 MAXIMUM SCALE Settings
        self.synthetic_options_per_symbol = 800  # 800-1200 per symbol
        self.max_total_options = 25000           # Target 25K options
        self.batch_size = 2500                   # Larger batches for efficiency
        self.update_interval = 1.0               # 1-second updates for demo
        
        # 🚀 PERFORMANCE MODES
        self.use_synthetic_data = True           # Enable for maximum GPU showcase
        self.use_cached_data = False             # Cache real data when available
        self.cache_duration = 60                 # Cache for 1 minute
        
        # GPU Memory Management
        self.gpu_memory_buffer = 0.90            # Use 90% of GPU memory
        self.prealloc_memory = True
        
        self.running = False
        self.options_cache = {}
        self.last_fetch_time = 0
        
        # Enhanced Stats tracking
        self.stats = {
            'updates': 0,
            'successful_updates': 0,
            'total_processed': 0,
            'avg_time': 0.0,
            'gpu_memory_used': 0.0,
            'max_concurrent_options': 0,
            'total_symbols_tracked': len(self.tracked_symbols),
            'avg_options_per_symbol': 0,
            'max_throughput': 0,
            'synthetic_data_used': 0
        }

    def generate_synthetic_options_data(self):
        """🚀 Generate massive synthetic options dataset for maximum GPU showcase"""
        print(f"🎯 Generating synthetic data for maximum GPU utilization...")
        start_time = time.time()
        
        # Base prices for realistic synthetic data
        base_prices = {
            'AAPL': 229.30, 'MSFT': 501.93, 'GOOGL': 207.19, 'AMZN': 228.72,
            'TSLA': 351.77, 'NVDA': 181.70, 'META': 245.32, 'NFLX': 456.78,
            'JPM': 298.52, 'BAC': 50.23, 'WFC': 81.52, 'GS': 749.16,
            'MS': 148.96, 'C': 95.71, 'V': 312.45, 'MA': 589.23,
            'PG': 167.89, 'JNJ': 145.67, 'UNH': 623.45, 'PFE': 25.34,
            'MRK': 134.56, 'KO': 67.89, 'PEP': 178.90, 'WMT': 198.76,
            'XOM': 134.56, 'CVX': 178.90, 'COP': 145.67, 'EOG': 156.78,
            'SLB': 67.89, 'FCX': 45.67, 'NEM': 78.90, 'AA': 23.45,
            'SPY': 645.13, 'QQQ': 572.66, 'IWM': 234.26, 'VTI': 318.06,
            'GLD': 245.67, 'TLT': 89.45, 'XLF': 45.67, 'XLE': 78.90,
            'ADBE': 567.89, 'CRM': 234.56, 'ORCL': 145.67, 'INTC': 56.78,
            'AMD': 178.90, 'QCOM': 189.45, 'CSCO': 67.89, 'IBM': 234.56
        }
        
        synthetic_data = {}
        total_options = 0
        
        for symbol in self.tracked_symbols:
            spot_price = base_prices.get(symbol, np.random.uniform(50, 500))
            
            # Generate 600-1000 options per symbol for massive dataset
            num_options = np.random.randint(600, 1000)
            options_list = []
            
            # Generate multiple expiry dates
            expiry_dates = ['2025-01-17', '2025-02-21', '2025-03-21', '2025-04-18', 
                          '2025-06-20', '2025-09-19', '2025-12-19']
            
            for i in range(num_options):
                # 🔧 FIXED: Simple random selection for moneyness
                rand_choice = np.random.random()
                if rand_choice < 0.6:  # 60% near-the-money
                    moneyness = np.random.uniform(0.8, 1.2)
                elif rand_choice < 0.8:  # 20% OTM puts  
                    moneyness = np.random.uniform(0.6, 0.8)
                else:  # 20% OTM calls
                    moneyness = np.random.uniform(1.2, 1.5)
                
                strike = round(spot_price * moneyness, 2)
                
                # 🔧 FIXED: Simple random selection for volume tiers
                vol_choice = np.random.random()
                if vol_choice < 0.7:  # 70% low volume
                    volume = np.random.randint(1, 50)
                elif vol_choice < 0.95:  # 25% medium volume
                    volume = np.random.randint(51, 500)
                else:  # 5% high volume
                    volume = np.random.randint(501, 2000)
                
                open_interest = np.random.randint(0, volume * 10)
                
                # IV based on moneyness (more realistic)
                if 0.9 <= moneyness <= 1.1:  # ATM higher IV
                    iv = np.random.uniform(0.25, 0.6)
                else:  # OTM lower IV
                    iv = np.random.uniform(0.15, 0.4)
                
                # Option type distribution (slightly more calls)
                option_type = 'call' if np.random.random() < 0.55 else 'put'
                
                # Realistic pricing based on moneyness
                if option_type == 'call':
                    if moneyness < 1.0:  # ITM calls
                        price = np.random.uniform(spot_price * (1-moneyness) + 0.5, 
                                                spot_price * (1-moneyness) + 5.0)
                    else:  # OTM calls
                        price = np.random.uniform(0.1, spot_price * 0.05)
                else:  # puts
                    if moneyness > 1.0:  # ITM puts
                        price = np.random.uniform(spot_price * (moneyness-1) + 0.5,
                                                spot_price * (moneyness-1) + 5.0)
                    else:  # OTM puts
                        price = np.random.uniform(0.1, spot_price * 0.05)
                
                options_list.append({
                    'strike': strike,
                    'volume': volume,
                    'open_interest': open_interest,
                    'expiry': expiry_dates[np.random.randint(0, len(expiry_dates))],
                    'implied_volatility': iv,
                    'option_type': option_type,
                    'last_price': max(0.01, round(price, 2))
                })
            
            synthetic_data[symbol] = {
                'spot_price': spot_price,
                'options': options_list
            }
            total_options += num_options
        
        generation_time = time.time() - start_time
        print(f"🎯 Generated {total_options:,} synthetic options across {len(self.tracked_symbols)} symbols in {generation_time:.2f}s")
        self.stats['synthetic_data_used'] = total_options
        
        return synthetic_data


    def generate_biased_synthetic_options_data(self):
       """🎯 Generate synthetic data with INTENTIONAL Greek bias for testing"""
       print(f"🎯 Generating BIASED synthetic data to ensure non-zero Greeks...")
       start_time = time.time()
       
       base_prices = {
           'AAPL': 229.30, 'MSFT': 501.93, 'GOOGL': 207.19, 'AMZN': 228.72,
           'TSLA': 351.77, 'NVDA': 181.70, 'META': 245.32, 'NFLX': 456.78,
           'JPM': 298.52, 'BAC': 50.23, 'WFC': 81.52, 'GS': 749.16,
           'MS': 148.96, 'C': 95.71, 'V': 312.45, 'MA': 589.23,
           'PG': 167.89, 'JNJ': 145.67, 'UNH': 623.45, 'PFE': 25.34,
           'MRK': 134.56, 'KO': 67.89, 'PEP': 178.90, 'WMT': 198.76,
           'XOM': 134.56, 'CVX': 178.90, 'COP': 145.67, 'EOG': 156.78,
           'SLB': 67.89, 'FCX': 45.67, 'NEM': 78.90, 'AA': 23.45,
           'SPY': 645.13, 'QQQ': 572.66, 'IWM': 234.26, 'VTI': 318.06,
           'GLD': 245.67, 'TLT': 89.45, 'XLF': 45.67, 'XLE': 78.90,
           'ADBE': 567.89, 'CRM': 234.56, 'ORCL': 145.67, 'INTC': 56.78,
           'AMD': 178.90, 'QCOM': 189.45, 'CSCO': 67.89, 'IBM': 234.56
       }
       
       synthetic_data = {}
       total_options = 0
       
       # 🔧 BIAS SETTINGS for non-zero Greeks
       call_bias = 0.7  # 70% calls, 30% puts for net positive delta
       
       for i, symbol in enumerate(self.tracked_symbols):
           spot_price = base_prices.get(symbol, np.random.uniform(50, 500))
           num_options = np.random.randint(600, 1000)
           options_list = []
           
           # Create directional bias per symbol
           symbol_bias = 1 if i < len(self.tracked_symbols)//2 else -1  # Half bullish, half bearish
           
           expiry_dates = ['2025-01-17', '2025-02-21', '2025-03-21', '2025-06-20']
           
           for j in range(num_options):
               # 🔧 BIASED moneyness to create Greeks
               if symbol_bias > 0:  # Bullish bias
                   if np.random.random() < 0.6:
                       moneyness = np.random.uniform(0.85, 1.05)  # Near ATM
                   else:
                       moneyness = np.random.uniform(1.05, 1.3)   # OTM calls
               else:  # Bearish bias
                   if np.random.random() < 0.6:
                       moneyness = np.random.uniform(0.95, 1.15)  # Near ATM  
                   else:
                       moneyness = np.random.uniform(0.7, 0.95)   # OTM puts
                       
               strike = round(spot_price * moneyness, 2)
               
               # 🔧 BIASED option type selection
               if symbol_bias > 0:
                   option_type = 'call' if np.random.random() < call_bias else 'put'
               else:
                   option_type = 'put' if np.random.random() < call_bias else 'call'
               
               # Higher IV for more pronounced Greeks
               iv = np.random.uniform(0.3, 0.8)  # Higher IV range
               
               # Longer time to expiry for more time decay
               expiry = expiry_dates[np.random.randint(0, len(expiry_dates))]
               
               volume = np.random.randint(10, 1000)  # Higher minimum volume
               
               # Realistic pricing
               intrinsic = max(0, (spot_price - strike) if option_type == 'call' 
                             else max(0, strike - spot_price))
               extrinsic = np.random.uniform(1, spot_price * iv * 0.1)
               price = intrinsic + extrinsic
               
               options_list.append({
                   'strike': strike,
                   'volume': volume,
                   'open_interest': np.random.randint(volume, volume * 5),
                   'expiry': expiry,
                   'implied_volatility': iv,
                   'option_type': option_type,
                   'last_price': max(0.01, round(price, 2))
               })
           
           synthetic_data[symbol] = {
               'spot_price': spot_price,
               'options': options_list
           }
           total_options += num_options
       
       generation_time = time.time() - start_time
       print(f"🎯 Generated {total_options:,} BIASED synthetic options in {generation_time:.2f}s")
       print(f"🔧 Applied directional bias: 50% bullish symbols, 50% bearish symbols")
       self.stats['synthetic_data_used'] = total_options
       
       return synthetic_data



    async def fetch_with_optimization(self):
        """🚀 Optimized fetching with caching and synthetic data options"""
        current_time = time.time()
        
        # Mode 1: Use synthetic data for maximum GPU showcase
        if self.use_synthetic_data:
            return self.generate_synthetic_options_data()
        
        # Mode 2: Use cached real data if available and recent
        if (self.use_cached_data and hasattr(self, 'last_fetch_time') and 
            current_time - self.last_fetch_time < self.cache_duration and
            self.options_cache):
            print(f"📦 Using cached data ({sum(len(d.get('options', [])) for d in self.options_cache.values()):,} options)")
            return self.options_cache
        
        # Mode 3: Fetch fresh real data
        live_data = await self.fetch_all_symbols_optimized()
        if live_data:
            self.options_cache = live_data
            self.last_fetch_time = current_time
        
        return live_data

    async def fetch_all_symbols_optimized(self):
        """🚀 OPTIMIZED: Maximize parallel fetching for 48 symbols"""
        fetch_start = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            # 🚀 MAXIMUM PARALLELISM: 15 workers for 48 symbols
            with ThreadPoolExecutor(max_workers=15) as executor:
                # Create smaller batches for better load distribution
                symbol_batches = [
                    self.tracked_symbols[i:i+6] 
                    for i in range(0, len(self.tracked_symbols), 6)
                ]
                
                tasks = []
                for batch in symbol_batches:
                    task = loop.run_in_executor(
                        executor,
                        lambda b=batch: self.data_fetcher.fetch_live_data(b)
                    )
                    tasks.append(task)
                
                # Execute all batch tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine all batch results
                live_data = {}
                successful_symbols = 0
                for result in results:
                    if not isinstance(result, Exception) and result:
                        live_data.update(result)
                        successful_symbols += len(result)
                
                fetch_time = time.time() - fetch_start
                print(f"⚡ Optimized parallel fetch: {fetch_time*1000:.1f}ms "
                      f"({successful_symbols}/{len(self.tracked_symbols)} symbols)")
                
                return live_data if live_data else None
                
        except Exception as e:
            self.logger.error(f"Optimized fetch failed: {e}")
            return None

    def time_to_expiry(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            expiry_dt = pd.to_datetime(expiry_str)
            now = pd.Timestamp.now()
            delta = (expiry_dt - now).total_seconds() / (365.25 * 24 * 3600)
            return max(delta, 0.001)  # Minimum 1 day
        except Exception:
            return 0.25  # Default 3 months

    def prepare_maximum_options_data(self, live_data):
        """🚀 MAXIMIZED: Process ALL options without limits for GPU showcase"""
        options = []
        market_data = {}
        processing_start = time.time()
        
        print(f"🚀 Processing data from {len(live_data)} symbols without limits...")
        
        for symbol, data in live_data.items():
            try:
                # Extract data efficiently
                if isinstance(data, dict):
                    spot_price = data.get('spot_price', 100.0)
                    options_list = data.get('options', [])
                else:
                    spot_price = getattr(data, 'spot_price', 100.0)
                    options_list = getattr(data, 'options', [])
                
                market_data[symbol] = {'spot_price': float(spot_price)}
                
                # 🚀 PROCESS ALL OPTIONS (no filtering or limits)
                processed_count = 0
                for opt in options_list:
                    try:
                        if isinstance(opt, dict):
                            strike = float(opt.get('strike', 0))
                            volume = float(opt.get('volume', 0))
                            oi = float(opt.get('open_interest', 0))
                            expiry = opt.get('expiry', '2025-01-17')
                            iv = float(opt.get('implied_volatility', 0.25))
                            opt_type = opt.get('option_type', opt.get('type', 'call'))
                            last_price = float(opt.get('last_price', opt.get('last', 1.0)))
                        else:
                            strike = float(getattr(opt, 'strike', 0))
                            volume = float(getattr(opt, 'volume', 0))
                            oi = float(getattr(opt, 'open_interest', 0))
                            expiry = getattr(opt, 'expiry', '2025-01-17')
                            iv = float(getattr(opt, 'implied_volatility', 0.25))
                            opt_type = getattr(opt, 'option_type', 'call')
                            last_price = float(getattr(opt, 'last', 1.0))
                        
                        # Minimal validation - keep almost all options
                        spot_val = float(spot_price)
                        if strike > 0 and spot_val > 0 and iv > 0:
                            options.append({
                                'symbol': symbol,
                                'strike': strike,
                                'spot_price': spot_val,
                                'time_to_expiry': self.time_to_expiry(expiry),
                                'risk_free_rate': 0.05,
                                'implied_volatility': max(iv, 0.01),
                                'is_call': str(opt_type).lower() == 'call',
                                'market_price': max(last_price, 0.01),
                                'volume': volume,
                                'open_interest': oi,
                                'liquidity_score': volume + 0.1 * oi
                            })
                            processed_count += 1
                    except (ValueError, TypeError):
                        continue
                
                print(f"  {symbol}: {processed_count:,} options processed")
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                market_data[symbol] = {'spot_price': 0.0}
        
        processing_time = time.time() - processing_start
        print(f"🔄 Maximum processing: {processing_time*1000:.1f}ms "
              f"({len(options):,} total options from {len(live_data)} symbols)")
        
        return options, market_data

    def print_maximized_status(self, live_data, processed_count, elapsed_time, greeks, market_data):
        """Display comprehensive system status optimized for large scale"""
        
        print(f"\n📈 MAXIMIZED MARKET DATA ({len(live_data)} symbols, {processed_count:,} options):")
        
        # Compact display for large number of symbols
        categories = [
            ("MEGA-TECH", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']),
            ("FINANCE", ['JPM', 'BAC', 'GS', 'V', 'MA', 'C']),
            ("ETFs", ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT']),
            ("CLOUD/AI", ['META', 'NFLX', 'ADBE', 'CRM', 'AMD', 'QCOM'])
        ]
        
        for category, symbols in categories:
            print(f"\n{category}:")
            symbols_in_row = 0
            for symbol in symbols:
                if symbol in live_data and symbols_in_row < 6:
                    spot = market_data.get(symbol, {}).get('spot_price', 0)
                    print(f"  {symbol}: ${spot:>6.1f}", end="")
                    symbols_in_row += 1
                    if symbols_in_row == 3:
                        print()
                        symbols_in_row = 0
            if symbols_in_row > 0:
                print()
        
        # 🔧 FIXED: Enhanced Greeks display with debugging
        print(f"\n💰 PORTFOLIO GREEKS (Real-time AAD on {processed_count:,} options):")
        
        try:
            # Debug: Show what we're working with
            print(f"🔍 DEBUG: Greeks object type: {type(greeks)}")
            
            # Multiple extraction strategies
            delta = vega = gamma = theta = rho = pnl = 0.0
            
            if hasattr(greeks, '__dict__'):
                attrs = list(greeks.__dict__.keys())
                print(f"🔍 DEBUG: Available attributes: {attrs}")
                
                # Try different attribute names
                for attr_name in ['total_delta', 'delta', 'Delta']:
                    if hasattr(greeks, attr_name):
                        delta = float(getattr(greeks, attr_name, 0.0))
                        break
                        
                for attr_name in ['total_vega', 'vega', 'Vega']:
                    if hasattr(greeks, attr_name):
                        vega = float(getattr(greeks, attr_name, 0.0))
                        break
                        
                for attr_name in ['total_gamma', 'gamma', 'Gamma']:
                    if hasattr(greeks, attr_name):
                        gamma = float(getattr(greeks, attr_name, 0.0))
                        break
                        
                for attr_name in ['total_theta', 'theta', 'Theta']:
                    if hasattr(greeks, attr_name):
                        theta = float(getattr(greeks, attr_name, 0.0))
                        break
                        
                for attr_name in ['total_rho', 'rho', 'Rho']:
                    if hasattr(greeks, attr_name):
                        rho = float(getattr(greeks, attr_name, 0.0))
                        break
                        
                for attr_name in ['total_pnl', 'pnl', 'PnL', 'unrealized_pnl']:
                    if hasattr(greeks, attr_name):
                        pnl = float(getattr(greeks, attr_name, 0.0))
                        break
                    
            elif isinstance(greeks, dict):
                print(f"🔍 DEBUG: Dictionary keys: {list(greeks.keys())}")
                delta = float(greeks.get('total_delta', greeks.get('delta', 0.0)))
                vega = float(greeks.get('total_vega', greeks.get('vega', 0.0)))
                gamma = float(greeks.get('total_gamma', greeks.get('gamma', 0.0)))
                theta = float(greeks.get('total_theta', greeks.get('theta', 0.0)))
                rho = float(greeks.get('total_rho', greeks.get('rho', 0.0)))
                pnl = float(greeks.get('total_pnl', greeks.get('pnl', 0.0)))
            
            # Debug: Show raw values
            print(f"🔍 DEBUG: Raw values - Delta: {delta}, Vega: {vega}, Gamma: {gamma}")
            print(f"🔍 DEBUG: Raw values - Theta: {theta}, Rho: {rho}, PnL: {pnl}")
            
            # Smart formatting function
            def smart_format(value, precision=3):
                """Format numbers with appropriate precision based on magnitude"""
                if abs(value) < 1e-10:  # Essentially zero
                    return "0.000"
                elif abs(value) >= 1000:
                    return f"{value:,.1f}"
                elif abs(value) >= 1:
                    return f"{value:.3f}"
                elif abs(value) >= 0.001:
                    return f"{value:.6f}"
                else:
                    return f"{value:.2e}"
            
            # Display with improved formatting
            print(f"  Delta: {smart_format(delta):>12s}  |  Vega: {smart_format(vega):>12s}")
            print(f"  Gamma: {smart_format(gamma):>12s}  |  Theta: {smart_format(theta):>12s}")
            print(f"  Rho: {smart_format(rho):>12s}    |  P&L: ${pnl:>15,.2f}")
            
            # Enhanced analysis for large portfolios
            print(f"\n🔍 GREEKS ANALYSIS:")
            
            if abs(delta) > 0.001:
                delta_exposure = abs(delta) * 100  # Rough dollar exposure per $1 move
                print(f"  Portfolio Delta Exposure: ~${delta_exposure:,.0f} per $1 underlying move")
            else:
                print(f"  Portfolio Delta: Near zero (delta-neutral)")
                
            if abs(gamma) > 0.000001:
                print(f"  Gamma Risk: {smart_format(gamma)} (delta changes per $1 move)")
            else:
                print(f"  Gamma Risk: Minimal")
                
            if abs(theta) > 0.001:
                daily_theta = theta * 365.25  # Annualized to daily
                print(f"  Time Decay: ${daily_theta:,.0f} per day (estimated)")
            else:
                print(f"  Time Decay: Minimal")
            
            # Portfolio size context
            avg_greeks_per_option = {
                'delta': delta / processed_count if processed_count > 0 else 0,
                'gamma': gamma / processed_count if processed_count > 0 else 0,
                'vega': vega / processed_count if processed_count > 0 else 0
            }
            
            print(f"\n📊 AVERAGE PER OPTION:")
            print(f"  Avg Delta/Option: {smart_format(avg_greeks_per_option['delta'])}")
            print(f"  Avg Gamma/Option: {smart_format(avg_greeks_per_option['gamma'])}")
            print(f"  Avg Vega/Option: {smart_format(avg_greeks_per_option['vega'])}")
            
        except Exception as e:
            print(f"❌ Error displaying Greeks: {e}")
            print(f"🔍 Greeks object: {greeks}")
            import traceback
            traceback.print_exc()
        
        # Calculate current throughput
        current_throughput = processed_count / elapsed_time if elapsed_time > 0 else 0
        self.stats['max_throughput'] = max(self.stats['max_throughput'], current_throughput)
        
        print(f"\n🚀 MAXIMIZED GPU PERFORMANCE METRICS:")
        print(f"  Processing Time: {elapsed_time*1000:>10.1f} ms")
        print(f"  Options Processed: {processed_count:>10,d}")
        print(f"  Max Concurrent: {self.stats['max_concurrent_options']:>10,d}")
        print(f"  GPU Memory Used: {self.stats['gpu_memory_used']:>10.1f}%")
        print(f"  Current Throughput: {current_throughput:>10,.0f} options/sec")
        print(f"  Peak Throughput: {self.stats['max_throughput']:>10,.0f} options/sec")
        print(f"  Success Rate: {self.stats['successful_updates']/max(1,self.stats['updates'])*100:>10.1f}%")
        print(f"  Symbols Active: {len(live_data):>10d}/{self.stats['total_symbols_tracked']}")
        print(f"  Data Mode: {'SYNTHETIC' if self.use_synthetic_data else 'LIVE':>10s}")
        print(f"  Compute Method: {'GPU' if self.gpu_interface.use_gpu else 'CPU':>10s}")
    
    
    async def update_cycle_maximized(self):
        """🚀 MAXIMIZED: Execute optimized update cycle for maximum GPU utilization"""
        start_time = time.time()
        
        try:
            self.logger.info("Fetching maximized dataset...")
            
            # 🚀 GET MAXIMUM DATASET
            live_data = await self.fetch_with_optimization()
            if not live_data:
                self.logger.warning("No data received")
                return False
            
            # Process ALL available options without limits
            gpu_start = time.time()
            options_data, market_data = self.prepare_maximum_options_data(live_data)
            
            if not options_data:
                self.logger.warning("No valid options data to process")
                return False
            
            # 🚀 SMART BATCH PROCESSING for maximum GPU utilization
            total_processed = 0
            options_count = len(options_data)
            
            # Dynamically adjust batch size based on options count
            if options_count > 10000:
                batch_size = 2000  # Smaller batches for very large datasets
            elif options_count > 5000:
                batch_size = 2500  # Medium batches
            else:
                batch_size = options_count  # Single batch for smaller datasets
            
            num_batches = max(1, (options_count + batch_size - 1) // batch_size)
            
            print(f"🚀 Processing {options_count:,} options in {num_batches} batch(es) of ~{batch_size:,}")
            
            for i in range(0, options_count, batch_size):
                batch = options_data[i:i + batch_size]
                batch_start = time.time()
                
                batch_count = self.gpu_interface.process_portfolio_options(batch, market_data)
                
                batch_time = time.time() - batch_start
                total_processed += batch_count
                
                batch_num = (i // batch_size) + 1
                print(f"🚀 Batch {batch_num}/{num_batches}: {batch_count:,} options in {batch_time*1000:.1f}ms")
                
                # Monitor GPU memory usage if available
                if hasattr(self.gpu_interface, 'get_gpu_memory_usage'):
                    try:
                        gpu_memory = self.gpu_interface.get_gpu_memory_usage()
                        self.stats['gpu_memory_used'] = gpu_memory
                    except:
                        pass
            
            # Track maximum concurrent options processed
            self.stats['max_concurrent_options'] = max(
                self.stats['max_concurrent_options'], 
                options_count
            )
            
            greeks = self.gpu_interface.get_portfolio_greeks()
            gpu_total_time = time.time() - gpu_start
            
            print(f"🚀 TOTAL GPU PROCESSING: {gpu_total_time*1000:.1f}ms "
                  f"({total_processed:,} options in {num_batches} batches)")
            
            # Update comprehensive stats
            elapsed_time = time.time() - start_time
            self.stats['updates'] += 1
            self.stats['successful_updates'] += 1
            self.stats['total_processed'] += total_processed
            self.stats['avg_time'] = (
                self.stats['avg_time'] * (self.stats['updates'] - 1) + elapsed_time
            ) / self.stats['updates']
            self.stats['avg_options_per_symbol'] = total_processed / len(live_data) if live_data else 0
            
            # Enhanced status display
            self.print_maximized_status(live_data, total_processed, elapsed_time, greeks, market_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Maximized update cycle failed: {e}", exc_info=True)
            self.stats['updates'] += 1
            return False

    async def run(self):
        """Run the complete maximized real-time system"""
        print("🚀 Starting MAXIMIZED Real-Time GPU Portfolio System (16GB RAM)")
        print("=" * 80)
        print(f"📊 Symbols: {len(self.tracked_symbols)} across multiple sectors")
        print(f"   Tech: {self.tracked_symbols[:8]}")
        print(f"   Finance: {self.tracked_symbols[8:16]}")
        print(f"   Others: {self.tracked_symbols[16:24]}...")
        print(f"⏰ Update Interval: {self.update_interval} seconds")
        print(f"🖥️ GPU Mode: {'✅ ACTIVE' if self.gpu_interface.use_gpu else '🔄 CPU FALLBACK'}")
        print(f"🎯 Target Options: {self.max_total_options:,}+ options")
        print(f"📦 Batch Size: {self.batch_size:,}")
        print(f"💾 GPU Memory Target: {self.gpu_memory_buffer*100:.0f}%")
        print(f"🔬 Data Mode: {'SYNTHETIC (Max GPU Demo)' if self.use_synthetic_data else 'LIVE MARKET'}")
        print("🛑 Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            cycle_count = 0
            while self.running:
                cycle_count += 1
                cycle_start = time.time()
                
                print(f"{'='*20} CYCLE {cycle_count} {'='*20}")
                success = await self.update_cycle_maximized()
                
                if success:
                    cycle_time = time.time() - cycle_start
                    next_update = datetime.now() + timedelta(seconds=self.update_interval)
                    print(f"\n⏰ Cycle {cycle_count} complete! Next update at {next_update.strftime('%H:%M:%S')} "
                          f"(cycle: {cycle_time:.1f}s, waiting {self.update_interval}s...)\n")
                else:
                    print("\n⚠️ Update failed, retrying in 3 seconds...")
                    await asyncio.sleep(3)
                    continue
                
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Stopping maximized system (Ctrl+C pressed)...")
        except Exception as e:
            self.logger.error(f"Fatal system error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system gracefully with comprehensive final stats"""
        self.running = False
        print("✅ Maximized Real-Time GPU Portfolio System stopped")
        print(f"\n📊 FINAL COMPREHENSIVE STATS:")
        print(f"  Total Updates: {self.stats['successful_updates']}/{self.stats['updates']} successful")
        if self.stats['successful_updates'] > 0:
            print(f"  Average Cycle Time: {self.stats['avg_time']:.2f}s per cycle")
            print(f"  Max Concurrent Options: {self.stats['max_concurrent_options']:,}")
            print(f"  Peak Throughput: {self.stats['max_throughput']:,.0f} options/sec")
            print(f"  Avg Options/Symbol: {self.stats['avg_options_per_symbol']:.0f}")
            if self.stats['synthetic_data_used'] > 0:
                print(f"  Synthetic Options Generated: {self.stats['synthetic_data_used']:,}")
            print(f"  Total Options Processed: {self.stats['total_processed']:,}")

# Main execution
async def main():
    system = RealtimePortfolioSystemMaximized()
    await system.run()

if __name__ == "__main__":
    print("🚀 Starting MAXIMIZED Real-Time GPU Portfolio System for 16GB RAM")
    print("🎯 Target: 15,000-25,000+ options with full AAD Greeks computation")
    print("🔬 Synthetic data mode enabled for maximum GPU showcase")
    asyncio.run(main())
