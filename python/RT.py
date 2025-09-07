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

class RealtimePortfolioSystemOptimized:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with CORRECT class names
        self.data_fetcher = LiveOptionsDataFetcher()
        self.gpu_interface = SafeGPUInterface()
        
        # 🚀 EXPANDED Configuration for 16GB GPU
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
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT', 'XLF', 'XLE'
        ]  # 40 symbols instead of 10
        
        # 🚀 MEMORY-OPTIMIZED Settings
        self.max_options_per_symbol = 500  # Increased from 100
        self.max_total_options = 15000     # Target ~15K options for 16GB
        self.batch_size = 2000             # Process in chunks
        self.update_interval = 1.5         # Faster updates
        
        # GPU Memory Management
        self.gpu_memory_buffer = 0.85      # Use 85% of GPU memory
        self.prealloc_memory = True        # Pre-allocate GPU arrays
        
        self.running = False
        
        # Enhanced Stats tracking
        self.stats = {
            'updates': 0,
            'successful_updates': 0,
            'total_processed': 0,
            'avg_time': 0.0,
            'gpu_memory_used': 0.0,
            'max_concurrent_options': 0,
            'total_symbols_tracked': len(self.tracked_symbols),
            'avg_options_per_symbol': 0
        }

    # 🚀 OPTIMIZED: Enhanced Parallel Data Fetching
    async def fetch_all_symbols_optimized(self):
        """🚀 OPTIMIZED: Maximize parallel fetching for 40+ symbols"""
        fetch_start = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            # 🚀 INCREASED PARALLELISM: More workers for more symbols
            with ThreadPoolExecutor(max_workers=12) as executor:  # Increased workers
                # Create batched fetch tasks to avoid API limits
                symbol_batches = [
                    self.tracked_symbols[i:i+8] 
                    for i in range(0, len(self.tracked_symbols), 8)
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
            self.logger.warning(f"Optimized fetch failed: {e}, falling back to batch mode")
            
            # Fallback to batch mode
            try:
                live_data = await loop.run_in_executor(
                    None,
                    self.data_fetcher.fetch_live_data,
                    self.tracked_symbols
                )
                fetch_time = time.time() - fetch_start
                successful_symbols = len(live_data) if live_data else 0
                print(f"⚡ Batch fetch: {fetch_time*1000:.1f}ms "
                      f"({successful_symbols}/{len(self.tracked_symbols)} symbols)")
                return live_data
            except Exception as e2:
                self.logger.error(f"All fetch methods failed: {e2}")
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

    def prepare_options_data_optimized(self, live_data):
        """🚀 MEMORY-OPTIMIZED: Process more options with intelligent filtering"""
        options = []
        market_data = {}
        processing_start = time.time()
        
        # Pre-calculate memory requirements
        estimated_options = sum(
            len(data.get('options', [])) if isinstance(data, dict) 
            else len(getattr(data, 'options', []))
            for data in live_data.values()
        )
        
        # Dynamic adjustment based on available options
        if estimated_options > self.max_total_options:
            options_per_symbol = max(50, self.max_total_options // len(live_data))
            print(f"🔧 Adjusting to {options_per_symbol} options/symbol "
                  f"(total estimated: {estimated_options})")
        else:
            options_per_symbol = self.max_options_per_symbol
        
        for symbol, data in live_data.items():
            try:
                # Extract data efficiently
                if isinstance(data, dict):
                    if 'market_data' in data:
                        market_info = data['market_data']
                        spot_price = getattr(market_info, 'spot_price', 0)
                    else:
                        spot_price = data.get('spot_price', 0)
                    options_list = data.get('options', [])
                else:
                    spot_price = getattr(data, 'spot_price', 0)
                    options_list = getattr(data, 'options', [])
                
                market_data[symbol] = {'spot_price': float(spot_price)}
                
                # 🚀 INTELLIGENT OPTION FILTERING
                processed_options = []
                for opt in options_list:
                    try:
                        if isinstance(opt, dict):
                            strike = float(opt.get('strike', 0))
                            volume = float(opt.get('volume', 0))
                            oi = float(opt.get('open_interest', 0))
                            expiry = opt.get('expiry', '2024-12-20')
                            iv = float(opt.get('implied_volatility', opt.get('impliedVol', 0.25)))
                            opt_type = opt.get('type', opt.get('option_type', 'call'))
                            last_price = float(opt.get('last', opt.get('last_price', 0)))
                        else:
                            strike = float(getattr(opt, 'strike', 0))
                            volume = float(getattr(opt, 'volume', 0))
                            oi = float(getattr(opt, 'open_interest', 0))
                            expiry = getattr(opt, 'expiry', '2024-12-20')
                            iv = float(getattr(opt, 'implied_volatility', 0.25))
                            opt_type = getattr(opt, 'option_type', 'call')
                            last_price = float(getattr(opt, 'last', 0))
                        
                        # Enhanced filtering criteria
                        spot_val = float(spot_price)
                        if (strike > 0 and spot_val > 0 and 
                            iv > 0.05 and iv < 5.0 and  # Reasonable IV range
                            (volume > 0 or oi > 0) and   # Some liquidity
                            0.5 <= strike/spot_val <= 2.0):  # Reasonable moneyness
                            
                            time_to_exp = self.time_to_expiry(expiry)
                            if time_to_exp > 0.001:  # Not expired
                                processed_options.append({
                                    'symbol': symbol,
                                    'strike': strike,
                                    'spot_price': spot_val,
                                    'time_to_expiry': time_to_exp,
                                    'risk_free_rate': 0.05,
                                    'implied_volatility': iv,
                                    'is_call': str(opt_type).lower() == 'call',
                                    'market_price': last_price,
                                    'volume': volume,
                                    'open_interest': oi,
                                    'liquidity_score': volume + 0.1 * oi  # Combined liquidity
                                })
                    except (ValueError, TypeError):
                        continue
                
                # Sort by liquidity and take top options
                processed_options.sort(key=lambda x: -x['liquidity_score'])
                options.extend(processed_options[:options_per_symbol])
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                market_data[symbol] = {'spot_price': 0.0}
        
        processing_time = time.time() - processing_start
        print(f"🔄 Enhanced processing: {processing_time*1000:.1f}ms "
              f"({len(options)} options from {len(live_data)} symbols)")
        
        return options, market_data

    def print_enhanced_status(self, live_data, processed_count, elapsed_time, greeks, market_data):
        """Display comprehensive system status with GPU utilization"""
        
        print(f"\n📈 EXPANDED MARKET DATA ({len(live_data)} symbols):")
        
        # Group symbols by sector for better display
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA']
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT']
        
        categories = [
            ("TECH", tech_symbols[:6]),
            ("FINANCE", financial_symbols[:6]),
            ("ETFs", etf_symbols[:4])
        ]
        
        for category, symbols in categories:
            print(f"\n{category}:")
            for i, symbol in enumerate(symbols):
                if symbol in live_data:
                    spot = market_data.get(symbol, {}).get('spot_price', 0)
                    data = live_data[symbol]
                    if isinstance(data, dict):
                        opts_count = len(data.get('options', []))
                    else:
                        opts_count = len(getattr(data, 'options', []))
                    
                    print(f"  {symbol}: ${spot:>8.2f} ({opts_count:>3d} opts)", end="")
                    if (i + 1) % 3 == 0:  # New line every 3 symbols
                        print()
            print()
        
        print(f"\n💰 PORTFOLIO GREEKS (Real-time AAD):")
        print(f"  Delta: {greeks.total_delta:>10.3f}  |  Vega: {greeks.total_vega:>10.3f}")
        print(f"  Gamma: {greeks.total_gamma:>10.6f}  |  Theta: {greeks.total_theta:>10.3f}")
        print(f"  Rho: {greeks.total_rho:>10.3f}    |  P&L: ${greeks.total_pnl:>12,.2f}")
        
        print(f"\n🚀 GPU PERFORMANCE METRICS:")
        print(f"  Processing Time: {elapsed_time*1000:>8.1f} ms")
        print(f"  Options Processed: {processed_count:>8,d}")
        print(f"  Max Concurrent: {self.stats['max_concurrent_options']:>8,d}")
        print(f"  GPU Memory Used: {self.stats['gpu_memory_used']:>8.1f}%")
        print(f"  Throughput: {processed_count/elapsed_time:>8,.0f} options/sec")
        print(f"  Success Rate: {self.stats['successful_updates']/max(1,self.stats['updates'])*100:>8.1f}%")
        print(f"  Symbols Active: {len(live_data):>8d}/{self.stats['total_symbols_tracked']}")
        print(f"  Compute Method: {'GPU' if self.gpu_interface.use_gpu else 'CPU':>8s}")

    async def update_cycle_optimized(self):
        """🚀 MEMORY-AWARE: Execute optimized update cycle with GPU monitoring"""
        start_time = time.time()
        
        try:
            self.logger.info("Fetching expanded market data...")
            
            # 🚀 OPTIMIZED PARALLEL FETCHING
            live_data = await self.fetch_all_symbols_optimized()
            if not live_data:
                self.logger.warning("No live data received")
                return False
            
            # Process with memory awareness
            gpu_start = time.time()
            options_data, market_data = self.prepare_options_data_optimized(live_data)
            
            if not options_data:
                self.logger.warning("No valid options data to process")
                return False
            
            # 🚀 BATCH PROCESSING for large datasets
            total_processed = 0
            num_batches = len(options_data) // self.batch_size + (1 if len(options_data) % self.batch_size else 0)
            
            for i in range(0, len(options_data), self.batch_size):
                batch = options_data[i:i + self.batch_size]
                batch_count = self.gpu_interface.process_portfolio_options(
                    batch, market_data
                )
                total_processed += batch_count
                
                # Monitor GPU memory usage if available
                if hasattr(self.gpu_interface, 'get_gpu_memory_usage'):
                    gpu_memory = self.gpu_interface.get_gpu_memory_usage()
                    self.stats['gpu_memory_used'] = gpu_memory
            
            # Track maximum concurrent options
            self.stats['max_concurrent_options'] = max(
                self.stats['max_concurrent_options'], 
                len(options_data)
            )
            
            greeks = self.gpu_interface.get_portfolio_greeks()
            gpu_time = time.time() - gpu_start
            
            print(f"🚀 GPU batch processing: {gpu_time*1000:.1f}ms "
                  f"({total_processed} options in {num_batches} batches)")
            
            # Update stats
            elapsed_time = time.time() - start_time
            self.stats['updates'] += 1
            self.stats['successful_updates'] += 1
            self.stats['total_processed'] += total_processed
            self.stats['avg_time'] = (
                self.stats['avg_time'] * (self.stats['updates'] - 1) + elapsed_time
            ) / self.stats['updates']
            self.stats['avg_options_per_symbol'] = total_processed / len(live_data) if live_data else 0
            
            # Enhanced status display
            self.print_enhanced_status(live_data, total_processed, elapsed_time, greeks, market_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Optimized update cycle failed: {e}", exc_info=True)
            self.stats['updates'] += 1
            return False

    async def run(self):
        """Run the complete optimized real-time system"""
        print("🚀 Starting OPTIMIZED Real-Time GPU Portfolio System (16GB RAM)")
        print("=" * 75)
        print(f"📊 Symbols: {len(self.tracked_symbols)} ({', '.join(self.tracked_symbols[:8])}...)")
        print(f"⏰ Update Interval: {self.update_interval} seconds")
        print(f"🖥️ GPU Mode: {'✅ ACTIVE' if self.gpu_interface.use_gpu else '🔄 CPU FALLBACK'}")
        print(f"⚡ Max Options/Symbol: {self.max_options_per_symbol}")
        print(f"🔢 Target Total Options: {self.max_total_options:,}")
        print(f"📦 Batch Size: {self.batch_size:,}")
        print(f"💾 GPU Memory Target: {self.gpu_memory_buffer*100:.0f}%")
        print("🛑 Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            while self.running:
                cycle_start = time.time()
                success = await self.update_cycle_optimized()
                
                if success:
                    cycle_time = time.time() - cycle_start
                    next_update = datetime.now() + timedelta(seconds=self.update_interval)
                    print(f"\n⏰ Next update at {next_update.strftime('%H:%M:%S')} "
                          f"(cycle: {cycle_time:.1f}s, waiting {self.update_interval}s...)")
                else:
                    print("\n⚠️ Update failed, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Stopping system (Ctrl+C pressed)...")
        except Exception as e:
            self.logger.error(f"Fatal system error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system gracefully"""
        self.running = False
        print("✅ Optimized Real-Time GPU Portfolio System stopped")
        print(f"📊 Final Stats: {self.stats['successful_updates']}/{self.stats['updates']} successful updates")
        if self.stats['successful_updates'] > 0:
            print(f"⚡ Average processing time: {self.stats['avg_time']:.2f}s per cycle")
            print(f"🔢 Max concurrent options: {self.stats['max_concurrent_options']:,}")
            print(f"📈 Avg options/symbol: {self.stats['avg_options_per_symbol']:.0f}")

# Main execution
async def main():
    system = RealtimePortfolioSystemOptimized()
    await system.run()

if __name__ == "__main__":
    print("🚀 Starting OPTIMIZED Real-Time GPU Portfolio System for 16GB RAM")
    print("🎯 Target: 15,000+ options across 40+ symbols with AAD Greeks")
    asyncio.run(main())
