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

class RealtimePortfolioSystemFixed:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with CORRECT class names
        self.data_fetcher = LiveOptionsDataFetcher()
        self.gpu_interface = SafeGPUInterface()
        
        # Configuration
        self.tracked_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'SPY', 'QQQ']
        self.update_interval = 2  # seconds
        self.running = False
        
        # Stats tracking
        self.stats = {
            'updates': 0,
            'successful_updates': 0,
            'total_processed': 0,
            'avg_time': 0.0
        }

    # 🚀 OPTIMIZED: True Parallel Data Fetching
    async def fetch_all_symbols_async(self):
        """🚀 OPTIMIZED: Use parallel processing with proper worker count"""
        fetch_start = time.time()
        
        loop = asyncio.get_event_loop()
        
        # Method 1: Try parallel individual fetches
        try:
            # ✅ PARALLEL APPROACH: Multiple workers
            with ThreadPoolExecutor(max_workers=5) as executor:  # 5 parallel workers
                tasks = []
                
                # Create individual fetch tasks
                for symbol in self.tracked_symbols:
                    task = loop.run_in_executor(
                        executor,
                        lambda s=symbol: self.data_fetcher.fetch_live_data([s])
                    )
                    tasks.append((symbol, task))
                
                # Execute all tasks in parallel
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                # Combine results
                live_data = {}
                successful_symbols = 0
                
                for (symbol, _), result in zip(tasks, results):
                    if not isinstance(result, Exception) and result and symbol in result:
                        live_data[symbol] = result[symbol]
                        successful_symbols += 1
                    else:
                        self.logger.warning(f"Failed to fetch {symbol}")
                
                fetch_time = time.time() - fetch_start
                print(f"⚡ Parallel fetch: {fetch_time*1000:.1f}ms ({successful_symbols}/{len(self.tracked_symbols)} symbols)")
                
                return live_data if live_data else None
                
        except Exception as e:
            self.logger.warning(f"Parallel fetch failed: {e}, falling back to batch mode")
            
            # Method 2: Fallback to batch mode (but still async)
            try:
                live_data = await loop.run_in_executor(
                    None, 
                    self.data_fetcher.fetch_live_data, 
                    self.tracked_symbols
                )
                
                fetch_time = time.time() - fetch_start
                successful_symbols = len(live_data) if live_data else 0
                print(f"⚡ Batch fetch: {fetch_time*1000:.1f}ms ({successful_symbols}/{len(self.tracked_symbols)} symbols)")
                
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

    def prepare_options_data(self, live_data):
        """OPTIMIZED: Streamlined data processing"""
        options = []
        market_data = {}
        
        processing_start = time.time()

        for symbol, data in live_data.items():
            try:
                # Fast extraction
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

                # 🚀 VECTORIZED PROCESSING
                processed_options = []
                for opt in options_list:
                    try:
                        if isinstance(opt, dict):
                            strike = opt.get('strike', 0)
                            volume = opt.get('volume', 0)
                            expiry = opt.get('expiry', '2024-12-20')
                            iv = opt.get('implied_volatility', opt.get('impliedVol', 0.25))
                            opt_type = opt.get('type', opt.get('option_type', 'call'))
                            last_price = opt.get('last', opt.get('last_price', 0))
                        else:
                            strike = getattr(opt, 'strike', 0)
                            volume = getattr(opt, 'volume', 0)
                            expiry = getattr(opt, 'expiry', '2024-12-20')
                            iv = getattr(opt, 'implied_volatility', 0.25)
                            opt_type = getattr(opt, 'option_type', 'call')
                            last_price = getattr(opt, 'last', 0)

                        # Minimal validation for speed
                        strike_val = float(strike)
                        spot_val = float(spot_price)
                        
                        if strike_val > 0 and spot_val > 0:
                            processed_options.append({
                                'symbol': symbol,
                                'strike': strike_val,
                                'spot_price': spot_val,
                                'time_to_expiry': self.time_to_expiry(expiry),
                                'risk_free_rate': 0.05,
                                'implied_volatility': max(float(iv), 0.05),
                                'is_call': str(opt_type).lower() == 'call',
                                'market_price': float(last_price),
                                'volume': float(volume)
                            })

                    except:
                        continue  # Skip invalid options silently

                # Take top options by volume
                processed_options.sort(key=lambda x: -abs(x.get('volume', 0)))
                options.extend(processed_options[:100])  # 100 per symbol

            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {e}")
                market_data[symbol] = {'spot_price': 0.0}

        processing_time = time.time() - processing_start
        print(f"🔄 Data processing: {processing_time*1000:.1f}ms ({len(options)} options)")
        
        return options, market_data

    def print_system_status(self, live_data, processed_count, elapsed_time, greeks, market_data):
        """Display system status with performance breakdown"""
        print(f"\n📈 MARKET DATA:")
        total_options = 0
        
        for symbol in self.tracked_symbols:
            if symbol in live_data:
                # ✅ FIXED: Use the processed market_data
                spot = market_data.get(symbol, {}).get('spot_price', 0)
                
                data = live_data[symbol]
                if isinstance(data, dict):
                    opts_count = len(data.get('options', []))
                else:
                    opts_count = len(getattr(data, 'options', []))
                
                total_options += opts_count
                
                position = self.gpu_interface.get_positions().get(symbol, {})
                pnl = 0
                if position:
                    pnl = (spot - position.get('entry_price', 0)) * position.get('quantity', 0)
                
                print(f"  {symbol}: ${spot:>8.2f} | {opts_count:>3d} options | "
                      f"Pos: {position.get('quantity', 0):>4d} | P&L: ${pnl:>+8,.0f}")

        print(f"\n💰 PORTFOLIO GREEKS:")
        print(f"  Delta:       {greeks.total_delta:>8.3f}   (Price sensitivity)")
        print(f"  Vega:        {greeks.total_vega:>8.3f}   (Volatility sensitivity)")
        print(f"  Gamma:     {greeks.total_gamma:>10.6f}   (Delta acceleration)")
        print(f"  Theta:       {greeks.total_theta:>8.3f}   (Time decay)")
        print(f"  Rho:         {greeks.total_rho:>8.3f}   (Interest rate sensitivity)")
        print(f"  P&L:    ${greeks.total_pnl:>11,.2f}   (Unrealized P&L)")

        print(f"\n⚡ PERFORMANCE:")
        print(f"  Processing Time:       {elapsed_time*1000:>8.1f} ms")
        print(f"  Options Processed:     {processed_count:>8d}")
        print(f"  Total Available:       {total_options:>8d}")
        print(f"  Success Rate:          {self.stats['successful_updates']/max(1,self.stats['updates'])*100:>8.1f}%")
        print(f"  Updates Completed:     {self.stats['updates']:>8d}")
        print(f"  Compute Method:        {'GPU' if self.gpu_interface.use_gpu else 'CPU':>8s}")

    async def update_cycle(self):
        """🚀 OPTIMIZED: Execute one complete update cycle with parallel fetching"""
        start_time = time.time()
        
        try:
            self.logger.info("Fetching live market data...")
            
            # 🚀 PARALLEL FETCHING
            live_data = await self.fetch_all_symbols_async()
            
            if not live_data:
                self.logger.warning("No live data received")
                return False

            # Process data for GPU computation
            gpu_start = time.time()
            options_data, market_data = self.prepare_options_data(live_data)
            
            if not options_data:
                self.logger.warning("No valid options data to process")
                return False

            # Process using GPU/CPU via SafeGPUInterface
            # processed_count = self.gpu_interface.process_portfolio_options(
            #     options_data, market_data
            # )
            processed_count = self.gpu_interface.process_portfolio_options_cached(
                options_data, market_data
            )
            # Get computed Greeks
            greeks = self.gpu_interface.get_portfolio_greeks()
            
            gpu_time = time.time() - gpu_start
            print(f"🚀 GPU processing: {gpu_time*1000:.1f}ms")

            # Update statistics
            elapsed_time = time.time() - start_time
            self.stats['updates'] += 1
            self.stats['successful_updates'] += 1
            self.stats['total_processed'] += processed_count
            self.stats['avg_time'] = (
                self.stats['avg_time'] * (self.stats['updates'] - 1) + elapsed_time
            ) / self.stats['updates']

            # Display results
            self.print_system_status(live_data, processed_count, elapsed_time, greeks, market_data)
            
            return True

        except Exception as e:
            self.logger.error(f"Update cycle failed: {e}", exc_info=True)
            self.stats['updates'] += 1
            return False

    # ✅ FIXED: Added the missing run method
    async def run(self):
        """Run the complete real-time system with async optimization"""
        print("🚀 Starting Real-Time GPU Portfolio System (ASYNC OPTIMIZED)")
        print("=" * 65)
        print(f"📊 Symbols: {', '.join(self.tracked_symbols)}")
        print(f"⏰ Update Interval: {self.update_interval} seconds")
        print(f"🖥️  GPU Mode: {'✅ ACTIVE' if self.gpu_interface.use_gpu else '🔄 CPU FALLBACK'}")
        print(f"⚡ Async Fetching: ✅ ENABLED (Parallel)")
        print("🛑 Press Ctrl+C to stop\n")

        self.running = True

        try:
            while self.running:
                cycle_start = time.time()
                success = await self.update_cycle()
                
                if success:
                    cycle_time = time.time() - cycle_start
                    next_update = datetime.now() + timedelta(seconds=self.update_interval)
                    print(f"\n⏰ Next update at {next_update.strftime('%H:%M:%S')} "
                          f"(cycle: {cycle_time:.1f}s, waiting {self.update_interval}s...)")
                else:
                    print("\n⚠️  Update failed, retrying in 5 seconds...")
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
        print("✅ Real-Time GPU Portfolio System stopped")
        print(f"📊 Final Stats: {self.stats['successful_updates']}/{self.stats['updates']} successful updates")
        if self.stats['successful_updates'] > 0:
            print(f"⚡ Average processing time: {self.stats['avg_time']:.2f}s per cycle")

# Main execution
async def main():
    system = RealtimePortfolioSystemFixed()
    await system.run()

if __name__ == "__main__":
    print("🚀 Starting Real-Time GPU Portfolio System with TRUE PARALLEL Optimization")
    asyncio.run(main())
