import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# Import WORKING components (using correct names)
from safe_gpu_interface import SafeGPUInterface
from live_options_fetcher import LiveOptionsDataFetcher  # ← FIXED: Correct class name

class RealtimePortfolioSystemFixed:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components with CORRECT class names
        self.data_fetcher = LiveOptionsDataFetcher()  # ← FIXED
        self.gpu_interface = SafeGPUInterface()

        # Configuration
        self.tracked_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.update_interval = 60  # seconds
        self.running = False

        # Stats tracking
        self.stats = {
            'updates': 0,
            'successful_updates': 0,
            'total_processed': 0,
            'avg_time': 0.0
        }

    def time_to_expiry(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            expiry_dt = pd.to_datetime(expiry_str)
            now = pd.Timestamp.now()
            delta = (expiry_dt - now).total_seconds() / (365.25 * 24 * 3600)
            return max(delta, 0.001)  # Minimum 1 day
        except Exception:
            return 0.25  # Default 3 months
    # In realtime_portfolio_system.py, replace the prepare_options_data method
# with the working version from complete_realtime_system.py

    def prepare_options_data(self, live_data):
        """FIXED: More lenient options filtering"""
        options = []
        market_data = {}

        for symbol, data in live_data.items():
            try:
                # Extract spot price (this part is working correctly)
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

                print(f"DEBUG: {symbol} spot_price={spot_price}, options_count={len(options_list)}")

                # FIXED: More lenient filtering criteria
                processed_options = []
                for opt in options_list[:20]:  # Process more options
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

                        # RELAXED FILTERING: Accept more options
                        strike_diff = abs(float(strike) - float(spot_price))
                        volume_check = float(volume) > 1  # Reduced from 10 to 1
                        strike_check = strike_diff <= 100  # Increased from 50 to 100

                        print(f"  Option: strike={strike}, volume={volume}, strike_diff={strike_diff:.2f}")

                        if strike_check and volume_check and float(strike) > 0:
                            processed_options.append({
                                'symbol': symbol,
                                'strike': float(strike),
                                'spot_price': float(spot_price),
                                'time_to_expiry': self.time_to_expiry(expiry),
                                'risk_free_rate': 0.05,
                                'implied_volatility': max(float(iv), 0.05),
                                'is_call': str(opt_type).lower() == 'call',
                                'market_price': float(last_price),
                                'volume': float(volume)
                            })
                            print(f"    ✅ ACCEPTED option: {strike} strike, {volume} volume")
                        else:
                            print(f"    ❌ REJECTED: strike_ok={strike_check}, volume_ok={volume_check}")

                    except Exception as e:
                        print(f"    Error processing option: {e}")
                        continue

                print(f"  Total processed options for {symbol}: {len(processed_options)}")

                # Sort by volume and take top options
                processed_options.sort(key=lambda x: -x.get('volume', 0))
                options.extend(processed_options[:5])  # Top 5 per symbol

            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {e}")
                market_data[symbol] = {'spot_price': 0.0}

        print(f"FINAL: Total options for processing: {len(options)}")
        return options, market_data


    def print_system_status(self, live_data, processed_count, elapsed_time, greeks, market_data):
        # ... existing code ...

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
        print(f"  Delta:  {greeks.total_delta:>12.3f}   (Price sensitivity)")
        print(f"  Vega:   {greeks.total_vega:>12.3f}   (Volatility sensitivity)")
        print(f"  Gamma:  {greeks.total_gamma:>12.6f}   (Delta acceleration)")
        print(f"  Theta:  {greeks.total_theta:>12.3f}   (Time decay)")
        print(f"  Rho:    {greeks.total_rho:>12.3f}   (Interest rate sensitivity)")
        print(f"  P&L:    ${greeks.total_pnl:>11,.2f}   (Unrealized P&L)")

        print(f"\n⚡ PERFORMANCE:")
        print(f"  Processing Time:     {elapsed_time*1000:>8.1f} ms")
        print(f"  Options Processed:   {processed_count:>8d}")
        print(f"  Total Available:     {total_options:>8d}")
        print(f"  Success Rate:        {self.stats['successful_updates']/max(1,self.stats['updates'])*100:>8.1f}%")
        print(f"  Updates Completed:   {self.stats['updates']:>8d}")
        print(f"  Compute Method:      {'GPU' if self.gpu_interface.use_gpu else 'CPU':>8s}")

    async def update_cycle(self):
        """Execute one complete update cycle"""
        start_time = time.time()

        try:
            self.logger.info("Fetching live market data...")
            
            # Fetch data using the CORRECT method name
            live_data = self.data_fetcher.fetch_live_data(self.tracked_symbols)
            
            if not live_data:
                self.logger.warning("No live data received")
                return False

            # Process data for GPU computation
            options_data, market_data = self.prepare_options_data(live_data)
            
            if not options_data:
                self.logger.warning("No valid options data to process")
                return False

            # Process using GPU/CPU via SafeGPUInterface
            processed_count = self.gpu_interface.process_portfolio_options(
                options_data, market_data
            )

            # Get computed Greeks
            greeks = self.gpu_interface.get_portfolio_greeks()

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

    async def run(self):
        """Run the complete real-time system"""
        print("🚀 Starting Real-Time Portfolio System")
        print("=" * 50)
        print(f"📊 Symbols: {', '.join(self.tracked_symbols)}")
        print(f"⏰ Update Interval: {self.update_interval} seconds")
        print(f"🖥️  GPU Mode: {'✅ ACTIVE' if self.gpu_interface.use_gpu else '🔄 CPU FALLBACK'}")
        print("🛑 Press Ctrl+C to stop\n")

        self.running = True

        try:
            while self.running:
                success = await self.update_cycle()
                
                if success:
                    next_update = datetime.now() + timedelta(seconds=self.update_interval)
                    print(f"\n⏰ Next update at {next_update.strftime('%H:%M:%S')} "
                          f"(waiting {self.update_interval} seconds...)")
                else:
                    print("\n⚠️  Update failed, retrying in 30 seconds...")
                    await asyncio.sleep(30)
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
        print("✅ Real-Time Portfolio System stopped")
        print(f"📊 Final Stats: {self.stats['successful_updates']}/{self.stats['updates']} successful updates")

# Main execution
async def main():
    system = RealtimePortfolioSystemFixed()
    await system.run()

if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import scipy
    except ImportError:
        print("📦 Installing scipy...")
        import subprocess
        subprocess.run(["pip3", "install", "--user", "scipy"])
    
    print("🚀 Starting Real-Time GPU Portfolio System")
    asyncio.run(main())
