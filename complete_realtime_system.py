import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# Import our modules
from live_options_fetcher import LiveOptionsDataFetcher
from gpu_portfolio_interface import GPUPortfolioInterface

class CompleteRealTimeSystem:
    def __init__(self):
        """Initialize the complete real-time system"""
        self.setup_logging()
        
        # Initialize components
        self.data_fetcher = LiveOptionsDataFetcher()
        self.gpu_interface = GPUPortfolioInterface()
        
        # System configuration
        self.tracked_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.update_interval = 60  # seconds
        self.running = False
        
        # Performance tracking
        self.stats = {
            'updates': 0,
            'successful_updates': 0,
            'total_options_processed': 0,
            'avg_processing_time': 0.0
        }
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_time_to_expiry(self, expiry_str: str) -> float:
        """Calculate time to expiry in years"""
        try:
            expiry_date = pd.to_datetime(expiry_str)
            now = pd.Timestamp.now()
            time_diff = (expiry_date - now).total_seconds()
            years = time_diff / (365.25 * 24 * 3600)
            return max(0.001, years)  # Minimum 1 day
        except:
            return 0.25  # Default to 3 months
    
    def process_live_data(self, live_data: Dict) -> List[Dict]:
        """Convert live options data to GPU format"""
        gpu_options = []
        market_data = {}
        risk_free_rate = 0.05
        
        for symbol, symbol_data in live_data.items():
            market_info = symbol_data['market_data']
            options = symbol_data['options']
            
            market_data[symbol] = {'spot_price': market_info.spot_price}
            
            # Process top ATM options for each symbol
            atm_options = [opt for opt in options 
                          if abs(opt.strike - market_info.spot_price) <= 20 
                          and opt.volume > 5]
            
            # Sort by volume and take top options
            atm_options.sort(key=lambda x: x.volume, reverse=True)
            
            for option in atm_options[:5]:  # Top 5 liquid options per symbol
                gpu_option = {
                    'symbol': symbol,
                    'strike': option.strike,
                    'spot_price': market_info.spot_price,
                    'time_to_expiry': self.calculate_time_to_expiry(option.expiry),
                    'risk_free_rate': risk_free_rate,
                    'implied_volatility': max(0.05, option.implied_volatility),
                    'is_call': (option.option_type == 'call'),
                    'market_price': option.last
                }
                gpu_options.append(gpu_option)
        
        return gpu_options, market_data
    
    def print_system_update(self, live_data: Dict, processed_count: int, 
                          processing_time: float, greeks):
        """Print comprehensive system update"""
        print(f"\n{'='*80}")
        print(f"🚀 REAL-TIME GPU PORTFOLIO SYSTEM - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        # Market Overview
        print(f"\n📈 MARKET DATA:")
        total_options_available = 0
        for symbol, symbol_data in live_data.items():
            market_data = symbol_data['market_data']
            options_count = len(symbol_data['options'])
            total_options_available += options_count
            
            position = self.gpu_interface.get_positions().get(symbol, {})
            if position:
                pnl = (market_data.spot_price - position['entry_price']) * position['quantity']
                print(f"  {symbol}: ${market_data.spot_price:>8.2f} | "
                      f"{options_count:>3d} options | "
                      f"Pos: {position['quantity']:>4d} | "
                      f"P&L: ${pnl:>+8,.0f}")
        
        # Portfolio Greeks
        print(f"\n💰 PORTFOLIO GREEKS:")
        print(f"  Delta:  {greeks.total_delta:>12.2f}   (Price sensitivity)")
        print(f"  Vega:   {greeks.total_vega:>12.2f}   (Volatility sensitivity)")
        print(f"  Gamma:  {greeks.total_gamma:>12.6f}   (Delta acceleration)")
        print(f"  Theta:  {greeks.total_theta:>12.2f}   (Time decay)")
        print(f"  Rho:    {greeks.total_rho:>12.2f}   (Interest rate sensitivity)")
        print(f"  P&L:    ${greeks.total_pnl:>11,.2f}   (Unrealized P&L)")
        
        # Performance Stats
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Processing Time:     {processing_time*1000:>8.1f} ms")
        print(f"  Options Processed:   {processed_count:>8d}")
        print(f"  Total Available:     {total_options_available:>8d}")
        print(f"  Success Rate:        {self.stats['successful_updates']/max(1,self.stats['updates'])*100:>8.1f}%")
        print(f"  Updates Completed:   {self.stats['updates']:>8d}")
        print(f"  Compute Method:      {'GPU' if self.gpu_interface.use_gpu else 'CPU':>8s}")
    
    async def run_update_cycle(self):
        """Run a single update cycle"""
        start_time = time.time()
        
        try:
            self.logger.info("Fetching live market data...")
            
            # Get live data
            live_data = self.data_fetcher.fetch_live_data(self.tracked_symbols)
            
            if not live_data:
                self.logger.warning("No live data received")
                return False
            
            # Process for GPU
            options_data, market_data = self.process_live_data(live_data)
            
            if not options_data:
                self.logger.warning("No valid options data to process")
                return False
            
            # Send to GPU/CPU for processing
            processed_count = self.gpu_interface.add_options_batch_with_positions(
                options_data, market_data
            )
            
            # Small delay for processing
            await asyncio.sleep(0.1)
            
            # Get updated Greeks
            greeks = self.gpu_interface.get_portfolio_greeks()
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['updates'] += 1
            self.stats['successful_updates'] += 1
            self.stats['total_options_processed'] += processed_count
            
            # Update average processing time
            alpha = 0.1
            if self.stats['avg_processing_time'] == 0:
                self.stats['avg_processing_time'] = processing_time
            else:
                self.stats['avg_processing_time'] = (
                    alpha * processing_time + 
                    (1 - alpha) * self.stats['avg_processing_time']
                )
            
            # Print update
            self.print_system_update(live_data, processed_count, processing_time, greeks)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}")
            self.stats['updates'] += 1
            return False
    
    async def run(self):
        """Run the complete real-time system"""
        print("🚀 Starting Complete Real-Time GPU Portfolio System")
        print("=" * 55)
        print(f"Symbols: {', '.join(self.tracked_symbols)}")
        print(f"Update Interval: {self.update_interval} seconds")
        print(f"Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            while self.running:
                success = await self.run_update_cycle()
                
                if self.running:
                    next_update = datetime.now() + timedelta(seconds=self.update_interval)
                    print(f"\n⏰ Next update at {next_update.strftime('%H:%M:%S')} "
                          f"(waiting {self.update_interval} seconds...)")
                    await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Stopping system...")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system"""
        self.running = False
        print("✅ Real-Time GPU Portfolio System stopped")

# Main execution
async def main():
    system = CompleteRealTimeSystem()
    await system.run()

if __name__ == "__main__":
    # Check dependencies
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run(["pip3", "install", "--user", "scipy"])
    
    asyncio.run(main())