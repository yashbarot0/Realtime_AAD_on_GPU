import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import logging

from live_options_fetcher import LiveOptionsDataFetcher, OptionContract, MarketData
from gpu_portfolio_interface import GPUPortfolioInterface, PortfolioGreeks

class RealTimePortfolioSystem:
    def __init__(self):
        """Initialize the real-time portfolio system"""
        self.setup_logging()
        
        # Initialize components
        self.data_fetcher = LiveOptionsDataFetcher()
        self.gpu_interface = GPUPortfolioInterface()
        
        # System state
        self.running = False
        self.update_interval = 30  # seconds
        
        # Portfolio configuration
        self.tracked_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        self.portfolio_positions = {
            'AAPL': {'quantity': 1000, 'entry_price': 150.0},
            'MSFT': {'quantity': 500, 'entry_price': 300.0},
            'GOOGL': {'quantity': 100, 'entry_price': 120.0},
            'TSLA': {'quantity': 200, 'entry_price': 200.0},
            'AMZN': {'quantity': 150, 'entry_price': 100.0}
        }
        
        # Risk management parameters
        self.risk_limits = {
            'max_delta': 10000,
            'max_vega': 50000,
            'max_gamma': 1000,
            'max_daily_loss': -10000
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_processing_time': 0,
            'last_update_time': None
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('realtime_portfolio.log'),
                logging.StreamHandler()
            ]
        )
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
    
    def process_options_data(self, live_data: Dict[str, Dict]) -> List[Dict]:
        """Convert live options data to GPU format"""
        gpu_options = []
        risk_free_rate = 0.05  # Assume 5% risk-free rate
        
        for symbol, symbol_data in live_data.items():
            market_data = symbol_data['market_data']
            options = symbol_data['options']
            
            for option in options:
                # Filter for liquid options
                if option.volume > 10 and option.implied_volatility > 0.05:
                    gpu_option = {
                        'symbol': symbol,
                        'strike': option.strike,
                        'spot_price': market_data.spot_price,
                        'time_to_expiry': self.calculate_time_to_expiry(option.expiry),
                        'risk_free_rate': risk_free_rate,
                        'implied_volatility': option.implied_volatility,
                        'is_call': (option.option_type == 'call'),
                        'market_price': option.last
                    }
                    gpu_options.append(gpu_option)
        
        return gpu_options
    
    def check_risk_limits(self, greeks: PortfolioGreeks) -> Dict[str, bool]:
        """Check if portfolio exceeds risk limits"""
        violations = {}
        
        violations['delta_exceeded'] = abs(greeks.total_delta) > self.risk_limits['max_delta']
        violations['vega_exceeded'] = abs(greeks.total_vega) > self.risk_limits['max_vega']
        violations['gamma_exceeded'] = abs(greeks.total_gamma) > self.risk_limits['max_gamma']
        violations['loss_exceeded'] = greeks.total_pnl < self.risk_limits['max_daily_loss']
        
        return violations
    
    def generate_alerts(self, greeks: PortfolioGreeks, violations: Dict[str, bool]):
        """Generate risk alerts"""
        if any(violations.values()):
            self.logger.warning("RISK ALERT!")
            
            if violations['delta_exceeded']:
                self.logger.warning(f"Delta exposure exceeded: {greeks.total_delta:.0f}")
            
            if violations['vega_exceeded']:
                self.logger.warning(f"Vega exposure exceeded: {greeks.total_vega:.0f}")
            
            if violations['gamma_exceeded']:
                self.logger.warning(f"Gamma exposure exceeded: {greeks.total_gamma:.0f}")
            
            if violations['loss_exceeded']:
                self.logger.warning(f"Daily loss exceeded: ${greeks.total_pnl:.2f}")
    
    def update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['total_updates'] += 1
        if success:
            self.performance_stats['successful_updates'] += 1
        
        # Update average processing time (exponential moving average)
        alpha = 0.1
        if self.performance_stats['avg_processing_time'] == 0:
            self.performance_stats['avg_processing_time'] = processing_time
        else:
            self.performance_stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.performance_stats['avg_processing_time']
            )
        
        self.performance_stats['last_update_time'] = datetime.now()
    
    def print_portfolio_update(self, greeks: PortfolioGreeks, num_options: int):
        """Print portfolio update to console"""
        print(f"\n{'='*80}")
        print(f"PORTFOLIO UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Options Processed: {num_options}")
        print(f"")
        print(f"PORTFOLIO GREEKS:")
        print(f"  Delta:  {greeks.total_delta:>10.2f}")
        print(f"  Vega:   {greeks.total_vega:>10.2f}")
        print(f"  Gamma:  {greeks.total_gamma:>10.4f}")
        print(f"  Theta:  {greeks.total_theta:>10.2f}")
        print(f"  Rho:    {greeks.total_rho:>10.2f}")
        print(f"  P&L:    ${greeks.total_pnl:>9.2f}")
        
        print(f"\nPERFORMANCE STATS:")
        success_rate = (self.performance_stats['successful_updates'] / 
                       max(1, self.performance_stats['total_updates']) * 100)
        print(f"  Success Rate:    {success_rate:>6.1f}%")
        print(f"  Avg Proc Time:   {self.performance_stats['avg_processing_time']*1000:>6.1f}ms")
        print(f"  Total Updates:   {self.performance_stats['total_updates']:>6d}")
    
    async def run_update_cycle(self):
        """Run a single update cycle"""
        start_time = time.time()
        success = False
        
        try:
            # Fetch live data
            self.logger.info(f"Fetching live data for {len(self.tracked_symbols)} symbols...")
            live_data = self.data_fetcher.fetch_live_data(self.tracked_symbols)
            
            if not live_data:
                self.logger.warning("No live data received")
                return
            
            # Convert to GPU format
            gpu_options = self.process_options_data(live_data)
            
            if not gpu_options:
                self.logger.warning("No valid options data to process")
                return
            
            self.logger.info(f"Processing {len(gpu_options)} options on GPU...")
            
            # Send to GPU for processing
            self.gpu_interface.add_options_batch(gpu_options)
            
            # Small delay to allow GPU processing
            await asyncio.sleep(0.1)
            
            # Get updated Greeks
            greeks = self.gpu_interface.get_portfolio_greeks()
            
            # Check risk limits
            violations = self.check_risk_limits(greeks)
            self.generate_alerts(greeks, violations)
            
            # Print update
            self.print_portfolio_update(greeks, len(gpu_options))
            
            success = True
            
        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self.update_performance_stats(processing_time, success)
    
    async def run(self):
        """Run the real-time portfolio system"""
        self.logger.info("Starting Real-Time Portfolio System...")
        self.logger.info(f"Tracking symbols: {', '.join(self.tracked_symbols)}")
        self.logger.info(f"Update interval: {self.update_interval} seconds")
        
        self.running = True
        
        try:
            while self.running:
                await self.run_update_cycle()
                
                # Wait for next update
                if self.running:
                    await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received stop signal...")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping Real-Time Portfolio System...")
        self.running = False

# Main execution
async def main():
    """Main function"""
    system = RealTimePortfolioSystem()
    
    try:
        await system.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()

if __name__ == "__main__":
    print("Real-Time GPU Portfolio Greeks System")
    print("====================================")
    print("Press Ctrl+C to stop")
    print()
    
    asyncio.run(main())
