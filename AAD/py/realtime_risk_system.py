import asyncio
import time
from datetime import datetime
import numpy as np
from collections import deque

from market_data import AsyncMarketDataFetcher
from gpu_aad_wrapper import compute_option_greeks_gpu
from gpu_monitor import GPUMonitor

class RealTimeRiskSystem:
    def __init__(self, symbols, max_history=100):
        self.symbols = symbols
        self.market_data = AsyncMarketDataFetcher(symbols)
        self.gpu_monitor = GPUMonitor()
        
        # Performance tracking
        self.performance_history = deque(maxlen=max_history)
        self.portfolio_history = deque(maxlen=max_history)
        
        # Statistics
        self.total_aad_operations = 0
        self.total_options_processed = 0
        self.cycle_count = 0
        
        print(f"🚀 Real-Time Risk System initialized with {len(symbols)} symbols")
    
    def aggregate_portfolio_greeks(self, option_results, option_params):
        """Aggregate option-level Greeks to portfolio level"""
        portfolio_greeks = {
            'total_delta': 0.0,
            'total_vega': 0.0,
            'total_gamma': 0.0,
            'total_theta': 0.0,
            'total_rho': 0.0,
            'total_value': 0.0,
            'num_options': len(option_results)
        }
        
        # Simple aggregation (in practice, you'd weight by position sizes)
        for i, result in enumerate(option_results):
            # Assume 1 contract per option for simplicity
            portfolio_greeks['total_delta'] += result['delta']
            portfolio_greeks['total_vega'] += result['vega']
            portfolio_greeks['total_gamma'] += result['gamma']
            portfolio_greeks['total_theta'] += result['theta']
            portfolio_greeks['total_rho'] += result['rho']
            portfolio_greeks['total_value'] += result['price']
        
        return portfolio_greeks
    
    def print_dashboard(self, portfolio_greeks, gpu_stats, performance_stats):
        """Print real-time dashboard"""
        print("\n" + "="*80)
        print(f"🔥 REAL-TIME PORTFOLIO RISK DASHBOARD - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # Portfolio Greeks
        print("📊 PORTFOLIO GREEKS:")
        print(f"   Delta:  {portfolio_greeks['total_delta']:>12.2f}")
        print(f"   Vega:   {portfolio_greeks['total_vega']:>12.2f}")
        print(f"   Gamma:  {portfolio_greeks['total_gamma']:>12.2f}")
        print(f"   Theta:  {portfolio_greeks['total_theta']:>12.2f}")
        print(f"   Rho:    {portfolio_greeks['total_rho']:>12.2f}")
        print(f"   Value:  ${portfolio_greeks['total_value']:>11.2f}")
        print(f"   Options: {portfolio_greeks['num_options']:>11d}")
        
        # GPU Statistics
        print("\n🔥 GPU UTILIZATION:")
        print(f"   GPU Usage:      {gpu_stats['gpu_utilization']:>6.1f}%")
        print(f"   Memory Usage:   {gpu_stats['memory_used_mb']:>6.1f}MB / {gpu_stats['memory_total_mb']:.1f}MB ({gpu_stats['memory_utilization']:.1f}%)")
        print(f"   Temperature:    {gpu_stats['temperature']:>6.1f}°C")
        print(f"   Power Draw:     {gpu_stats['power_draw']:>6.1f}W")
        
        # Performance Statistics
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"   Cycle Time:     {performance_stats['cycle_time']:>6.3f}s")
        print(f"   Data Fetch:     {performance_stats['fetch_time']:>6.3f}s")
        print(f"   GPU Compute:    {performance_stats['compute_time']:>6.3f}s")
        print(f"   AAD Operations: {performance_stats['aad_operations']:>6,d}")
        print(f"   Options/sec:    {performance_stats['options_per_sec']:>6,.0f}")
        print(f"   Total Cycles:   {self.cycle_count:>6,d}")
        print(f"   Total AAD Ops:  {self.total_aad_operations:>6,d}")
        
        print("="*80)
    
    async def run_single_cycle(self):
        """Execute one risk calculation cycle"""
        cycle_start = time.time()
        
        # 1. Fetch market data
        fetch_start = time.time()
        prices, fetch_time = await self.market_data.fetch_all_prices()
        
        # 2. Generate option parameters
        options = self.market_data.generate_option_parameters(prices, options_per_symbol=10000)
        
        # 3. Compute Greeks using GPU AAD
        compute_start = time.time()
        results, aad_ops = compute_option_greeks_gpu(options)
        compute_time = time.time() - compute_start
        
        # 4. Aggregate to portfolio level
        portfolio_greeks = self.aggregate_portfolio_greeks(results, options)
        
        # 5. Get GPU statistics
        gpu_stats = self.gpu_monitor.get_gpu_stats()
        
        # 6. Update statistics
        cycle_time = time.time() - cycle_start
        self.total_aad_operations += aad_ops
        self.total_options_processed += len(options)
        self.cycle_count += 1
        
        performance_stats = {
            'cycle_time': cycle_time,
            'fetch_time': fetch_time,
            'compute_time': compute_time,
            'aad_operations': aad_ops,
            'options_per_sec': len(options) / cycle_time if cycle_time > 0 else 0
        }
        
        # 7. Store history
        self.performance_history.append(performance_stats)
        self.portfolio_history.append(portfolio_greeks)
        
        return portfolio_greeks, gpu_stats, performance_stats
    
    async def run_realtime_loop(self, update_frequency=1.0):
        """Main real-time loop"""
        print("🚀 Starting Real-Time Risk Analysis...")
        print(f"📊 Monitoring {len(self.symbols)} symbols")
        print(f"⚡ Update frequency: {update_frequency}s")
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring()
        
        try:
            while True:
                loop_start = time.time()
                
                # Execute risk calculation cycle
                portfolio_greeks, gpu_stats, perf_stats = await self.run_single_cycle()
                
                # Print dashboard
                self.print_dashboard(portfolio_greeks, gpu_stats, perf_stats)
                
                # Calculate sleep time to maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_frequency - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"⚠️  Warning: Cycle took {elapsed:.3f}s, target was {update_frequency}s")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping real-time risk analysis...")
        finally:
            self.gpu_monitor.stop_monitoring()
            self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final performance summary"""
        print("\n" + "="*60)
        print("📈 FINAL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Cycles:           {self.cycle_count:,}")
        print(f"Total Options Processed: {self.total_options_processed:,}")
        print(f"Total AAD Operations:   {self.total_aad_operations:,}")
        
        if self.performance_history:
            avg_cycle_time = np.mean([p['cycle_time'] for p in self.performance_history])
            avg_throughput = np.mean([p['options_per_sec'] for p in self.performance_history])
            print(f"Average Cycle Time:     {avg_cycle_time:.3f}s")
            print(f"Average Throughput:     {avg_throughput:,.0f} options/sec")
        
        print("="*60)

# Market symbols to monitor
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NFLX", "NVDA", "ADBE", "INTC", 
    "CSCO", "ORCL", "CRM", "QCOM", "TXN"
]

async def main():
    """Main entry point"""
    # Create and run the real-time risk system
    risk_system = RealTimeRiskSystem(SYMBOLS)
    
    # Run with 1-second updates (adjust as needed)
    await risk_system.run_realtime_loop(update_frequency=1.0)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
