import psutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class PerformanceMetrics:
    batch_size: int
    processing_time_ms: float
    throughput_ops_per_sec: float
    time_per_option_us: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_utilization_percent: float = 0.0

class PerformanceAnalyzer:
    def __init__(self, gpu_interface):
        self.gpu_interface = gpu_interface
        self.metrics_history = []
        
    def get_gpu_utilization(self):
        """Get GPU utilization using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    gpu_util, memory_used = lines[0].split(', ')
                    return float(gpu_util), float(memory_used)
        except:
            pass
        return 0.0, 0.0
        
    def benchmark_comprehensive(self, batch_sizes=[100, 500, 1000, 5000, 10000], 
                               trials=5):
        """Comprehensive performance benchmarking"""
        print("🔬 Comprehensive Performance Analysis")
        print("=" * 50)
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\n📏 Testing batch size: {batch_size:,}")
            
            # Generate test data
            test_options = self._generate_test_options(batch_size)
            market_data = {'TEST': {'spot_price': 100}}
            
            batch_metrics = []
            
            for trial in range(trials):
                # Clear caches
                if hasattr(self.gpu_interface, 'data_cache'):
                    self.gpu_interface.data_cache.clear()
                
                # Measure system state before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                cpu_before = psutil.cpu_percent(interval=None)
                gpu_util_before, gpu_mem_before = self.get_gpu_utilization()
                
                # Execute benchmark
                start_time = time.perf_counter()
                processed_count = self.gpu_interface.process_portfolio_options(
                    test_options, market_data
                )
                end_time = time.perf_counter()
                
                # Measure system state after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                cpu_after = psutil.cpu_percent(interval=0.1)
                gpu_util_after, gpu_mem_after = self.get_gpu_utilization()
                
                # Calculate metrics
                processing_time = (end_time - start_time) * 1000  # ms
                throughput = processed_count / (processing_time / 1000)  # ops/sec
                time_per_option = processing_time * 1000 / processed_count  # µs
                
                metrics = PerformanceMetrics(
                    batch_size=batch_size,
                    processing_time_ms=processing_time,
                    throughput_ops_per_sec=throughput,
                    time_per_option_us=time_per_option,
                    memory_usage_mb=memory_after - memory_before,
                    cpu_usage_percent=cpu_after - cpu_before,
                    gpu_utilization_percent=gpu_util_after - gpu_util_before
                )
                
                batch_metrics.append(metrics)
                
                print(f"  Trial {trial+1}: {processing_time:.2f}ms, "
                      f"{throughput:,.0f} ops/sec, {time_per_option:.2f}µs/opt")
            
            # Average results across trials
            avg_metrics = self._average_metrics(batch_metrics)
            results.append(avg_metrics)
            
            print(f"  Average: {avg_metrics.processing_time_ms:.2f}ms, "
                  f"{avg_metrics.throughput_ops_per_sec:,.0f} ops/sec")
        
        self.metrics_history.extend(results)
        return results
    
    def _generate_test_options(self, count):
        """Generate test options for benchmarking"""
        options = []
        for i in range(count):
            options.append({
                'symbol': 'TEST',
                'spot_price': 100 + (i % 50),
                'strike': 105 + (i % 40), 
                'time_to_expiry': 0.25 + (i % 12) * 0.083,  # 3 months to 1 year
                'risk_free_rate': 0.05,
                'implied_volatility': 0.15 + (i % 50) * 0.01,  # 15% to 65%
                'is_call': (i % 2 == 0),
                'market_price': 5.0 + (i % 20)
            })
        return options
    
    def _average_metrics(self, metrics_list):
        """Average a list of PerformanceMetrics"""
        if not metrics_list:
            return None
            
        avg = PerformanceMetrics(
            batch_size=metrics_list[0].batch_size,
            processing_time_ms=sum(m.processing_time_ms for m in metrics_list) / len(metrics_list),
            throughput_ops_per_sec=sum(m.throughput_ops_per_sec for m in metrics_list) / len(metrics_list),
            time_per_option_us=sum(m.time_per_option_us for m in metrics_list) / len(metrics_list),
            memory_usage_mb=sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list),
            cpu_usage_percent=sum(m.cpu_usage_percent for m in metrics_list) / len(metrics_list),
            gpu_utilization_percent=sum(m.gpu_utilization_percent for m in metrics_list) / len(metrics_list)
        )
        return avg
    
    def generate_performance_charts(self, save_dir="performance_charts"):
        """Generate performance visualization charts"""
        if not self.metrics_history:
            print("No performance data available for charting")
            return
            
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data
        batch_sizes = [m.batch_size for m in self.metrics_history]
        throughput = [m.throughput_ops_per_sec for m in self.metrics_history]
        time_per_option = [m.time_per_option_us for m in self.metrics_history]
        
        # Throughput vs Batch Size
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(batch_sizes, throughput, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (options/sec)')
        plt.title('GPU AAD Throughput Scaling')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Time per Option vs Batch Size  
        plt.subplot(2, 2, 2)
        plt.plot(batch_sizes, time_per_option, 'r-s', linewidth=2, markersize=6)
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Option (µs)')
        plt.title('Processing Time per Option')
        plt.grid(True, alpha=0.3)
        
        # Memory Usage
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        plt.subplot(2, 2, 3)
        plt.plot(batch_sizes, memory_usage, 'g-^', linewidth=2, markersize=6)
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Consumption')
        plt.grid(True, alpha=0.3)
        
        # Efficiency (theoretical peak vs actual)
        theoretical_peak = 20000000  # 20M options/sec theoretical
        efficiency = [t/theoretical_peak * 100 for t in throughput]
        plt.subplot(2, 2, 4)
        plt.plot(batch_sizes, efficiency, 'm-d', linewidth=2, markersize=6)
        plt.xlabel('Batch Size')
        plt.ylabel('Efficiency (%)')
        plt.title('GPU Utilization Efficiency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance charts saved to: {save_dir}/performance_analysis.png")
    
    def export_results(self, filename="performance_results.csv"):
        """Export performance results to CSV"""
        if not self.metrics_history:
            return
            
        df = pd.DataFrame([
            {
                'batch_size': m.batch_size,
                'processing_time_ms': m.processing_time_ms,
                'throughput_ops_per_sec': m.throughput_ops_per_sec,
                'time_per_option_us': m.time_per_option_us,
                'memory_usage_mb': m.memory_usage_mb,
                'cpu_usage_percent': m.cpu_usage_percent,
                'gpu_utilization_percent': m.gpu_utilization_percent
            }
            for m in self.metrics_history
        ])
        
        df.to_csv(filename, index=False)
        print(f"📄 Results exported to: {filename}")
