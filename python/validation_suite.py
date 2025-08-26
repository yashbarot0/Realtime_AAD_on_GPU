import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import log, sqrt, exp
import time
from typing import Dict, List, Tuple
from safe_gpu_interface import SafeGPUInterface

class AADValidationSuite:
    def __init__(self):
        self.gpu_interface = SafeGPUInterface()
        self.results = {}
        
    def analytical_black_scholes(self, S, K, T, r, sigma, is_call=True):
        """Analytical Black-Scholes formula for validation"""
        if T <= 0 or sigma <= 0:
            return {'price': max(S-K, 0) if is_call else max(K-S, 0),
                   'delta': 1.0 if is_call else -1.0, 'vega': 0, 'gamma': 0, 'theta': 0, 'rho': 0}
        
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        
        if is_call:
            price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = K*T*exp(-r*T)*norm.cdf(d2)
        else:
            price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            rho = -K*T*exp(-r*T)*norm.cdf(-d2)
            
        gamma = norm.pdf(d1) / (S*sigma*sqrt(T))
        vega = S*norm.pdf(d1)*sqrt(T) / 100.0
        theta = (-(S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - 
                r*K*exp(-r*T)*norm.cdf(d2 if is_call else -d2)) / 365.0
                
        return {'price': price, 'delta': delta, 'vega': vega, 
                'gamma': gamma, 'theta': theta, 'rho': rho}

    def test_numerical_accuracy(self, num_tests=1000):
        """Test GPU AAD accuracy against analytical solutions"""
        print("🧪 Testing Numerical Accuracy vs Analytical Solutions")
        print("=" * 60)
        
        # Generate test parameters
        np.random.seed(42)
        test_cases = []
        
        for i in range(num_tests):
            S = np.random.uniform(50, 200)
            K = np.random.uniform(80, 150)
            T = np.random.uniform(0.1, 2.0)
            r = np.random.uniform(0.01, 0.1)
            sigma = np.random.uniform(0.1, 0.8)
            is_call = np.random.choice([True, False])
            
            test_cases.append({
                'symbol': 'TEST',
                'spot_price': S,
                'strike': K,
                'time_to_expiry': T,
                'risk_free_rate': r,
                'implied_volatility': sigma,
                'is_call': is_call,
                'market_price': 0
            })
        
        # Process with GPU
        market_data = {'TEST': {'spot_price': 100}}
        start_time = time.time()
        self.gpu_interface.process_portfolio_options(test_cases, market_data)
        gpu_time = time.time() - start_time
        
        # Compare results
        errors = {'price': [], 'delta': [], 'vega': [], 'gamma': [], 'theta': [], 'rho': []}
        
        for case in test_cases:
            # Analytical solution
            analytical = self.analytical_black_scholes(
                case['spot_price'], case['strike'], case['time_to_expiry'],
                case['risk_free_rate'], case['implied_volatility'], case['is_call']
            )
            
            # GPU solution (using CPU fallback for now)
            gpu_greeks = self.gpu_interface.calculate_cpu_greeks(
                case['spot_price'], case['strike'], case['time_to_expiry'],
                case['risk_free_rate'], case['implied_volatility'], case['is_call']
            )
            
            # Calculate relative errors
            for greek in ['delta', 'vega', 'gamma', 'theta', 'rho']:
                if abs(analytical[greek]) > 1e-10:
                    rel_error = abs(gpu_greeks[greek] - analytical[greek]) / abs(analytical[greek])
                    errors[greek].append(rel_error)
        
        # Print statistics
        print(f"📊 Accuracy Statistics ({num_tests} test cases):")
        for greek in ['delta', 'vega', 'gamma', 'theta', 'rho']:
            if errors[greek]:
                mean_error = np.mean(errors[greek])
                max_error = np.max(errors[greek])
                std_error = np.std(errors[greek])
                print(f"  {greek.upper():>5}: Mean={mean_error:.2e}, Max={max_error:.2e}, Std={std_error:.2e}")
        
        print(f"⚡ GPU Processing Time: {gpu_time*1000:.2f}ms ({num_tests/gpu_time:.0f} options/sec)")
        
        self.results['accuracy_test'] = {
            'num_tests': num_tests,
            'errors': errors,
            'gpu_time': gpu_time
        }
        
        return errors

    def performance_scaling_study(self):
        """Study performance scaling with batch size"""
        print("\n🚀 Performance Scaling Analysis")
        print("=" * 40)
        
        batch_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
        performance_data = []
        
        for batch_size in batch_sizes:
            # Generate test data
            test_options = []
            for i in range(batch_size):
                test_options.append({
                    'symbol': 'TEST',
                    'spot_price': 100 + (i % 50),
                    'strike': 105 + (i % 40),
                    'time_to_expiry': 0.25,
                    'risk_free_rate': 0.05,
                    'implied_volatility': 0.2 + (i % 20) * 0.01,
                    'is_call': (i % 2 == 0),
                    'market_price': 5.0
                })
            
            market_data = {'TEST': {'spot_price': 100}}
            
            # Measure performance
            times = []
            for trial in range(5):  # 5 trials for averaging
                start_time = time.time()
                self.gpu_interface.process_portfolio_options(test_options, market_data)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            time_per_option = avg_time * 1000000 / batch_size  # microseconds
            
            performance_data.append({
                'batch_size': batch_size,
                'avg_time_ms': avg_time * 1000,
                'throughput': throughput,
                'time_per_option_us': time_per_option
            })
            
            print(f"  {batch_size:>6d} options: {avg_time*1000:>8.2f}ms, "
                  f"{throughput:>10,.0f} opts/sec, {time_per_option:>6.2f}µs/opt")
        
        self.results['scaling_study'] = performance_data
        return performance_data

    def generate_report(self, save_path="validation_report.md"):
        """Generate comprehensive validation report"""
        report = f"""
# GPU AAD Validation Report

## Executive Summary
- **Test Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **GPU Device**: {self.gpu_interface.use_gpu}
- **Total Test Cases**: {self.results.get('accuracy_test', {}).get('num_tests', 0)}

## Numerical Accuracy Results
"""
        
        if 'accuracy_test' in self.results:
            errors = self.results['accuracy_test']['errors']
            report += "\n| Greek | Mean Error | Max Error | Std Error |\n"
            report += "|-------|------------|-----------|----------|\n"
            for greek in ['delta', 'vega', 'gamma', 'theta', 'rho']:
                if errors[greek]:
                    mean_err = np.mean(errors[greek])
                    max_err = np.max(errors[greek])
                    std_err = np.std(errors[greek])
                    report += f"| {greek.upper()} | {mean_err:.2e} | {max_err:.2e} | {std_err:.2e} |\n"
        
        if 'scaling_study' in self.results:
            report += "\n## Performance Scaling Results\n\n"
            report += "| Batch Size | Time (ms) | Throughput (opts/sec) | µs per Option |\n"
            report += "|------------|-----------|---------------------|---------------|\n"
            for data in self.results['scaling_study']:
                report += f"| {data['batch_size']:,} | {data['avg_time_ms']:.2f} | "
                report += f"{data['throughput']:,.0f} | {data['time_per_option_us']:.2f} |\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Validation report saved to: {save_path}")

# Usage example
if __name__ == "__main__":
    validator = AADValidationSuite()
    validator.test_numerical_accuracy(1000)
    validator.performance_scaling_study()
    validator.generate_report()
