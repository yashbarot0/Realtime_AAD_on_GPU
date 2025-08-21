#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include "AADTypes.h"

#ifdef CPU_ONLY
// CPU-only mode - define CUDA stubs
typedef int cudaError_t;
#define cudaSuccess 0
struct cudaDeviceProp { char name[256]; };
inline int cudaGetDeviceCount(int* count) { *count = 0; return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp* prop, int device) { return 0; }
#else
// Include CUDA headers if available
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Fallback definitions if CUDA headers not available but not CPU_ONLY
typedef int cudaError_t;
#define cudaSuccess 0
struct cudaDeviceProp { char name[256]; };
extern "C" {
    int cudaGetDeviceCount(int* count);
    int cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
}
#endif
#endif

// External kernel launcher
extern "C" void launch_blackscholes_kernel(
    const BlackScholesParams* h_params,
    OptionResults* h_results,
    int num_scenarios,
    GPUConfig config
);

void test_single_option() {
    std::cout << "=== Single Black-Scholes Option Test ===" << std::endl;
    
    BlackScholesParams params;
    params.spot = 100.0;
    params.strike = 105.0;
    params.time = 0.25;  // 3 months
    params.rate = 0.05;
    params.volatility = 0.2;
    params.is_call = true;
    
    OptionResults result;
    GPUConfig config;
    
    launch_blackscholes_kernel(&params, &result, 1, config);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Spot: $" << params.spot << std::endl;
    std::cout << "Strike: $" << params.strike << std::endl;
    std::cout << "Time: " << params.time << " years" << std::endl;
    std::cout << "Rate: " << (params.rate * 100) << "%" << std::endl;
    std::cout << "Volatility: " << (params.volatility * 100) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "Price: $" << result.price << std::endl;
    std::cout << "Delta: " << result.delta << std::endl;
    std::cout << "Vega: " << result.vega << std::endl;
    std::cout << "Gamma: " << result.gamma << std::endl;
    std::cout << "Theta: " << result.theta << std::endl;
    std::cout << "Rho: " << result.rho << std::endl;
    std::cout << std::endl;
}

void benchmark_gpu_performance() {
    std::cout << "=== GPU Performance Benchmark ===" << std::endl;
    
    const int num_scenarios = 1000000;
    std::vector<BlackScholesParams> params(num_scenarios);
    std::vector<OptionResults> results(num_scenarios);
    
    // Generate random parameters
    for (int i = 0; i < num_scenarios; i++) {
        params[i].spot = 90.0 + (i % 20);  // 90-110
        params[i].strike = 95.0 + (i % 20); // 95-115
        params[i].time = 0.1 + (i % 10) * 0.1; // 0.1 to 1.0 years
        params[i].rate = 0.02 + (i % 5) * 0.01; // 2% to 6%
        params[i].volatility = 0.15 + (i % 10) * 0.05; // 15% to 65%
        params[i].is_call = (i % 2 == 0);
    }
    
    GPUConfig config;
    config.max_scenarios = num_scenarios;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    launch_blackscholes_kernel(params.data(), results.data(), num_scenarios, config);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double total_time_ms = duration.count() / 1000.0;
    double avg_time_per_option_us = duration.count() / (double)num_scenarios;
    double throughput = num_scenarios / (duration.count() / 1000000.0);
    
    std::cout << "Processed " << num_scenarios << " options" << std::endl;
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average time per option: " << avg_time_per_option_us << " µs" << std::endl;
    std::cout << "Throughput: " << (int)throughput << " options/second" << std::endl;
    
    // Show sample results
    std::cout << "\nSample results:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Option " << i << ": Price=$" << std::fixed << std::setprecision(4) 
                  << results[i].price << ", Delta=" << results[i].delta 
                  << ", Vega=" << results[i].vega << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "GPU Black-Scholes AAD Implementation" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Check CUDA availability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Print GPU information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
    
    try {
        // Run tests
        test_single_option();
        benchmark_gpu_performance();
        
        std::cout << "All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
