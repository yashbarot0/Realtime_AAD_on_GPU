#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

// ===== SAFE DEVICE MATH FUNCTIONS =====
__device__ double safe_log(double x) {
    return (x > 1e-15) ? log(x) : log(1e-15);
}

__device__ double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ double safe_exp(double x) {
    return exp(fmin(x, 700.0));
}

__device__ double safe_div(double a, double b) {
    return (fabs(b) > 1e-15) ? a / b : 0.0;
}

__device__ double device_norm_cdf(double x) {
    // Abramowitz and Stegun approximation
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;
    
    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x) / sqrt(2.0);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * exp(-x*x);
    
    return 0.5 * (1.0 + sign * y);
}

__device__ double device_norm_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

// ===== SIMPLIFIED BLACK-SCHOLES WITH FINITE DIFFERENCES =====
__device__ void blackscholes_simple(
    double S, double K, double T, double r, double sigma, bool is_call,
    double* price, double* delta, double* vega, double* gamma, double* theta, double* rho
) {
    // Clamp inputs to safe ranges
    S = fmax(S, 0.01);
    K = fmax(K, 0.01);
    T = fmax(T, 0.001);
    sigma = fmax(sigma, 0.01);
    
    // Calculate d1 and d2
    double sqrt_T = safe_sqrt(T);
    double d1 = (safe_log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt_T);
    double d2 = d1 - sigma*sqrt_T;
    
    // Calculate price
    double nd1 = device_norm_cdf(d1);
    double nd2 = device_norm_cdf(d2);
    double discount = safe_exp(-r*T);
    
    if (is_call) {
        *price = S * nd1 - K * discount * nd2;
    } else {
        *price = K * discount * (1.0 - nd2) - S * (1.0 - nd1);
    }
    
    // Calculate Greeks using analytical formulas (more stable than AAD for this demo)
    double pdf_d1 = device_norm_pdf(d1);
    
    // Delta
    if (is_call) {
        *delta = nd1;
    } else {
        *delta = nd1 - 1.0;
    }
    
    // Gamma
    *gamma = pdf_d1 / (S * sigma * sqrt_T);
    
    // Vega (per 1% change)
    *vega = S * pdf_d1 * sqrt_T / 100.0;
    
    // Theta (per day)
    if (is_call) {
        *theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T) - r * K * discount * nd2;
    } else {
        *theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * K * discount * (1.0 - nd2);
    }
    *theta /= 365.0;
    
    // Rho
    if (is_call) {
        *rho = K * T * discount * nd2;
    } else {
        *rho = -K * T * discount * (1.0 - nd2);
    }
}

// ===== MEMORY-SAFE BATCH KERNEL =====
__global__ void blackscholes_batch_kernel(
    const BlackScholesParams* params,
    OptionResults* results,
    int num_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx >= num_options) return;
    
    // Get input parameters
    const BlackScholesParams& p = params[idx];
    
    // Initialize result to zero
    OptionResults& r = results[idx];
    r.price = 0.0;
    r.delta = 0.0;
    r.vega = 0.0;
    r.gamma = 0.0;
    r.theta = 0.0;
    r.rho = 0.0;
    
    // Validate inputs
    if (p.spot <= 0.0 || p.strike <= 0.0 || p.time <= 0.0 || p.volatility <= 0.0) {
        return; // Skip invalid inputs
    }
    
    // Calculate using safe implementation
    blackscholes_simple(
        p.spot, p.strike, p.time, p.rate, p.volatility, p.is_call,
        &r.price, &r.delta, &r.vega, &r.gamma, &r.theta, &r.rho
    );
}

// ===== HOST LAUNCHER FUNCTION =====
extern "C" void launch_blackscholes_kernel(
    const BlackScholesParams* h_params,
    OptionResults* h_results,
    int num_scenarios,
    GPUConfig config
) {
    // Validate inputs
    if (!h_params || !h_results || num_scenarios <= 0) {
        printf("Invalid input parameters\n");
        return;
    }
    
    // GPU memory pointers
    BlackScholesParams* d_params = nullptr;
    OptionResults* d_results = nullptr;
    
    size_t params_size = num_scenarios * sizeof(BlackScholesParams);
    size_t results_size = num_scenarios * sizeof(OptionResults);
    
    cudaError_t err;
    
    // Check available memory
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        printf("Failed to get memory info: %s\n", cudaGetErrorString(err));
        return;
    }
    
    size_t required_mem = params_size + results_size;
    if (required_mem > free_mem) {
        printf("Insufficient GPU memory: need %zu bytes, have %zu bytes\n", 
               required_mem, free_mem);
        return;
    }
    
    // Allocate GPU memory with error checking
    err = cudaMalloc(&d_params, params_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for params: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_results, results_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate GPU memory for results: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        return;
    }
    
    // Copy input data to GPU
    err = cudaMemcpy(d_params, h_params, params_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy params to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        cudaFree(d_results);
        return;
    }
    
    // Initialize results to zero
    err = cudaMemset(d_results, 0, results_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize results: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        cudaFree(d_results);
        return;
    }
    
    // Configure kernel launch
    int block_size = min(config.block_size, 512); // Cap at 512 for safety
    int grid_size = (num_scenarios + block_size - 1) / block_size;
    
    // Launch kernel
    blackscholes_batch_kernel<<<grid_size, block_size>>>(
        d_params, d_results, num_scenarios
    );
    
    // Check for kernel launch error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        cudaFree(d_results);
        return;
    }
    
    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        cudaFree(d_results);
        return;
    }
    
    // Copy results back to host
    err = cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy results from GPU: %s\n", cudaGetErrorString(err));
    }
    
    // Clean up GPU memory
    cudaFree(d_params);
    cudaFree(d_results);
}
