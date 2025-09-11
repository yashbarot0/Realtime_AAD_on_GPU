#include "../include/device_aad.cuh"
#include "../include/AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// ===== DEVICE MATH FUNCTIONS =====
__device__ double device_safe_log(double x) {
    return log(fmax(x, 1e-15));
}

__device__ double device_safe_exp(double x) {
    return exp(fmin(x, 700.0));
}

__device__ double device_safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ double device_norm_cdf(double x) {
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

// ===== DEVICE AAD TAPE METHODS =====
__device__ int DeviceAADTape::record_variable(double value) {
    int idx = atomicAdd(current_size, 1);
    if (idx < max_size) {
        values[idx] = value;
        adjoints[idx] = 0.0;
        entries[idx] = {idx, -1, -1, 0, 0.0, 0.0}; // LEAF
    }
    return idx;
}

__device__ int DeviceAADTape::record_operation(int op, int in1, int in2, double p1, double p2, double result_val) {
    int idx = atomicAdd(current_size, 1);
    if (idx < max_size) {
        values[idx] = result_val;
        adjoints[idx] = 0.0;
        entries[idx] = {idx, in1, in2, op, p1, p2};
    }
    return idx;
}

__device__ void DeviceAADTape::propagate_adjoints() {
    for (int i = *current_size - 1; i >= 0; i--) {
        DeviceTapeEntry& entry = entries[i];
        double adjoint = adjoints[i];

        if (entry.input1_idx >= 0) {
            atomicAdd(&adjoints[entry.input1_idx], adjoint * entry.partial1);
        }
        if (entry.input2_idx >= 0) {
            atomicAdd(&adjoints[entry.input2_idx], adjoint * entry.partial2);
        }
    }
}

// ===== DEVICE AAD NUMBER OPERATORS =====
__device__ DeviceAADNumber DeviceAADNumber::operator+(const DeviceAADNumber& other) const {
    if (!tape) return DeviceAADNumber(value + other.value);
    
    double result = value + other.value;
    int idx = tape->record_operation(1, tape_index, other.tape_index, 1.0, 1.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::operator-(const DeviceAADNumber& other) const {
    if (!tape) return DeviceAADNumber(value - other.value);
    
    double result = value - other.value;
    int idx = tape->record_operation(2, tape_index, other.tape_index, 1.0, -1.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::operator*(const DeviceAADNumber& other) const {
    if (!tape) return DeviceAADNumber(value * other.value);
    
    double result = value * other.value;
    int idx = tape->record_operation(3, tape_index, other.tape_index, other.value, value, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::operator/(const DeviceAADNumber& other) const {
    if (!tape) return DeviceAADNumber(value / other.value);
    
    double result = value / other.value;
    double partial1 = 1.0 / other.value;
    double partial2 = -value / (other.value * other.value);
    int idx = tape->record_operation(4, tape_index, other.tape_index, partial1, partial2, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::operator-() const {
    if (!tape) return DeviceAADNumber(-value);
    
    double result = -value;
    int idx = tape->record_operation(8, tape_index, -1, -1.0, 0.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::log() const {
    if (!tape) return DeviceAADNumber(device_safe_log(value));
    
    double result = device_safe_log(value);
    double partial = 1.0 / fmax(value, 1e-15);
    int idx = tape->record_operation(5, tape_index, -1, partial, 0.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::exp() const {
    if (!tape) return DeviceAADNumber(device_safe_exp(value));
    
    double result = device_safe_exp(value);
    int idx = tape->record_operation(6, tape_index, -1, result, 0.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::sqrt() const {
    if (!tape) return DeviceAADNumber(device_safe_sqrt(value));
    
    double result = device_safe_sqrt(value);
    double partial = 0.5 / fmax(result, 1e-15);
    int idx = tape->record_operation(7, tape_index, -1, partial, 0.0, result);
    return DeviceAADNumber(result, idx, tape);
}

__device__ DeviceAADNumber DeviceAADNumber::norm_cdf() const {
    if (!tape) return DeviceAADNumber(device_norm_cdf(value));
    
    double result = device_norm_cdf(value);
    double partial = device_norm_pdf(value);
    int idx = tape->record_operation(9, tape_index, -1, partial, 0.0, result);
    return DeviceAADNumber(result, idx, tape);
}

// ===== BLACK-SCHOLES WITH AAD =====
__device__ DeviceAADNumber blackscholes_aad(
    DeviceAADNumber S, DeviceAADNumber K, DeviceAADNumber T, 
    DeviceAADNumber r, DeviceAADNumber vol, bool is_call
) {
    // d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    DeviceAADNumber ln_SK = (S / K).log();
    DeviceAADNumber vol_squared = vol * vol;
    DeviceAADNumber half_vol_squared = vol_squared * DeviceAADNumber(0.5);
    DeviceAADNumber drift = r + half_vol_squared;
    DeviceAADNumber numerator = ln_SK + drift * T;
    DeviceAADNumber sqrt_T = T.sqrt();
    DeviceAADNumber denominator = vol * sqrt_T;
    DeviceAADNumber d1 = numerator / denominator;
    
    // d2 = d1 - sigma*sqrt(T)
    DeviceAADNumber d2 = d1 - denominator;
    
    // Calculate option price
    DeviceAADNumber nd1 = d1.norm_cdf();
    DeviceAADNumber nd2 = d2.norm_cdf();
    DeviceAADNumber discount = (r * T * DeviceAADNumber(-1.0)).exp();
    
    if (is_call) {
        return S * nd1 - K * discount * nd2;
    } else {
        DeviceAADNumber one_minus_nd1 = DeviceAADNumber(1.0) - nd1;
        DeviceAADNumber one_minus_nd2 = DeviceAADNumber(1.0) - nd2;
        return K * discount * one_minus_nd2 - S * one_minus_nd1;
    }
}

// ===== CUDA KERNEL =====
__global__ void blackscholes_aad_kernel(
    const BlackScholesParams* params,
    OptionResults* results,
    int num_options,
    double* d_tape_values,
    double* d_tape_adjoints,
    DeviceTapeEntry* d_tape_entries,
    int* d_tape_sizes,
    int max_tape_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_options) return;
    
    const BlackScholesParams& p = params[idx];
    OptionResults& result = results[idx];
    
    // Initialize tape for this thread
    int tape_offset = idx * max_tape_size;
    d_tape_sizes[idx] = 0;
    
    DeviceAADTape tape(
        d_tape_values + tape_offset,
        d_tape_adjoints + tape_offset, 
        d_tape_entries + tape_offset,
        &d_tape_sizes[idx],
        max_tape_size
    );
    
    // Create AAD variables
    DeviceAADNumber S(p.spot, tape.record_variable(p.spot), &tape);
    DeviceAADNumber K(p.strike, tape.record_variable(p.strike), &tape);  
    DeviceAADNumber T(p.time, tape.record_variable(p.time), &tape);
    DeviceAADNumber r(p.rate, tape.record_variable(p.rate), &tape);
    DeviceAADNumber vol(p.volatility, tape.record_variable(p.volatility), &tape);
    
    // Forward pass - compute option price
    DeviceAADNumber price = blackscholes_aad(S, K, T, r, vol, p.is_call);
    result.price = price.value;
    
    // Backward pass - compute Greeks
    // Reset adjoints
    for (int i = 0; i < d_tape_sizes[idx]; i++) {
        tape.adjoints[i] = 0.0;
    }
    
    // Set derivative of price w.r.t. itself to 1
    tape.adjoints[price.tape_index] = 1.0;
    
    // Propagate adjoints
    tape.propagate_adjoints();
    
    // Extract Greeks
    result.delta = tape.adjoints[S.tape_index];        // dP/dS
    result.vega = tape.adjoints[vol.tape_index];       // dP/dsigma  
    result.theta = -tape.adjoints[T.tape_index];       // -dP/dT
    result.rho = tape.adjoints[r.tape_index];          // dP/dr
    result.gamma = 0.0; // Second-order derivative requires more complex implementation
}

// ===== HOST LAUNCHER =====
extern "C" void launch_blackscholes_kernel(
    const BlackScholesParams* h_params,
    OptionResults* h_results,
    int num_scenarios,
    GPUConfig config
) 



{
    const int MAX_TAPE_SIZE = 100;
    const int MAX_BATCH_SIZE = 1000000; // Process in batches if needed
    
    // Process in batches if too large for memory
    int batches = (num_scenarios + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    
    for (int batch = 0; batch < batches; batch++) {
        int batch_start = batch * MAX_BATCH_SIZE;
        int batch_size = std::min(MAX_BATCH_SIZE, num_scenarios - batch_start);
        
        // Your existing kernel launch code here for each batch
        // ... (same as before but with batch_start and batch_size)

    
    // Device memory pointers
    BlackScholesParams* d_params;
    OptionResults* d_results;
    double* d_tape_values;
    double* d_tape_adjoints;
    DeviceTapeEntry* d_tape_entries;
    int* d_tape_sizes;
    
    // Calculate sizes
    size_t params_size = num_scenarios * sizeof(BlackScholesParams);
    size_t results_size = num_scenarios * sizeof(OptionResults);
    size_t tape_values_size = num_scenarios * MAX_TAPE_SIZE * sizeof(double);
    size_t tape_entries_size = num_scenarios * MAX_TAPE_SIZE * sizeof(DeviceTapeEntry);
    size_t tape_sizes_size = num_scenarios * sizeof(int);
    
    // Allocate GPU memory
    cudaMalloc(&d_params, params_size);
    cudaMalloc(&d_results, results_size);
    cudaMalloc(&d_tape_values, tape_values_size);
    cudaMalloc(&d_tape_adjoints, tape_values_size);
    cudaMalloc(&d_tape_entries, tape_entries_size);
    cudaMalloc(&d_tape_sizes, tape_sizes_size);
    
    // Copy input data
    cudaMemcpy(d_params, h_params, params_size, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, results_size);
    cudaMemset(d_tape_sizes, 0, tape_sizes_size);
    
    // Launch kernel
    int block_size = min(config.block_size, 256);
    int grid_size = (num_scenarios + block_size - 1) / block_size;
    
    blackscholes_aad_kernel<<<grid_size, block_size>>>(
        d_params, d_results, num_scenarios,
        d_tape_values, d_tape_adjoints, d_tape_entries, d_tape_sizes,
        MAX_TAPE_SIZE
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_params);
    cudaFree(d_results);
    cudaFree(d_tape_values);
    cudaFree(d_tape_adjoints);
    cudaFree(d_tape_entries);
    cudaFree(d_tape_sizes);

        
        std::cout << "Processed batch " << (batch + 1) << "/" << batches 
                  << " (" << batch_size << " options)" << std::endl;
    }
}






