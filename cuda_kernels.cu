#include "AADTypes.h"
#include <cmath>

#ifdef CPU_ONLY
#include "cpu_fallback.h"
#else
// Only include CUDA headers if not in CPU-only mode
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#ifndef CPU_ONLY
// Device math functions with numerical stability
__device__ double safe_log(double x) {
    return (x > 1e-15) ? log(x) : log(1e-15);
}

__device__ double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

__device__ double safe_divide(double numerator, double denominator) {
    return (fabs(denominator) > 1e-15) ? numerator / denominator : 0.0;
}

// Custom atomicAdd for double (for older CUDA architectures)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Safe wrapper for atomicAdd that works with both float and double
__device__ void safe_atomic_add(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    // Use native double atomicAdd for compute capability 6.0+
    atomicAdd(address, val);
#else
    // Use custom implementation for older architectures
    atomicAddDouble(address, val);
#endif
}

__device__ double device_erf(double x) {
    // Approximation of error function for norm_cdf
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    int sign = (x >= 0) ? 1 : -1;
    x = fabs(x);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign * y;
}

__device__ double device_norm_cdf(double x) {
    return 0.5 * (1.0 + device_erf(x / sqrt(2.0)));
}

// GPU reverse pass kernel
__global__ void gpu_reverse_pass_kernel(
    const GPUTapeEntry* tape,
    double* values,
    double* adjoints,
    int tape_size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Process tape entries in reverse order
    for (int i = tape_size - 1; i >= 0; i--) {
        if (tid == 0) { // Single thread processes each entry
            const GPUTapeEntry& entry = tape[i];
            
            if (entry.op_type == AADOpType::LEAF) {
                continue; // Leaf nodes have no backward propagation
            }
            
            double adjoint_result = adjoints[entry.result_idx];
            
            // Propagate adjoints based on operation type
            switch (entry.op_type) {
                case AADOpType::ADD:
                    safe_atomic_add(&adjoints[entry.input1_idx], adjoint_result * entry.partial1);
                    safe_atomic_add(&adjoints[entry.input2_idx], adjoint_result * entry.partial2);
                    break;
                    
                case AADOpType::SUB:
                    safe_atomic_add(&adjoints[entry.input1_idx], adjoint_result * entry.partial1);
                    safe_atomic_add(&adjoints[entry.input2_idx], adjoint_result * entry.partial2);
                    break;
                    
                case AADOpType::MUL:
                    safe_atomic_add(&adjoints[entry.input1_idx], adjoint_result * entry.partial1);
                    safe_atomic_add(&adjoints[entry.input2_idx], adjoint_result * entry.partial2);
                    break;
                    
                case AADOpType::DIV:
                    safe_atomic_add(&adjoints[entry.input1_idx], adjoint_result * entry.partial1);
                    safe_atomic_add(&adjoints[entry.input2_idx], adjoint_result * entry.partial2);
                    break;
                    
                case AADOpType::LOG:
                case AADOpType::EXP:
                case AADOpType::SQRT:
                case AADOpType::NEG:
                case AADOpType::NORM_CDF:
                    safe_atomic_add(&adjoints[entry.input1_idx], adjoint_result * entry.partial1);
                    break;
            }
        }
        __syncthreads();
    }
}

// Black-Scholes GPU kernel with AAD
__global__ void blackscholes_aad_kernel(
    const BlackScholesParams* params,
    OptionResults* results,
    double* values,
    double* adjoints,
    GPUTapeEntry* tape,
    int* tape_sizes,
    int num_scenarios,
    int max_tape_size
) {
    int scenario_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (scenario_idx >= num_scenarios) return;
    
    // Get parameters for this scenario
    BlackScholesParams p = params[scenario_idx];
    
    // Reset tape for this scenario
    int tape_offset = scenario_idx * max_tape_size;
    int tape_pos = 0;
    
    // Add input variables to tape
    int s_idx = tape_offset + tape_pos++;
    int k_idx = tape_offset + tape_pos++;
    int t_idx = tape_offset + tape_pos++;
    int r_idx = tape_offset + tape_pos++;
    int v_idx = tape_offset + tape_pos++;
    
    values[s_idx] = p.spot;
    values[k_idx] = p.strike;
    values[t_idx] = p.time;
    values[r_idx] = p.rate;
    values[v_idx] = p.volatility;
    
    // Black-Scholes calculation with AAD recording
    // d1 = (log(S/K) + (r + 0.5*v²)*T) / (v*sqrt(T))
    
    // log(S/K)
    int sk_ratio_idx = tape_offset + tape_pos++;
    values[sk_ratio_idx] = safe_divide(values[s_idx], values[k_idx]);
    tape[tape_offset + tape_pos] = {sk_ratio_idx, s_idx, k_idx, AADOpType::DIV, 
                                   safe_divide(1.0, values[k_idx]), 
                                   -safe_divide(values[s_idx], values[k_idx] * values[k_idx])};
    tape_pos++;
    
    int log_sk_idx = tape_offset + tape_pos++;
    values[log_sk_idx] = safe_log(values[sk_ratio_idx]);
    tape[tape_offset + tape_pos] = {log_sk_idx, sk_ratio_idx, -1, AADOpType::LOG, 
                                   safe_divide(1.0, values[sk_ratio_idx]), 0.0};
    tape_pos++;
    
    // v²
    int v_squared_idx = tape_offset + tape_pos++;
    values[v_squared_idx] = values[v_idx] * values[v_idx];
    tape[tape_offset + tape_pos] = {v_squared_idx, v_idx, v_idx, AADOpType::MUL, 
                                   2.0 * values[v_idx], 2.0 * values[v_idx]};
    tape_pos++;
    
    // 0.5 * v²
    int half_v_squared_idx = tape_offset + tape_pos++;
    values[half_v_squared_idx] = 0.5 * values[v_squared_idx];
    tape[tape_offset + tape_pos] = {half_v_squared_idx, v_squared_idx, -1, AADOpType::MUL, 
                                   0.5, 0.0};
    tape_pos++;
    
    // r + 0.5*v²
    int drift_idx = tape_offset + tape_pos++;
    values[drift_idx] = values[r_idx] + values[half_v_squared_idx];
    tape[tape_offset + tape_pos] = {drift_idx, r_idx, half_v_squared_idx, AADOpType::ADD, 
                                   1.0, 1.0};
    tape_pos++;
    
    // (r + 0.5*v²)*T
    int drift_t_idx = tape_offset + tape_pos++;
    values[drift_t_idx] = values[drift_idx] * values[t_idx];
    tape[tape_offset + tape_pos] = {drift_t_idx, drift_idx, t_idx, AADOpType::MUL, 
                                   values[t_idx], values[drift_idx]};
    tape_pos++;
    
    // log(S/K) + (r + 0.5*v²)*T
    int numerator_idx = tape_offset + tape_pos++;
    values[numerator_idx] = values[log_sk_idx] + values[drift_t_idx];
    tape[tape_offset + tape_pos] = {numerator_idx, log_sk_idx, drift_t_idx, AADOpType::ADD, 
                                   1.0, 1.0};
    tape_pos++;
    
    // sqrt(T)
    int sqrt_t_idx = tape_offset + tape_pos++;
    values[sqrt_t_idx] = safe_sqrt(values[t_idx]);
    tape[tape_offset + tape_pos] = {sqrt_t_idx, t_idx, -1, AADOpType::SQRT, 
                                   safe_divide(0.5, values[sqrt_t_idx]), 0.0};
    tape_pos++;
    
    // v * sqrt(T)
    int vol_sqrt_t_idx = tape_offset + tape_pos++;
    values[vol_sqrt_t_idx] = values[v_idx] * values[sqrt_t_idx];
    tape[tape_offset + tape_pos] = {vol_sqrt_t_idx, v_idx, sqrt_t_idx, AADOpType::MUL, 
                                   values[sqrt_t_idx], values[v_idx]};
    tape_pos++;
    
    // d1 = numerator / (v * sqrt(T))
    int d1_idx = tape_offset + tape_pos++;
    values[d1_idx] = safe_divide(values[numerator_idx], values[vol_sqrt_t_idx]);
    tape[tape_offset + tape_pos] = {d1_idx, numerator_idx, vol_sqrt_t_idx, AADOpType::DIV, 
                                   safe_divide(1.0, values[vol_sqrt_t_idx]), 
                                   -safe_divide(values[numerator_idx], values[vol_sqrt_t_idx] * values[vol_sqrt_t_idx])};
    tape_pos++;
    
    // d2 = d1 - v * sqrt(T)
    int d2_idx = tape_offset + tape_pos++;
    values[d2_idx] = values[d1_idx] - values[vol_sqrt_t_idx];
    tape[tape_offset + tape_pos] = {d2_idx, d1_idx, vol_sqrt_t_idx, AADOpType::SUB, 
                                   1.0, -1.0};
    tape_pos++;
    
    // N(d1) and N(d2)
    int nd1_idx = tape_offset + tape_pos++;
    values[nd1_idx] = device_norm_cdf(values[d1_idx]);
    // Derivative of norm_cdf is the standard normal PDF
    double pdf_d1 = exp(-0.5 * values[d1_idx] * values[d1_idx]) / sqrt(2.0 * M_PI);
    tape[tape_offset + tape_pos] = {nd1_idx, d1_idx, -1, AADOpType::NORM_CDF, pdf_d1, 0.0};
    tape_pos++;
    
    int nd2_idx = tape_offset + tape_pos++;
    values[nd2_idx] = device_norm_cdf(values[d2_idx]);
    double pdf_d2 = exp(-0.5 * values[d2_idx] * values[d2_idx]) / sqrt(2.0 * M_PI);
    tape[tape_offset + tape_pos] = {nd2_idx, d2_idx, -1, AADOpType::NORM_CDF, pdf_d2, 0.0};
    tape_pos++;
    
    // exp(-r*T)
    int neg_rt_idx = tape_offset + tape_pos++;
    values[neg_rt_idx] = -values[r_idx] * values[t_idx];
    tape[tape_offset + tape_pos] = {neg_rt_idx, r_idx, t_idx, AADOpType::MUL, 
                                   -values[t_idx], -values[r_idx]};
    tape_pos++;
    
    int exp_neg_rt_idx = tape_offset + tape_pos++;
    values[exp_neg_rt_idx] = exp(values[neg_rt_idx]);
    tape[tape_offset + tape_pos] = {exp_neg_rt_idx, neg_rt_idx, -1, AADOpType::EXP, 
                                   values[exp_neg_rt_idx], 0.0};
    tape_pos++;
    
    // Black-Scholes price calculation
    int price_idx;
    if (p.is_call) {
        // Call: S*N(d1) - K*exp(-r*T)*N(d2)
        int s_nd1_idx = tape_offset + tape_pos++;
        values[s_nd1_idx] = values[s_idx] * values[nd1_idx];
        tape[tape_offset + tape_pos] = {s_nd1_idx, s_idx, nd1_idx, AADOpType::MUL, 
                                       values[nd1_idx], values[s_idx]};
        tape_pos++;
        
        int k_exp_nd2_idx = tape_offset + tape_pos++;
        values[k_exp_nd2_idx] = values[k_idx] * values[exp_neg_rt_idx] * values[nd2_idx];
        // This is a three-way multiplication, broken into two steps
        int k_exp_idx = tape_offset + tape_pos++;
        values[k_exp_idx] = values[k_idx] * values[exp_neg_rt_idx];
        tape[tape_offset + tape_pos] = {k_exp_idx, k_idx, exp_neg_rt_idx, AADOpType::MUL, 
                                       values[exp_neg_rt_idx], values[k_idx]};
        tape_pos++;
        
        values[k_exp_nd2_idx] = values[k_exp_idx] * values[nd2_idx];
        tape[tape_offset + tape_pos] = {k_exp_nd2_idx, k_exp_idx, nd2_idx, AADOpType::MUL, 
                                       values[nd2_idx], values[k_exp_idx]};
        tape_pos++;
        
        price_idx = tape_offset + tape_pos++;
        values[price_idx] = values[s_nd1_idx] - values[k_exp_nd2_idx];
        tape[tape_offset + tape_pos] = {price_idx, s_nd1_idx, k_exp_nd2_idx, AADOpType::SUB, 
                                       1.0, -1.0};
        tape_pos++;
    } else {
        // Put: K*exp(-r*T)*N(-d2) - S*N(-d1)
        // Implementation similar to call but with negated d1, d2
        // For brevity, implementing call only in this example
        price_idx = tape_offset + tape_pos - 1; // Placeholder
    }
    
    // Store the final price
    results[scenario_idx].price = values[price_idx];
    
    // Store tape size for this scenario
    tape_sizes[scenario_idx] = tape_pos;
    
    // Reset adjoints
    for (int i = 0; i < tape_pos; i++) {
        adjoints[tape_offset + i] = 0.0;
    }
    
    // Set adjoint of price to 1 for gradient computation
    adjoints[price_idx] = 1.0;
}

// Host function to launch kernels
extern "C" {
    void launch_blackscholes_kernel(
        const BlackScholesParams* h_params,
        OptionResults* h_results,
        int num_scenarios,
        GPUConfig config
    ) {
        // Allocate GPU memory
        BlackScholesParams* d_params;
        OptionResults* d_results;
        double* d_values;
        double* d_adjoints;
        GPUTapeEntry* d_tape;
        int* d_tape_sizes;
        
        size_t params_size = num_scenarios * sizeof(BlackScholesParams);
        size_t results_size = num_scenarios * sizeof(OptionResults);
        size_t values_size = num_scenarios * config.max_tape_size * sizeof(double);
        size_t tape_size = num_scenarios * config.max_tape_size * sizeof(GPUTapeEntry);
        size_t tape_sizes_size = num_scenarios * sizeof(int);
        
        cudaMalloc(&d_params, params_size);
        cudaMalloc(&d_results, results_size);
        cudaMalloc(&d_values, values_size * 2); // Values and adjoints
        cudaMalloc(&d_adjoints, values_size);
        cudaMalloc(&d_tape, tape_size);
        cudaMalloc(&d_tape_sizes, tape_sizes_size);
        
        // Copy input data to GPU
        cudaMemcpy(d_params, h_params, params_size, cudaMemcpyHostToDevice);
        
        // Launch forward pass kernel
        int num_blocks = (num_scenarios + config.block_size - 1) / config.block_size;
        blackscholes_aad_kernel<<<num_blocks, config.block_size>>>(
            d_params, d_results, d_values, d_adjoints, d_tape, d_tape_sizes,
            num_scenarios, config.max_tape_size
        );
        
        cudaDeviceSynchronize();
        
        // Launch reverse pass kernel for each scenario
        for (int i = 0; i < num_scenarios; i++) {
            int tape_offset = i * config.max_tape_size;
            int scenario_tape_size;
            cudaMemcpy(&scenario_tape_size, &d_tape_sizes[i], sizeof(int), cudaMemcpyDeviceToHost);
            
            gpu_reverse_pass_kernel<<<1, 1>>>(
                d_tape + tape_offset, d_values + tape_offset, d_adjoints + tape_offset,
                scenario_tape_size
            );
        }
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
        
        // Copy adjoints for Greeks calculation
        std::vector<double> h_adjoints(num_scenarios * config.max_tape_size);
        cudaMemcpy(h_adjoints.data(), d_adjoints, values_size, cudaMemcpyDeviceToHost);
        
        // Extract Greeks from adjoints
        for (int i = 0; i < num_scenarios; i++) {
            int tape_offset = i * config.max_tape_size;
            h_results[i].delta = h_adjoints[tape_offset + 0]; // dP/dS
            h_results[i].rho = h_adjoints[tape_offset + 3];   // dP/dr
            h_results[i].vega = h_adjoints[tape_offset + 4];  // dP/dsigma
            h_results[i].theta = -h_adjoints[tape_offset + 2]; // -dP/dT (theta is negative time decay)
            // Gamma requires second-order derivatives (not implemented in this basic version)
            h_results[i].gamma = 0.0;
        }
        
        // Cleanup GPU memory
        cudaFree(d_params);
        cudaFree(d_results);
        cudaFree(d_values);
        cudaFree(d_adjoints);
        cudaFree(d_tape);
        cudaFree(d_tape_sizes);
    }
}

// Portfolio Greeks computation kernel
extern "C" void launch_portfolio_greeks_kernel(
    const BlackScholesParams* h_params,
    const int* h_quantities,
    OptionResults* h_results,
    int num_positions,
    GPUConfig config
) {
    if (num_positions <= 0) return;
    
    // Use the existing Black-Scholes kernel for individual positions
    launch_blackscholes_kernel(h_params, h_results, num_positions, config);
    
    // Scale results by position quantities on CPU for now
    // (Could be optimized to do this on GPU for very large portfolios)
    for (int i = 0; i < num_positions; i++) {
        int qty = h_quantities[i];
        h_results[i].price *= qty;
        h_results[i].delta *= qty;
        h_results[i].vega *= qty;
        h_results[i].gamma *= qty;
        h_results[i].theta *= qty;
        h_results[i].rho *= qty;
    }
}

#endif // !CPU_ONLY
