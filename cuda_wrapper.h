#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#ifdef CPU_ONLY
#include "cpu_fallback.h"
#else
// Forward declarations for CUDA functions
extern "C" {
    void launch_blackscholes_kernel(
        const BlackScholesParams* h_params,
        OptionResults* h_results,
        int num_scenarios,
        GPUConfig config
    );

    void launch_portfolio_greeks_kernel(
        const BlackScholesParams* h_params,
        const int* h_quantities,
        OptionResults* h_results,
        int num_positions,
        GPUConfig config
    );
}
#endif

// Unified wrapper functions that handle CUDA/CPU fallback
inline void compute_portfolio_greeks(
    const BlackScholesParams* params,
    const int* quantities,
    OptionResults* results,
    int num_positions,
    GPUConfig config = {1000, 50000}
) {
#ifdef CPU_ONLY
    cpu_launch_portfolio_greeks_kernel(params, quantities, results, num_positions, config);
#else
    // Try CUDA first, fall back to CPU if needed
    try {
        launch_portfolio_greeks_kernel(params, quantities, results, num_positions, config);
    } catch (...) {
        // If CUDA fails, use CPU fallback
        cpu_launch_portfolio_greeks_kernel(params, quantities, results, num_positions, config);
    }
#endif
}

inline void compute_blackscholes_greeks(
    const BlackScholesParams* params,
    OptionResults* results,
    int num_scenarios,
    GPUConfig config = {1000, 50000}
) {
#ifdef CPU_ONLY
    cpu_launch_blackscholes_kernel(params, results, num_scenarios, config);
#else
    // Try CUDA first, fall back to CPU if needed
    try {
        launch_blackscholes_kernel(params, results, num_scenarios, config);
    } catch (...) {
        // If CUDA fails, use CPU fallback
        cpu_launch_blackscholes_kernel(params, results, num_scenarios, config);
    }
#endif
}

// Runtime check for CUDA availability
inline bool is_cuda_available() {
#ifdef CPU_ONLY
    return false;
#else
    // This would be implemented to check CUDA runtime
    // For now, return true if compiled with CUDA
    return true;
#endif
}

#endif // CUDA_WRAPPER_H
