#include "cpu_fallback.h"
#include <cstring>

// CPU-only implementation - all functions defined in cpu_fallback.h
// This file is created automatically when building in CPU-only mode

extern "C" {
    // Dummy implementations that redirect to CPU fallback
    void launch_blackscholes_kernel(
        const BlackScholesParams* h_params,
        OptionResults* h_results,
        int num_scenarios,
        GPUConfig config
    ) {
        cpu_launch_blackscholes_kernel(h_params, h_results, num_scenarios, config);
    }

    void launch_portfolio_greeks_kernel(
        const BlackScholesParams* h_params,
        const int* h_quantities,
        OptionResults* h_results,
        int num_positions,
        GPUConfig config
    ) {
        cpu_launch_portfolio_greeks_kernel(h_params, h_quantities, h_results, num_positions, config);
    }
}
