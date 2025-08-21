#include "cpu_fallback.h"
#include <cstring>

// CPU-only implementation - all functions defined in cpu_fallback.h
// This file is created automatically when building in CPU-only mode

extern "C" {
    // CUDA runtime function stubs for CPU-only builds
    int cudaGetDeviceCount(int* count) {
        *count = 0;
        return 0;
    }
    
    int cudaGetDeviceProperties(void* prop, int device) {
        // Fill with dummy data
        memset(prop, 0, 256); // Assume cudaDeviceProp is max 256 bytes
        return 0;
    }
    
    int cudaGetLastError() {
        return 0;
    }
    
    const char* cudaGetErrorString(int error) {
        return "CPU-Only Mode";
    }
    
    int cudaDeviceSynchronize() {
        return 0;
    }

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
