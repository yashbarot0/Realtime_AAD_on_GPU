#include "cpu_fallback.h"
#include <cstring>

// Define CPU_ONLY for this compilation unit to access the functions
#ifndef CPU_ONLY
#define CPU_ONLY
#endif

// CPU-only implementation - all functions defined in cpu_fallback.h
// This file is created automatically when building in CPU-only mode

extern "C" {
    // CUDA runtime function stubs for CPU-only builds
    int cudaGetDeviceCount(int* count) {
        *count = 0;
        return 0;
    }
    
    int cudaGetDeviceProperties(void* prop_void, int device) {
        // Cast to our known struct type and fill with dummy data
        struct cudaDeviceProp {
            char name[256]; 
            int major;
            int minor;
            size_t totalGlobalMem;
            size_t sharedMemPerBlock;
            int maxThreadsPerBlock;
            int multiProcessorCount;
        };
        
        cudaDeviceProp* prop = (cudaDeviceProp*)prop_void;
        strcpy(prop->name, "CPU-Only Mode");
        prop->major = 0;
        prop->minor = 0;
        prop->totalGlobalMem = 0;
        prop->sharedMemPerBlock = 0;
        prop->maxThreadsPerBlock = 1;
        prop->multiProcessorCount = 1;
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
