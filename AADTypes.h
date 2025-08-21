#ifndef AAD_TYPES_H
#define AAD_TYPES_H

#ifdef CPU_ONLY
// CPU-only mode - don't include CUDA headers
#else
// Include CUDA headers only if available
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
// Fallback definitions if CUDA headers not available
typedef int cudaError_t;
#define cudaSuccess 0
#define __device__
#define __global__
#define __host__
#endif
#endif

#include <vector>
#include <memory>

// AAD Operation Types
enum class AADOpType {
    LEAF = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    LOG = 5,
    EXP = 6,
    SQRT = 7,
    NEG = 8,
    NORM_CDF = 9
};

// GPU Tape Entry - optimized for memory alignment
struct GPUTapeEntry {
    int result_idx;     // 4 bytes
    int input1_idx;     // 4 bytes
    int input2_idx;     // 4 bytes (unused for unary ops)
    AADOpType op_type;  // 4 bytes
    double partial1;    // 8 bytes
    double partial2;    // 8 bytes (unused for unary ops)
};

// Black-Scholes input parameters
struct BlackScholesParams {
    double spot;        // Current stock price
    double strike;      // Strike price
    double time;        // Time to expiration
    double rate;        // Risk-free rate
    double volatility;  // Volatility
    bool is_call;       // Call (true) or Put (false)
};

// Results structure
struct OptionResults {
    double price;
    double delta;   // dP/dS
    double vega;    // dP/dsigma
    double gamma;   // d²P/dS²
    double theta;   // dP/dT
    double rho;     // dP/dr
};

// GPU configuration
struct GPUConfig {
    int max_tape_size = 1000;
    int max_scenarios = 10000;
    int block_size = 256;
    bool use_fast_math = true;
};

#endif // AAD_TYPES_H
