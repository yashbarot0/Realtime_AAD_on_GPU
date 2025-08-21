#ifndef CPU_FALLBACK_H
#define CPU_FALLBACK_H

#include "AADTypes.h"
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef CPU_ONLY

// CPU implementations of device functions
inline double safe_log(double x) {
    return (x > 1e-15) ? std::log(x) : std::log(1e-15);
}

inline double safe_sqrt(double x) {
    return std::sqrt(std::max(x, 0.0));
}

inline double safe_divide(double numerator, double denominator) {
    return (std::abs(denominator) > 1e-15) ? numerator / denominator : 0.0;
}

inline double cpu_erf(double x) {
    // Approximation of error function for norm_cdf
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    int sign = (x >= 0) ? 1 : -1;
    x = std::abs(x);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);
    
    return sign * y;
}

inline double cpu_norm_cdf(double x) {
    return 0.5 * (1.0 + cpu_erf(x / std::sqrt(2.0)));
}

// CPU implementation of Black-Scholes computation
inline void cpu_blackscholes_single(const BlackScholesParams& params, OptionResults& result) {
    double S = params.spot;
    double K = params.strike;
    double T = params.time;
    double r = params.rate;
    double sigma = params.volatility;
    bool is_call = params.is_call;
    
    // Black-Scholes formula
    double sqrt_T = safe_sqrt(T);
    double d1 = (safe_log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    
    double N_d1 = cpu_norm_cdf(d1);
    double N_d2 = cpu_norm_cdf(d2);
    double N_minus_d1 = cpu_norm_cdf(-d1);
    double N_minus_d2 = cpu_norm_cdf(-d2);
    
    double discount = std::exp(-r * T);
    
    if (is_call) {
        result.price = S * N_d1 - K * discount * N_d2;
        result.delta = N_d1;
    } else {
        result.price = K * discount * N_minus_d2 - S * N_minus_d1;
        result.delta = -N_minus_d1;
    }
    
    // Greeks (same for calls and puts)
    double phi_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);
    
    result.gamma = phi_d1 / (S * sigma * sqrt_T);
    result.vega = S * phi_d1 * sqrt_T / 100.0;  // Per 1% vol change
    
    if (is_call) {
        result.theta = (-S * phi_d1 * sigma / (2.0 * sqrt_T) 
                       - r * K * discount * N_d2) / 365.0;  // Per day
        result.rho = K * T * discount * N_d2 / 100.0;  // Per 1% rate change
    } else {
        result.theta = (-S * phi_d1 * sigma / (2.0 * sqrt_T) 
                       + r * K * discount * N_minus_d2) / 365.0;
        result.rho = -K * T * discount * N_minus_d2 / 100.0;
    }
}

// CPU implementation of parallel Black-Scholes
inline void cpu_blackscholes_parallel(const BlackScholesParams* params, 
                                     OptionResults* results, 
                                     int num_scenarios) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < num_scenarios; i++) {
        cpu_blackscholes_single(params[i], results[i]);
    }
}

// CPU implementation of portfolio Greeks
inline void cpu_portfolio_greeks(const BlackScholesParams* params,
                                const int* quantities,
                                OptionResults* results,
                                int num_positions) {
    // First compute individual option results
    cpu_blackscholes_parallel(params, results, num_positions);
    
    // Scale by quantities
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < num_positions; i++) {
        int qty = quantities[i];
        results[i].price *= qty;
        results[i].delta *= qty;
        results[i].vega *= qty;
        results[i].gamma *= qty;
        results[i].theta *= qty;
        results[i].rho *= qty;
    }
}

#endif // CPU_ONLY

#endif // CPU_FALLBACK_H
