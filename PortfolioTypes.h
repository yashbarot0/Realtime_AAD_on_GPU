#ifndef PORTFOLIO_TYPES_H
#define PORTFOLIO_TYPES_H

#include "AADTypes.h"
#include <vector>
#include <string>
#include <unordered_map>

// Position in the portfolio
struct OptionPosition {
    int position_id;
    std::string symbol;
    double strike;
    double expiration_time;  // Time to expiration in years
    int quantity;           // Positive for long, negative for short
    bool is_call;
    double premium_paid;    // For P&L calculation
    double entry_spot;      // Spot price when position was entered
};

// Market data for an option
struct OptionMarketData {
    std::string symbol;
    double spot_price;
    double risk_free_rate;
    double implied_volatility;
    double bid;
    double ask;
    double last_price;
    int volume;
    double open_interest;
    long long timestamp;    // Unix timestamp
};

// Portfolio-level Greeks
struct PortfolioGreeks {
    double total_delta;
    double total_gamma;
    double total_vega;
    double total_theta;
    double total_rho;
    double total_pnl;       // Mark-to-market P&L
    double total_exposure;  // Total notional exposure
    int num_positions;
    long long timestamp;
};

// Risk limits and alerts
struct RiskLimits {
    double max_delta = 1000.0;
    double max_gamma = 100.0;
    double max_vega = 5000.0;
    double max_loss = -50000.0;
    double max_exposure = 1000000.0;
};

// Alert when limits are breached
struct RiskAlert {
    std::string metric_name;
    double current_value;
    double limit_value;
    bool is_breach;
    long long timestamp;
};

// Batch computation request
struct PortfolioComputeRequest {
    std::vector<OptionPosition> positions;
    std::unordered_map<std::string, OptionMarketData> market_data;
    RiskLimits limits;
    bool compute_second_order = true;  // Compute gamma
    bool compute_cross_gamma = false;  // Compute cross-asset gamma
};

// Enhanced results for portfolio
struct PortfolioComputeResult {
    PortfolioGreeks portfolio_greeks;
    std::vector<OptionResults> position_results;  // Per-position results
    std::vector<RiskAlert> risk_alerts;
    double computation_time_ms;
    bool success;
    std::string error_message;
};

// Configuration for real-time engine
struct RealTimeConfig {
    int max_positions = 1000;
    int update_frequency_ms = 1000;    // Update every second
    bool enable_risk_monitoring = true;
    bool enable_logging = true;
    bool use_implied_volatility = true; // Use market IV vs historical
    double volatility_floor = 0.01;    // Minimum volatility (1%)
    double volatility_ceiling = 5.0;   // Maximum volatility (500%)
};

#endif // PORTFOLIO_TYPES_H
