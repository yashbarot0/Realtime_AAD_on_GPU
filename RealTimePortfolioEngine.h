#ifndef REALTIME_PORTFOLIO_ENGINE_H
#define REALTIME_PORTFOLIO_ENGINE_H

#include "PortfolioTypes.h"
#include "AADTypes.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// External CUDA kernel for portfolio computation
extern "C" void launch_portfolio_greeks_kernel(
    const BlackScholesParams* h_params,
    const int* h_quantities,
    OptionResults* h_results,
    int num_positions,
    GPUConfig config
);

class RealTimePortfolioEngine {
public:
    using UpdateCallback = std::function<void(const PortfolioComputeResult&)>;
    using AlertCallback = std::function<void(const RiskAlert&)>;

private:
    // Configuration
    RealTimeConfig config_;
    RiskLimits risk_limits_;
    
    // Thread management
    std::atomic<bool> is_running_{false};
    std::thread compute_thread_;
    std::thread monitoring_thread_;
    
    // Data synchronization
    std::mutex data_mutex_;
    std::condition_variable data_cv_;
    std::queue<PortfolioComputeRequest> request_queue_;
    
    // Current state
    PortfolioComputeResult last_result_;
    std::unordered_map<std::string, OptionMarketData> current_market_data_;
    std::vector<OptionPosition> current_positions_;
    
    // Callbacks
    UpdateCallback update_callback_;
    AlertCallback alert_callback_;
    
    // GPU resources
    GPUConfig gpu_config_;
    
    // Performance tracking
    std::atomic<int> computations_completed_{0};
    std::atomic<double> avg_computation_time_{0.0};

public:
    RealTimePortfolioEngine(const RealTimeConfig& config = RealTimeConfig{});
    ~RealTimePortfolioEngine();
    
    // Engine lifecycle
    bool initialize();
    bool start();
    void stop();
    
    // Portfolio management
    bool add_position(const OptionPosition& position);
    bool remove_position(int position_id);
    bool update_position_quantity(int position_id, int new_quantity);
    std::vector<OptionPosition> get_positions() const;
    
    // Market data updates
    bool update_market_data(const std::string& symbol, const OptionMarketData& data);
    bool update_market_data_batch(const std::unordered_map<std::string, OptionMarketData>& data);
    
    // Risk management
    void set_risk_limits(const RiskLimits& limits);
    RiskLimits get_risk_limits() const;
    std::vector<RiskAlert> check_risk_limits(const PortfolioGreeks& greeks) const;
    
    // Callbacks
    void set_update_callback(UpdateCallback callback);
    void set_alert_callback(AlertCallback callback);
    
    // Status and monitoring
    bool is_running() const { return is_running_; }
    PortfolioComputeResult get_last_result() const;
    int get_computations_completed() const { return computations_completed_; }
    double get_average_computation_time() const { return avg_computation_time_; }
    
    // Manual computation trigger
    PortfolioComputeResult compute_portfolio_greeks();

private:
    // Worker threads
    void compute_worker();
    void monitoring_worker();
    
    // Computation methods
    PortfolioComputeResult compute_portfolio_greeks_internal(
        const std::vector<OptionPosition>& positions,
        const std::unordered_map<std::string, OptionMarketData>& market_data
    );
    
    BlackScholesParams position_to_bs_params(
        const OptionPosition& position,
        const OptionMarketData& market_data
    ) const;
    
    PortfolioGreeks aggregate_greeks(
        const std::vector<OptionPosition>& positions,
        const std::vector<OptionResults>& results
    ) const;
    
    // Utility methods
    double calculate_time_to_expiration(double expiration_time) const;
    double get_implied_volatility(const OptionPosition& position, 
                                 const OptionMarketData& market_data) const;
    void log_performance_metrics();
};

// Inline implementations for simple methods
inline RealTimePortfolioEngine::RealTimePortfolioEngine(const RealTimeConfig& config)
    : config_(config) {
    
    // Initialize GPU configuration
    gpu_config_.max_scenarios = config_.max_positions;
    gpu_config_.block_size = 256;
    gpu_config_.use_fast_math = true;
    
    // Default risk limits
    risk_limits_ = RiskLimits{};
}

inline void RealTimePortfolioEngine::set_risk_limits(const RiskLimits& limits) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    risk_limits_ = limits;
}

inline RiskLimits RealTimePortfolioEngine::get_risk_limits() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return risk_limits_;
}

inline PortfolioComputeResult RealTimePortfolioEngine::get_last_result() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return last_result_;
}

inline void RealTimePortfolioEngine::set_update_callback(UpdateCallback callback) {
    update_callback_ = callback;
}

inline void RealTimePortfolioEngine::set_alert_callback(AlertCallback callback) {
    alert_callback_ = callback;
}

#endif // REALTIME_PORTFOLIO_ENGINE_H
