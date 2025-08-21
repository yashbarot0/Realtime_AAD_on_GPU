#include "RealTimePortfolioEngine.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef CPU_ONLY
// CPU-only mode - define CUDA stubs
typedef int cudaError_t;
#define cudaSuccess 0
struct cudaDeviceProp { char name[256]; };
inline int cudaGetDeviceCount(int* count) { *count = 0; return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp* prop, int device) { strcpy(prop->name, "CPU-Only Mode"); return 0; }
inline int cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(int error) { return "CPU-Only Mode"; }
inline int cudaDeviceSynchronize() { return 0; }
#else
// Include CUDA headers if available
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Fallback definitions if CUDA headers not available
typedef int cudaError_t;
#define cudaSuccess 0
struct cudaDeviceProp { char name[256]; };
extern "C" {
    int cudaGetDeviceCount(int* count);
    int cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
    int cudaGetLastError();
    const char* cudaGetErrorString(int error);
    int cudaDeviceSynchronize();
}
#endif
#endif

RealTimePortfolioEngine::~RealTimePortfolioEngine() {
    stop();
}

bool RealTimePortfolioEngine::initialize() {
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices available: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (config_.enable_logging) {
        std::cout << "Initialized Real-Time Portfolio Engine" << std::endl;
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Max positions: " << config_.max_positions << std::endl;
        std::cout << "Update frequency: " << config_.update_frequency_ms << "ms" << std::endl;
    }
    
    return true;
}

bool RealTimePortfolioEngine::start() {
    if (is_running_) {
        return false; // Already running
    }
    
    is_running_ = true;
    
    // Start worker threads
    compute_thread_ = std::thread(&RealTimePortfolioEngine::compute_worker, this);
    
    if (config_.enable_risk_monitoring) {
        monitoring_thread_ = std::thread(&RealTimePortfolioEngine::monitoring_worker, this);
    }
    
    if (config_.enable_logging) {
        std::cout << "Real-Time Portfolio Engine started" << std::endl;
    }
    
    return true;
}

void RealTimePortfolioEngine::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    data_cv_.notify_all();
    
    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    
    if (config_.enable_logging) {
        std::cout << "Real-Time Portfolio Engine stopped" << std::endl;
        log_performance_metrics();
    }
}

bool RealTimePortfolioEngine::add_position(const OptionPosition& position) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // Check for duplicate position ID
    auto it = std::find_if(current_positions_.begin(), current_positions_.end(),
                          [&](const OptionPosition& p) { return p.position_id == position.position_id; });
    
    if (it != current_positions_.end()) {
        return false; // Position ID already exists
    }
    
    current_positions_.push_back(position);
    
    if (config_.enable_logging) {
        std::cout << "Added position: " << position.symbol << " " 
                  << position.strike << " " << (position.is_call ? "CALL" : "PUT")
                  << " qty=" << position.quantity << std::endl;
    }
    
    return true;
}

bool RealTimePortfolioEngine::remove_position(int position_id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = std::find_if(current_positions_.begin(), current_positions_.end(),
                          [position_id](const OptionPosition& p) { return p.position_id == position_id; });
    
    if (it == current_positions_.end()) {
        return false; // Position not found
    }
    
    if (config_.enable_logging) {
        std::cout << "Removed position ID: " << position_id << std::endl;
    }
    
    current_positions_.erase(it);
    return true;
}

bool RealTimePortfolioEngine::update_position_quantity(int position_id, int new_quantity) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = std::find_if(current_positions_.begin(), current_positions_.end(),
                          [position_id](const OptionPosition& p) { return p.position_id == position_id; });
    
    if (it == current_positions_.end()) {
        return false; // Position not found
    }
    
    it->quantity = new_quantity;
    
    if (config_.enable_logging) {
        std::cout << "Updated position " << position_id << " quantity to " << new_quantity << std::endl;
    }
    
    return true;
}

std::vector<OptionPosition> RealTimePortfolioEngine::get_positions() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return current_positions_;
}

bool RealTimePortfolioEngine::update_market_data(const std::string& symbol, const OptionMarketData& data) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_market_data_[symbol] = data;
    return true;
}

bool RealTimePortfolioEngine::update_market_data_batch(
    const std::unordered_map<std::string, OptionMarketData>& data) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    for (const auto& pair : data) {
        current_market_data_[pair.first] = pair.second;
    }
    
    return true;
}

void RealTimePortfolioEngine::compute_worker() {
    auto last_compute_time = std::chrono::steady_clock::now();
    
    while (is_running_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_compute_time);
        
        if (elapsed.count() >= config_.update_frequency_ms) {
            // Perform computation
            auto result = compute_portfolio_greeks();
            
            // Update last result
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                last_result_ = result;
            }
            
            // Call update callback
            if (update_callback_ && result.success) {
                update_callback_(result);
            }
            
            // Update performance metrics
            computations_completed_++;
            double new_avg = (avg_computation_time_ * (computations_completed_ - 1) + result.computation_time_ms) 
                           / computations_completed_;
            avg_computation_time_ = new_avg;
            
            last_compute_time = now;
        }
        
        // Sleep for a short time to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void RealTimePortfolioEngine::monitoring_worker() {
    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.update_frequency_ms));
        
        PortfolioComputeResult current_result;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            current_result = last_result_;
        }
        
        if (current_result.success) {
            auto alerts = check_risk_limits(current_result.portfolio_greeks);
            
            for (const auto& alert : alerts) {
                if (alert_callback_) {
                    alert_callback_(alert);
                }
                
                if (config_.enable_logging) {
                    std::cout << "RISK ALERT: " << alert.metric_name 
                              << " = " << alert.current_value 
                              << " (limit: " << alert.limit_value << ")" << std::endl;
                }
            }
        }
    }
}

PortfolioComputeResult RealTimePortfolioEngine::compute_portfolio_greeks() {
    std::vector<OptionPosition> positions;
    std::unordered_map<std::string, OptionMarketData> market_data;
    
    // Copy current state
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        positions = current_positions_;
        market_data = current_market_data_;
    }
    
    return compute_portfolio_greeks_internal(positions, market_data);
}

PortfolioComputeResult RealTimePortfolioEngine::compute_portfolio_greeks_internal(
    const std::vector<OptionPosition>& positions,
    const std::unordered_map<std::string, OptionMarketData>& market_data) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PortfolioComputeResult result;
    result.success = false;
    
    if (positions.empty()) {
        result.error_message = "No positions in portfolio";
        result.computation_time_ms = 0.0;
        return result;
    }
    
    // Prepare data for GPU computation
    std::vector<BlackScholesParams> bs_params;
    std::vector<int> quantities;
    std::vector<OptionResults> option_results(positions.size());
    
    for (const auto& position : positions) {
        auto market_it = market_data.find(position.symbol);
        if (market_it == market_data.end()) {
            result.error_message = "Missing market data for symbol: " + position.symbol;
            return result;
        }
        
        BlackScholesParams params = position_to_bs_params(position, market_it->second);
        bs_params.push_back(params);
        quantities.push_back(position.quantity);
    }
    
    // Launch GPU computation
    try {
        launch_portfolio_greeks_kernel(
            bs_params.data(),
            quantities.data(),
            option_results.data(),
            static_cast<int>(positions.size()),
            gpu_config_
        );
        
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            result.error_message = "CUDA error: " + std::string(cudaGetErrorString(error));
            return result;
        }
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        // Aggregate portfolio-level Greeks
        result.portfolio_greeks = aggregate_greeks(positions, option_results);
        result.position_results = option_results;
        result.risk_alerts = check_risk_limits(result.portfolio_greeks);
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = "Computation error: " + std::string(e.what());
        return result;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.computation_time_ms = duration.count() / 1000.0;
    
    return result;
}

BlackScholesParams RealTimePortfolioEngine::position_to_bs_params(
    const OptionPosition& position,
    const OptionMarketData& market_data) const {
    
    BlackScholesParams params;
    params.spot = market_data.spot_price;
    params.strike = position.strike;
    params.time = calculate_time_to_expiration(position.expiration_time);
    params.rate = market_data.risk_free_rate;
    params.volatility = get_implied_volatility(position, market_data);
    params.is_call = position.is_call;
    
    return params;
}

PortfolioGreeks RealTimePortfolioEngine::aggregate_greeks(
    const std::vector<OptionPosition>& positions,
    const std::vector<OptionResults>& results) const {
    
    PortfolioGreeks greeks = {};
    greeks.num_positions = static_cast<int>(positions.size());
    greeks.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& position = positions[i];
        const auto& result = results[i];
        
        // Scale by position quantity
        int qty = position.quantity;
        
        greeks.total_delta += result.delta * qty;
        greeks.total_gamma += result.gamma * qty;
        greeks.total_vega += result.vega * qty;
        greeks.total_theta += result.theta * qty;
        greeks.total_rho += result.rho * qty;
        
        // Calculate P&L (current value - premium paid)
        double position_value = result.price * qty;
        double position_cost = position.premium_paid * std::abs(qty);
        greeks.total_pnl += (qty > 0 ? position_value - position_cost : position_cost - position_value);
        
        // Calculate exposure (absolute notional value)
        greeks.total_exposure += std::abs(result.price * qty);
    }
    
    return greeks;
}

std::vector<RiskAlert> RealTimePortfolioEngine::check_risk_limits(const PortfolioGreeks& greeks) const {
    std::vector<RiskAlert> alerts;
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Check delta limit
    if (std::abs(greeks.total_delta) > risk_limits_.max_delta) {
        alerts.push_back({"Delta", greeks.total_delta, risk_limits_.max_delta, true, timestamp});
    }
    
    // Check gamma limit
    if (std::abs(greeks.total_gamma) > risk_limits_.max_gamma) {
        alerts.push_back({"Gamma", greeks.total_gamma, risk_limits_.max_gamma, true, timestamp});
    }
    
    // Check vega limit
    if (std::abs(greeks.total_vega) > risk_limits_.max_vega) {
        alerts.push_back({"Vega", greeks.total_vega, risk_limits_.max_vega, true, timestamp});
    }
    
    // Check loss limit
    if (greeks.total_pnl < risk_limits_.max_loss) {
        alerts.push_back({"P&L", greeks.total_pnl, risk_limits_.max_loss, true, timestamp});
    }
    
    // Check exposure limit
    if (greeks.total_exposure > risk_limits_.max_exposure) {
        alerts.push_back({"Exposure", greeks.total_exposure, risk_limits_.max_exposure, true, timestamp});
    }
    
    return alerts;
}

double RealTimePortfolioEngine::calculate_time_to_expiration(double expiration_time) const {
    return std::max(0.001, expiration_time); // Minimum 1 day to avoid division by zero
}

double RealTimePortfolioEngine::get_implied_volatility(
    const OptionPosition& position,
    const OptionMarketData& market_data) const {
    
    double vol = market_data.implied_volatility;
    
    // Apply volatility bounds
    vol = std::max(config_.volatility_floor, std::min(config_.volatility_ceiling, vol));
    
    return vol;
}

void RealTimePortfolioEngine::log_performance_metrics() {
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Total computations: " << computations_completed_ << std::endl;
    std::cout << "Average computation time: " << avg_computation_time_ << " ms" << std::endl;
    std::cout << "Throughput: " << (1000.0 / avg_computation_time_) << " computations/second" << std::endl;
}
