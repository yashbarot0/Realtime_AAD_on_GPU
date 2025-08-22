#include "python_gpu_interface.h"
#include <chrono>
#include <iostream>
#include <thread>

extern "C" void launch_blackscholes_kernel(
    const BlackScholesParams* h_params,
    OptionResults* h_results,
    int num_scenarios,
    GPUConfig config
);

LivePortfolioManager::LivePortfolioManager() : running_(false) {
    std::cout << "LivePortfolioManager created" << std::endl;
    // Initialize current_greeks_
    current_greeks_ = {};
    current_greeks_.last_update = std::chrono::system_clock::now();
}

LivePortfolioManager::~LivePortfolioManager() {
    stop_processing();
}

void LivePortfolioManager::add_data_batch(const std::vector<LiveOptionData>& batch) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    data_queue_.push(batch);
    queue_cv_.notify_one();
}

void LivePortfolioManager::update_positions(const std::map<std::string, PortfolioPosition>& positions) {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    positions_ = positions;
}

PortfolioGreeks LivePortfolioManager::get_current_greeks() {
    std::lock_guard<std::mutex> lock(greeks_mutex_);
    return current_greeks_;
}

std::map<std::string, PortfolioPosition> LivePortfolioManager::get_positions() {
    std::lock_guard<std::mutex> lock(positions_mutex_);
    return positions_;
}

void LivePortfolioManager::start_processing() {
    if (!running_) {
        running_ = true;
        processing_thread_ = std::thread(&LivePortfolioManager::processing_loop, this);
        std::cout << "Portfolio processing started" << std::endl;
    }
}

void LivePortfolioManager::stop_processing() {
    if (running_) {
        running_ = false;
        queue_cv_.notify_all();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
    std::cout << "Portfolio processing stopped" << std::endl;
}

void LivePortfolioManager::processing_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Wait for data or stop signal
        queue_cv_.wait(lock, [this] { return !data_queue_.empty() || !running_; });
        
        if (!running_) break;
        
        if (!data_queue_.empty()) {
            auto batch = data_queue_.front();
            data_queue_.pop();
            lock.unlock();
            
            // Process the batch
            process_data_batch(batch);
        }
    }
}

void LivePortfolioManager::process_data_batch(const std::vector<LiveOptionData>& batch) {
    if (batch.empty()) return;
    
    // Convert to GPU format
    std::vector<BlackScholesParams> gpu_params;
    gpu_params.reserve(batch.size());
    
    for (const auto& option : batch) {
        BlackScholesParams params;
        params.spot = option.spot_price;
        params.strike = option.strike;
        params.time = option.time_to_expiry;
        params.rate = option.risk_free_rate;
        params.volatility = option.implied_volatility;
        params.is_call = option.is_call;
        gpu_params.push_back(params);
    }
    
    // Allocate results
    std::vector<OptionResults> results(gpu_params.size());
    
    // Launch GPU computation
    GPUConfig config;
    config.block_size = 256;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    launch_blackscholes_kernel(gpu_params.data(), results.data(),
                              gpu_params.size(), config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Processed " << batch.size() << " options in "
              << duration.count() << " µs" << std::endl;
    
    // Update portfolio Greeks
    update_portfolio_greeks(batch, results);
}

void LivePortfolioManager::update_portfolio_greeks(const std::vector<LiveOptionData>& data,
                                                   const std::vector<OptionResults>& results) {
    std::lock_guard<std::mutex> positions_lock(positions_mutex_);
    std::lock_guard<std::mutex> greeks_lock(greeks_mutex_);
    
    // Reset portfolio Greeks
    PortfolioGreeks new_greeks = {};
    new_greeks.last_update = std::chrono::system_clock::now();
    
    // Aggregate Greeks based on positions
    for (size_t i = 0; i < data.size() && i < results.size(); ++i) {
        const auto& option = data[i];
        const auto& result = results[i];
        
        // Convert char array to string for lookup
        std::string symbol_str(option.symbol);
        
        // Find if we have a position in this option
        auto pos_it = positions_.find(symbol_str);
        if (pos_it != positions_.end()) {
            const auto& position = pos_it->second;
            
            // Calculate position Greeks (simplified)
            double position_multiplier = position.quantity / 100.0; // Assuming 1 option = 100 shares
            
            new_greeks.total_delta += result.delta * position_multiplier;
            new_greeks.total_vega += result.vega * position_multiplier;
            new_greeks.total_gamma += result.gamma * position_multiplier;
            new_greeks.total_theta += result.theta * position_multiplier;
            new_greeks.total_rho += result.rho * position_multiplier;
            
            // Calculate P&L
            double pnl = (option.spot_price - position.entry_price) * position.quantity;
            new_greeks.total_pnl += pnl;
        }
    }
    
    current_greeks_ = new_greeks;
}

// C interface implementation
extern "C" {

LivePortfolioManager* create_portfolio_manager() {
    return new LivePortfolioManager();
}

void destroy_portfolio_manager(LivePortfolioManager* manager) {
    delete manager;
}

void add_options_batch(LivePortfolioManager* manager,
                       const LiveOptionData* batch,
                       size_t count) {
    if (!manager || !batch || count == 0) return;
    
    std::vector<LiveOptionData> vec_batch(batch, batch + count);
    manager->add_data_batch(vec_batch);
}

void add_option_data(LivePortfolioManager* manager,
                     const char* symbol,
                     double strike,
                     double spot_price,
                     double time_to_expiry,
                     double risk_free_rate,
                     double implied_volatility,
                     int is_call,
                     double market_price) {
    if (!manager) return;
    
    std::vector<LiveOptionData> batch;
    LiveOptionData option;
    
    // Use strncpy for fixed-length string field
    strncpy(option.symbol, symbol, sizeof(option.symbol) - 1);
    option.symbol[sizeof(option.symbol) - 1] = '\0'; // Ensure null termination
    
    option.strike = strike;
    option.spot_price = spot_price;
    option.time_to_expiry = time_to_expiry;
    option.risk_free_rate = risk_free_rate;
    option.implied_volatility = implied_volatility;
    option.is_call = (is_call != 0);
    option.market_price = market_price;
    option.timestamp = std::chrono::system_clock::now();
    
    batch.push_back(option);
    manager->add_data_batch(batch);
}

void get_portfolio_greeks(LivePortfolioManager* manager,
                         double* total_delta,
                         double* total_vega,
                         double* total_gamma,
                         double* total_theta,
                         double* total_rho,
                         double* total_pnl) {
    if (!manager) return;
    
    auto greeks = manager->get_current_greeks();
    *total_delta = greeks.total_delta;
    *total_vega = greeks.total_vega;
    *total_gamma = greeks.total_gamma;
    *total_theta = greeks.total_theta;
    *total_rho = greeks.total_rho;
    *total_pnl = greeks.total_pnl;
}

void start_processing(LivePortfolioManager* manager) {
    if (manager) {
        manager->start_processing();
    }
}

void stop_processing(LivePortfolioManager* manager) {
    if (manager) {
        manager->stop_processing();
    }
}

}
