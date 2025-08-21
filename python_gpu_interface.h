#ifndef PYTHON_GPU_INTERFACE_H
#define PYTHON_GPU_INTERFACE_H

#include "AADTypes.h"
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

struct LiveOptionData {
    std::string symbol;
    double strike;
    double spot_price;
    double time_to_expiry;
    double risk_free_rate;
    double implied_volatility;
    bool is_call;
    double market_price;
    std::chrono::system_clock::time_point timestamp;
};

struct PortfolioPosition {
    std::string symbol;
    int quantity;
    double entry_price;
    double current_pnl;
    double delta_exposure;
    double vega_exposure;
    double gamma_exposure;
};

struct PortfolioGreeks {
    double total_delta;
    double total_vega;
    double total_gamma;
    double total_theta;
    double total_rho;
    double total_pnl;
    std::chrono::system_clock::time_point last_update;
};

class LivePortfolioManager {
private:
    std::queue<std::vector<LiveOptionData>> data_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::map<std::string, PortfolioPosition> positions_;
    std::mutex positions_mutex_;
    
    PortfolioGreeks current_greeks_;
    std::mutex greeks_mutex_;
    
    bool running_;
    std::thread processing_thread_;
    
    // GPU AAD processing
    void process_data_batch(const std::vector<LiveOptionData>& batch);
    void update_portfolio_greeks(const std::vector<LiveOptionData>& data, 
                               const std::vector<OptionResults>& results);

public:
    LivePortfolioManager();
    ~LivePortfolioManager();
    
    // Interface for Python
    void add_data_batch(const std::vector<LiveOptionData>& batch);
    void update_positions(const std::map<std::string, PortfolioPosition>& positions);
    
    // Get current state
    PortfolioGreeks get_current_greeks();
    std::map<std::string, PortfolioPosition> get_positions();
    
    // Control
    void start_processing();
    void stop_processing();
    
private:
    void processing_loop();
};

// C interface for Python integration
extern "C" {
    LivePortfolioManager* create_portfolio_manager();
    void destroy_portfolio_manager(LivePortfolioManager* manager);
    
    void add_option_data(LivePortfolioManager* manager, 
                        const char* symbol,
                        double strike,
                        double spot_price,
                        double time_to_expiry,
                        double risk_free_rate,
                        double implied_volatility,
                        int is_call,
                        double market_price);
    
    void get_portfolio_greeks(LivePortfolioManager* manager,
                            double* total_delta,
                            double* total_vega,
                            double* total_gamma,
                            double* total_theta,
                            double* total_rho,
                            double* total_pnl);
    
    void start_processing(LivePortfolioManager* manager);
    void stop_processing(LivePortfolioManager* manager);
}

#endif // PYTHON_GPU_INTERFACE_H
