#ifndef PYTHON_GPU_INTERFACE_H
#define PYTHON_GPU_INTERFACE_H

#include "../include/AADTypes.h"
#include <vector>
#include <map>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <cstring>

// Fixed-size struct for C interface compatibility
struct LiveOptionData {
    char symbol[16];
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
    
    void process_data_batch(const std::vector<LiveOptionData>& batch);
    void update_portfolio_greeks(const std::vector<LiveOptionData>& data,
                               const std::vector<OptionResults>& results);
    void processing_loop();

public:
    LivePortfolioManager();
    ~LivePortfolioManager();
    
    void add_data_batch(const std::vector<LiveOptionData>& batch);
    void update_positions(const std::map<std::string, PortfolioPosition>& positions);
    
    PortfolioGreeks get_current_greeks();
    std::map<std::string, PortfolioPosition> get_positions();
    
    void start_processing();
    void stop_processing();
};

// C interface for Python integration
extern "C" {
    LivePortfolioManager* create_portfolio_manager();
    void destroy_portfolio_manager(LivePortfolioManager* manager);
    void add_options_batch(LivePortfolioManager* manager, const LiveOptionData* batch, size_t count);
    void add_option_data(LivePortfolioManager* manager, const char* symbol, double strike,
                        double spot_price, double time_to_expiry, double risk_free_rate,
                        double implied_volatility, int is_call, double market_price);
    void get_portfolio_greeks(LivePortfolioManager* manager, double* total_delta,
                            double* total_vega, double* total_gamma, double* total_theta,
                            double* total_rho, double* total_pnl);
    void start_processing(LivePortfolioManager* manager);
    void stop_processing(LivePortfolioManager* manager);
}

#endif // PYTHON_GPU_INTERFACE_H
