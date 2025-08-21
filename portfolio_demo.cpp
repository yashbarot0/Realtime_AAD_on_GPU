#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <random>
#include "RealTimePortfolioEngine.h"

// Simulate market data updates
class MarketDataSimulator {
private:
    std::mt19937 rng;
    std::normal_distribution<double> price_dist;
    std::normal_distribution<double> vol_dist;
    
public:
    MarketDataSimulator() : rng(std::random_device{}()), 
                           price_dist(0.0, 0.01),  // 1% price moves
                           vol_dist(0.0, 0.02) {}  // 2% vol moves
    
    OptionMarketData generate_market_data(const std::string& symbol, 
                                         double base_price = 100.0,
                                         double base_vol = 0.25) {
        OptionMarketData data;
        data.symbol = symbol;
        data.spot_price = base_price * (1.0 + price_dist(rng));
        data.risk_free_rate = 0.05;  // 5% risk-free rate
        data.implied_volatility = std::max(0.1, base_vol * (1.0 + vol_dist(rng)));
        data.bid = data.spot_price * 0.999;
        data.ask = data.spot_price * 1.001;
        data.last_price = data.spot_price;
        data.volume = 1000 + (rng() % 10000);
        data.open_interest = 5000 + (rng() % 50000);
        data.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        return data;
    }
};

void print_portfolio_greeks(const PortfolioComputeResult& result) {
    if (!result.success) {
        std::cout << "❌ Computation failed: " << result.error_message << std::endl;
        return;
    }
    
    const auto& greeks = result.portfolio_greeks;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n┌─ Portfolio Greeks ─────────────────────┐" << std::endl;
    std::cout << "│ Delta:    " << std::setw(12) << greeks.total_delta << "          │" << std::endl;
    std::cout << "│ Gamma:    " << std::setw(12) << greeks.total_gamma << "          │" << std::endl;
    std::cout << "│ Vega:     " << std::setw(12) << greeks.total_vega << "          │" << std::endl;
    std::cout << "│ Theta:    " << std::setw(12) << greeks.total_theta << "          │" << std::endl;
    std::cout << "│ Rho:      " << std::setw(12) << greeks.total_rho << "          │" << std::endl;
    std::cout << "│ P&L:      $" << std::setw(11) << greeks.total_pnl << "          │" << std::endl;
    std::cout << "│ Exposure: $" << std::setw(11) << greeks.total_exposure << "          │" << std::endl;
    std::cout << "│ Positions:" << std::setw(12) << greeks.num_positions << "          │" << std::endl;
    std::cout << "│ Compute:  " << std::setw(9) << result.computation_time_ms << " ms     │" << std::endl;
    std::cout << "└────────────────────────────────────────┘" << std::endl;
    
    // Show any risk alerts
    if (!result.risk_alerts.empty()) {
        std::cout << "\n🚨 Risk Alerts:" << std::endl;
        for (const auto& alert : result.risk_alerts) {
            std::cout << "   " << alert.metric_name << ": " << alert.current_value 
                      << " (limit: " << alert.limit_value << ")" << std::endl;
        }
    }
}

void print_position_details(const std::vector<OptionPosition>& positions,
                           const std::vector<OptionResults>& results) {
    if (positions.size() != results.size()) return;
    
    std::cout << "\n┌─ Position Details ─────────────────────────────────────────┐" << std::endl;
    std::cout << "│ ID │ Symbol │ Strike │  Type │ Qty │  Price │  Delta │  Vega │" << std::endl;
    std::cout << "├────┼────────┼────────┼───────┼─────┼────────┼────────┼───────┤" << std::endl;
    
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i];
        const auto& res = results[i];
        
        std::cout << "│" << std::setw(3) << pos.position_id << " │"
                  << std::setw(7) << pos.symbol << "│"
                  << std::setw(7) << std::fixed << std::setprecision(0) << pos.strike << " │"
                  << std::setw(6) << (pos.is_call ? "CALL" : "PUT") << " │"
                  << std::setw(4) << pos.quantity << " │"
                  << std::setw(7) << std::fixed << std::setprecision(2) << res.price << "│"
                  << std::setw(7) << std::fixed << std::setprecision(2) << res.delta << " │"
                  << std::setw(6) << std::fixed << std::setprecision(1) << res.vega << " │" << std::endl;
    }
    
    std::cout << "└────────────────────────────────────────────────────────────┘" << std::endl;
}

int main() {
    std::cout << "Real-Time Portfolio Greeks Demo" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Initialize the engine
    RealTimeConfig config;
    config.max_positions = 100;
    config.update_frequency_ms = 2000;  // Update every 2 seconds
    config.enable_risk_monitoring = true;
    config.enable_logging = true;
    
    RealTimePortfolioEngine engine(config);
    
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize portfolio engine" << std::endl;
        return 1;
    }
    
    // Set risk limits
    RiskLimits limits;
    limits.max_delta = 500.0;
    limits.max_gamma = 50.0;
    limits.max_vega = 2000.0;
    limits.max_loss = -25000.0;
    limits.max_exposure = 250000.0;
    
    engine.set_risk_limits(limits);
    
    // Add sample portfolio positions
    std::cout << "\nAdding sample positions..." << std::endl;
    
    // Apple positions
    OptionPosition pos1;
    pos1.position_id = 1;
    strcpy(pos1.symbol, "AAPL");
    pos1.strike = 190.0;
    pos1.expiration_time = 0.25;  // 3 months
    pos1.quantity = 10;
    pos1.is_call = true;
    pos1.premium_paid = 5.50;
    pos1.entry_spot = 185.0;
    engine.add_position(pos1);
    
    // Apple protective put
    OptionPosition pos2;
    pos2.position_id = 2;
    strcpy(pos2.symbol, "AAPL");
    pos2.strike = 180.0;
    pos2.expiration_time = 0.25;
    pos2.quantity = 10;
    pos2.is_call = false;
    pos2.premium_paid = 3.20;
    pos2.entry_spot = 185.0;
    engine.add_position(pos2);
    
    // Microsoft position
    OptionPosition pos3;
    pos3.position_id = 3;
    strcpy(pos3.symbol, "MSFT");
    pos3.strike = 380.0;
    pos3.expiration_time = 0.17;  // 2 months
    pos3.quantity = 5;
    pos3.is_call = true;
    pos3.premium_paid = 12.80;
    pos3.entry_spot = 375.0;
    engine.add_position(pos3);
    
    // Google position
    OptionPosition pos4;
    pos4.position_id = 4;
    strcpy(pos4.symbol, "GOOGL");
    pos4.strike = 140.0;
    pos4.expiration_time = 0.33;  // 4 months
    pos4.quantity = 8;
    pos4.is_call = true;
    pos4.premium_paid = 8.90;
    pos4.entry_spot = 135.0;
    engine.add_position(pos4);
    
    // Short Tesla position (bearish view)
    OptionPosition pos5;
    pos5.position_id = 5;
    strcpy(pos5.symbol, "TSLA");
    pos5.strike = 250.0;
    pos5.expiration_time = 0.08;  // 1 month
    pos5.quantity = -3;  // Short position
    pos5.is_call = true;
    pos5.premium_paid = 15.20;
    pos5.entry_spot = 240.0;
    engine.add_position(pos5);
    
    std::cout << "Added " << engine.get_positions().size() << " positions" << std::endl;
    
    // Set up callbacks
    engine.set_update_callback([](const PortfolioComputeResult& result) {
        static int update_count = 0;
        update_count++;
        
        std::cout << "\n🔄 Update #" << update_count << " - " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count() 
                  << std::endl;
        
        print_portfolio_greeks(result);
        
        // Show position details every 5th update
        if (update_count % 5 == 0) {
            auto positions = result.position_results;
            // Note: In a real implementation, we'd need to match positions with results
            std::cout << "\nDetailed position breakdown:" << std::endl;
            for (size_t i = 0; i < positions.size() && i < 5; ++i) {
                std::cout << "Position " << (i+1) << ": Price=$" << positions[i].price
                          << ", Delta=" << positions[i].delta 
                          << ", Gamma=" << positions[i].gamma << std::endl;
            }
        }
    });
    
    engine.set_alert_callback([](const RiskAlert& alert) {
        std::cout << "\n🚨 RISK ALERT: " << alert.metric_name 
                  << " = " << alert.current_value 
                  << " exceeds limit of " << alert.limit_value << std::endl;
    });
    
    // Market data simulator
    MarketDataSimulator market_sim;
    
    // Base prices for our symbols
    std::unordered_map<std::string, double> base_prices = {
        {"AAPL", 185.0},
        {"MSFT", 375.0},
        {"GOOGL", 135.0},
        {"TSLA", 240.0}
    };
    
    // Start the engine
    if (!engine.start()) {
        std::cerr << "Failed to start portfolio engine" << std::endl;
        return 1;
    }
    
    std::cout << "\nStarting real-time simulation..." << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    
    // Simulation loop
    int iteration = 0;
    while (iteration < 30) {  // Run for 30 iterations (about 1 minute)
        iteration++;
        
        // Generate market data updates
        std::unordered_map<std::string, OptionMarketData> market_update;
        
        for (const auto& [symbol, base_price] : base_prices) {
            market_update[symbol] = market_sim.generate_market_data(symbol, base_price);
        }
        
        // Update market data in engine
        engine.update_market_data_batch(market_update);
        
        // Simulate some position changes occasionally
        if (iteration == 10) {
            std::cout << "\n📈 Adding new position..." << std::endl;
            OptionPosition new_pos;
            new_pos.position_id = 6;
            strcpy(new_pos.symbol, "AAPL");
            new_pos.strike = 195.0;
            new_pos.expiration_time = 0.15;
            new_pos.quantity = 5;
            new_pos.is_call = true;
            new_pos.premium_paid = 7.30;
            new_pos.entry_spot = 185.0;
            engine.add_position(new_pos);
        }
        
        if (iteration == 20) {
            std::cout << "\n📉 Closing Tesla position..." << std::endl;
            engine.remove_position(5);
        }
        
        // Wait before next update
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    
    // Stop the engine
    engine.stop();
    
    // Final statistics
    std::cout << "\n=== Final Performance Statistics ===" << std::endl;
    std::cout << "Total computations: " << engine.get_computations_completed() << std::endl;
    std::cout << "Average computation time: " << engine.get_average_computation_time() << " ms" << std::endl;
    
    auto final_result = engine.get_last_result();
    if (final_result.success) {
        std::cout << "\nFinal Portfolio Summary:" << std::endl;
        print_portfolio_greeks(final_result);
    }
    
    std::cout << "\nDemo completed successfully!" << std::endl;
    return 0;
}
