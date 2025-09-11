#include "../include/GPUAADNumber.h"
#include <stdexcept>
#include <cmath>

GPUAADNumber::GPUAADNumber(double value, GPUAADTape* tape) 
    : tape_(tape) {
    if (tape_) {
        tape_index_ = tape_->add_variable(value);
    } else {
        tape_index_ = -1;
    }
}

GPUAADNumber::GPUAADNumber(int tape_index, GPUAADTape* tape)
    : tape_index_(tape_index), tape_(tape) {}

GPUAADNumber::GPUAADNumber(const GPUAADNumber& other)
    : tape_index_(other.tape_index_), tape_(other.tape_) {}

GPUAADNumber& GPUAADNumber::operator=(const GPUAADNumber& other) {
    tape_index_ = other.tape_index_;
    tape_ = other.tape_;
    return *this;
}

GPUAADNumber GPUAADNumber::operator+(const GPUAADNumber& other) const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double result_value = value() + other.value();
    int result_idx = tape_->record_operation(AADOpType::ADD, tape_index_, 
                                           other.tape_index_, 1.0, 1.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::operator-(const GPUAADNumber& other) const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double result_value = value() - other.value();
    int result_idx = tape_->record_operation(AADOpType::SUB, tape_index_, 
                                           other.tape_index_, 1.0, -1.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::operator*(const GPUAADNumber& other) const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double result_value = value() * other.value();
    int result_idx = tape_->record_operation(AADOpType::MUL, tape_index_, 
                                           other.tape_index_, other.value(), value());
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::operator/(const GPUAADNumber& other) const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double other_val = other.value();
    double result_value = value() / other_val;
    double partial1 = 1.0 / other_val;
    double partial2 = -value() / (other_val * other_val);
    
    int result_idx = tape_->record_operation(AADOpType::DIV, tape_index_, 
                                           other.tape_index_, partial1, partial2);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::operator-() const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double result_value = -value();
    int result_idx = tape_->record_operation(AADOpType::NEG, tape_index_, 
                                           -1, -1.0, 0.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::log() const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double val = value();
    double result_value = std::log(std::max(val, 1e-15));
    double partial = 1.0 / std::max(val, 1e-15);
    
    int result_idx = tape_->record_operation(AADOpType::LOG, tape_index_, 
                                           -1, partial, 0.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::exp() const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double val = value();
    double result_value = std::exp(std::min(val, 700.0));
    
    int result_idx = tape_->record_operation(AADOpType::EXP, tape_index_, 
                                           -1, result_value, 0.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::sqrt() const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    double val = value();
    double result_value = std::sqrt(std::max(val, 0.0));
    double partial = 0.5 / std::max(result_value, 1e-15);
    
    int result_idx = tape_->record_operation(AADOpType::SQRT, tape_index_, 
                                           -1, partial, 0.0);
    return GPUAADNumber(result_idx, tape_);
}

GPUAADNumber GPUAADNumber::norm_cdf() const {
    if (!tape_) throw std::runtime_error("No tape attached");
    
    // Standard normal CDF approximation
    double x = value();
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;
    
    int sign = (x >= 0) ? 1 : -1;
    x = std::abs(x) / std::sqrt(2.0);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * std::exp(-x*x);
    double result_value = 0.5 * (1.0 + sign * y);
    
    // PDF for derivative
    double partial = std::exp(-0.5 * value() * value()) / std::sqrt(2.0 * M_PI);
    
    int result_idx = tape_->record_operation(AADOpType::NORM_CDF, tape_index_, 
                                           -1, partial, 0.0);
    return GPUAADNumber(result_idx, tape_);
}

double GPUAADNumber::value() const {
    return tape_ ? tape_->get_value(tape_index_) : 0.0;
}

double GPUAADNumber::adjoint() const {
    return tape_ ? tape_->get_adjoint(tape_index_) : 0.0;
}

// Free function implementations
GPUAADNumber log(const GPUAADNumber& x) { return x.log(); }
GPUAADNumber exp(const GPUAADNumber& x) { return x.exp(); }
GPUAADNumber sqrt(const GPUAADNumber& x) { return x.sqrt(); }
GPUAADNumber norm_cdf(const GPUAADNumber& x) { return x.norm_cdf(); }
