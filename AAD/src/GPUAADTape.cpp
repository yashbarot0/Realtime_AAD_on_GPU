#include "../include/GPUAADTape.h"
#include <iostream>
#include <cuda_runtime.h>

GPUAADTape::GPUAADTape(int max_tape_size, int max_variables)
    : max_tape_size_(max_tape_size), max_variables_(max_variables),
      current_tape_size_(0), initialized_(false),
      d_values_(nullptr), d_adjoints_(nullptr), d_tape_(nullptr), d_tape_size_(nullptr) {
    
    // Create CUDA streams
    cudaStreamCreate(&compute_stream_);
    cudaStreamCreate(&memory_stream_);
}

GPUAADTape::~GPUAADTape() {
    cleanup();
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(memory_stream_);
}

bool GPUAADTape::initialize() {
    if (initialized_) return true;
    
    h_values_.reserve(max_variables_);
    h_adjoints_.reserve(max_variables_);
    h_tape_.reserve(max_tape_size_);
    
    if (!allocate_gpu_memory()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool GPUAADTape::allocate_gpu_memory() {
    cudaError_t err;
    
    err = cudaMalloc(&d_values_, max_variables_ * sizeof(double));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_adjoints_, max_variables_ * sizeof(double));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_tape_, max_tape_size_ * sizeof(GPUTapeEntry));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_tape_size_, sizeof(int));
    if (err != cudaSuccess) return false;
    
    return true;
}

void GPUAADTape::free_gpu_memory() {
    if (d_values_) { cudaFree(d_values_); d_values_ = nullptr; }
    if (d_adjoints_) { cudaFree(d_adjoints_); d_adjoints_ = nullptr; }
    if (d_tape_) { cudaFree(d_tape_); d_tape_ = nullptr; }
    if (d_tape_size_) { cudaFree(d_tape_size_); d_tape_size_ = nullptr; }
}

void GPUAADTape::cleanup() {
    if (initialized_) {
        free_gpu_memory();
        initialized_ = false;
    }
}

int GPUAADTape::add_variable(double value) {
    h_values_.push_back(value);
    h_adjoints_.push_back(0.0);
    return h_values_.size() - 1;
}

int GPUAADTape::record_operation(AADOpType op, int input1_idx, int input2_idx,
                                double partial1, double partial2) {
    GPUTapeEntry entry;
    entry.result_idx = current_tape_size_;
    entry.input1_idx = input1_idx;
    entry.input2_idx = input2_idx;
    entry.op_type = op;
    entry.partial1 = partial1;
    entry.partial2 = partial2;
    
    h_tape_.push_back(entry);
    current_tape_size_++;
    
    return entry.result_idx;
}

void GPUAADTape::copy_to_gpu() {
    if (!initialized_) return;
    
    cudaMemcpy(d_values_, h_values_.data(), h_values_.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjoints_, h_adjoints_.data(), h_adjoints_.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tape_, h_tape_.data(), h_tape_.size() * sizeof(GPUTapeEntry), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tape_size_, &current_tape_size_, sizeof(int), cudaMemcpyHostToDevice);
}

void GPUAADTape::copy_from_gpu() {
    if (!initialized_) return;
    
    cudaMemcpy(h_values_.data(), d_values_, h_values_.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_adjoints_.data(), d_adjoints_, h_adjoints_.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

void GPUAADTape::reset_adjoints() {
    std::fill(h_adjoints_.begin(), h_adjoints_.end(), 0.0);
    if (initialized_) {
        cudaMemset(d_adjoints_, 0, h_adjoints_.size() * sizeof(double));
    }
}

void GPUAADTape::set_adjoint(int var_idx, double value) {
    if (var_idx >= 0 && var_idx < h_adjoints_.size()) {
        h_adjoints_[var_idx] = value;
    }
}

void GPUAADTape::propagate_adjoints() {
    // This would be implemented for CPU-side AD if needed
    // For GPU implementation, see the CUDA kernel
}

double GPUAADTape::get_value(int idx) const {
    if (idx >= 0 && idx < h_values_.size()) {
        return h_values_[idx];
    }
    return 0.0;
}

double GPUAADTape::get_adjoint(int idx) const {
    if (idx >= 0 && idx < h_adjoints_.size()) {
        return h_adjoints_[idx];
    }
    return 0.0;
}
