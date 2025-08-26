#ifndef GPU_AAD_TAPE_H
#define GPU_AAD_TAPE_H

#include "AADTypes.h"
#include <cuda_runtime.h>
#include <vector>

class GPUAADTape {
private:
    // GPU memory pointers
    double* d_values_;
    double* d_adjoints_;
    GPUTapeEntry* d_tape_;
    int* d_tape_size_;
    
    // Host memory
    std::vector<double> h_values_;
    std::vector<double> h_adjoints_;
    std::vector<GPUTapeEntry> h_tape_;
    
    // Configuration
    int max_tape_size_;
    int max_variables_;
    int current_tape_size_;
    bool initialized_;
    
    // CUDA streams
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;

public:
    GPUAADTape(int max_tape_size = 1000, int max_variables = 1000);
    ~GPUAADTape();
    
    // Initialization
    bool initialize();
    void cleanup();
    
    // Memory management
    bool allocate_gpu_memory();
    void free_gpu_memory();
    
    // Tape operations
    int add_variable(double value);
    int record_operation(AADOpType op, int input1_idx, int input2_idx = -1, 
                        double partial1 = 0.0, double partial2 = 0.0);
    
    // GPU operations
    void copy_to_gpu();
    void copy_from_gpu();
    void reset_adjoints();
    void set_adjoint(int var_idx, double value);
    
    // Reverse pass
    void propagate_adjoints();
    
    // Getters
    double get_value(int idx) const;
    double get_adjoint(int idx) const;
    int get_tape_size() const { return current_tape_size_; }
    bool is_initialized() const { return initialized_; }
    
    // GPU pointers (for kernels)
    double* get_gpu_values() { return d_values_; }
    double* get_gpu_adjoints() { return d_adjoints_; }
    GPUTapeEntry* get_gpu_tape() { return d_tape_; }
    int* get_gpu_tape_size() { return d_tape_size_; }
};

#endif // GPU_AAD_TAPE_H
