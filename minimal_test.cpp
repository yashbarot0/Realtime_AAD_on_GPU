#include <iostream>
#include <cstring>

// Minimal CUDA test - just check if basic CUDA calls work
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Define minimal CUDA structs for compilation
typedef int cudaError_t;
#define cudaSuccess 0

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int maxThreadsPerBlock;
    int multiProcessorCount;
    // Padding to match actual struct size
    char padding[1024];
};

extern "C" {
    cudaError_t cudaGetDeviceCount(int* count);
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
}
#endif

int main() {
    std::cout << "Minimal CUDA Test" << std::endl;
    std::cout << "=================" << std::endl;
    
    try {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        std::cout << "CUDA Device Count: " << deviceCount << std::endl;
        std::cout << "Error Code: " << error << std::endl;
        
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            memset(&prop, 0, sizeof(prop));
            
            error = cudaGetDeviceProperties(&prop, 0);
            std::cout << "Get Properties Error: " << error << std::endl;
            
            if (error == cudaSuccess) {
                // Ensure null termination
                prop.name[255] = '\0';
                std::cout << "GPU Name: " << prop.name << std::endl;
                std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
                std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
            } else {
                std::cout << "Failed to get device properties" << std::endl;
            }
        }
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (...) {
        std::cout << "Exception caught!" << std::endl;
        return 1;
    }
}
