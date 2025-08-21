#include <iostream>
#include <cstring>

// Test to check actual struct sizes
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define REAL_CUDA_STRUCTS
#endif

int main() {
    std::cout << "CUDA Struct Size Analysis" << std::endl;
    std::cout << "=========================" << std::endl;
    
#ifdef REAL_CUDA_STRUCTS
    std::cout << "Using real CUDA headers" << std::endl;
    std::cout << "sizeof(cudaDeviceProp): " << sizeof(cudaDeviceProp) << " bytes" << std::endl;
    
    // Test if our access works
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        
        cudaError_t error = cudaGetDeviceProperties(&prop, 0);
        if (error == cudaSuccess) {
            std::cout << "Real values:" << std::endl;
            std::cout << "  Name: " << prop.name << std::endl;
            std::cout << "  Major: " << prop.major << std::endl;
            std::cout << "  Minor: " << prop.minor << std::endl;
            std::cout << "  Total Memory: " << prop.totalGlobalMem << std::endl;
            std::cout << "  Shared Memory: " << prop.sharedMemPerBlock << std::endl;
        }
    }
#else
    std::cout << "Using fallback definitions" << std::endl;
    
    struct FakeCudaDeviceProp {
        char name[256];
        int major;
        int minor;
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int maxThreadsPerBlock;
        int multiProcessorCount;
    };
    
    std::cout << "sizeof(FakeCudaDeviceProp): " << sizeof(FakeCudaDeviceProp) << " bytes" << std::endl;
#endif
    
    return 0;
}
