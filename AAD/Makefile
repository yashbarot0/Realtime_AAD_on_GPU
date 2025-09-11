# Enhanced Makefile with better optimization
NVCC = nvcc
CXX = g++

# Optimized flags for benchmarking
NVCC_FLAGS = -std=c++17 -arch=sm_75 --extended-lambda -rdc=true -O3 -use_fast_math --maxrregcount=32 -lineinfo
CXX_FLAGS = -std=c++17 -O3 -march=native -fPIC -fopenmp -DNDEBUG

INCLUDES = -Iinclude -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -lcudart -lpthread -lgomp

CUDA_SOURCES = src/blackscholes_aad_kernel.cu
CPP_SOURCES = src/main.cpp src/GPUAADNumber.cpp src/GPUAADTape.cpp src/python_gpu_interface.cpp
ALL_OBJECTS = $(CUDA_SOURCES:.cu=.o) $(CPP_SOURCES:.cpp=.o)

TARGET = aad_gpu_benchmark

all: $(TARGET)

$(TARGET): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(ALL_OBJECTS) $(TARGET)

test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test
