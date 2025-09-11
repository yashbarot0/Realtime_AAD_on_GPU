#ifndef DEVICE_AAD_CUH
#define DEVICE_AAD_CUH

#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Forward declaration
class DeviceAADTape;

// Device-compatible AAD Number
class DeviceAADNumber {
	public:
		    double value;
		        int tape_index;
			    DeviceAADTape* tape;

			        // Constructors
			        __device__ DeviceAADNumber() : value(0.0), tape_index(-1), tape(nullptr) {}
				    __device__ DeviceAADNumber(double val) : value(val), tape_index(-1), tape(nullptr) {}
				        __device__ DeviceAADNumber(double val, int idx, DeviceAADTape* t) : value(val), tape_index(idx), tape(t) {}

					    // Arithmetic operators
					    __device__ DeviceAADNumber operator+(const DeviceAADNumber& other) const;
					        __device__ DeviceAADNumber operator-(const DeviceAADNumber& other) const;
						    __device__ DeviceAADNumber operator*(const DeviceAADNumber& other) const;
						        __device__ DeviceAADNumber operator/(const DeviceAADNumber& other) const;
							    __device__ DeviceAADNumber operator-() const;

							        // Math functions
							        __device__ DeviceAADNumber log() const;
								    __device__ DeviceAADNumber exp() const;
								        __device__ DeviceAADNumber sqrt() const;
									    __device__ DeviceAADNumber norm_cdf() const;
};

// Device-compatible tape entry
struct DeviceTapeEntry {
	    int result_idx;
	        int input1_idx;
		    int input2_idx;
		        int op_type;
			    double partial1;
			        double partial2;
};

// Device-compatible tape
class DeviceAADTape {
	public:
		    double* values;
		        double* adjoints;
			    DeviceTapeEntry* entries;
			        int* current_size;
				    int max_size;

				        __device__ DeviceAADTape(double* vals, double* adjs, DeviceTapeEntry* ents, int* size, int max_sz)
						        : values(vals), adjoints(adjs), entries(ents), current_size(size), max_size(max_sz) {}

					    __device__ int record_variable(double value);
					        __device__ int record_operation(int op, int in1, int in2, double p1, double p2, double result_val);
						    __device__ void propagate_adjoints();
};

// Device math functions
__device__ double device_safe_log(double x);
__device__ double device_safe_exp(double x);
__device__ double device_safe_sqrt(double x);
__device__ double device_norm_cdf(double x);
__device__ double device_norm_pdf(double x);

#endif // DEVICE_AAD_CUH

