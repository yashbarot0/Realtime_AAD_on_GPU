#ifndef GPU_AAD_NUMBER_H
#define GPU_AAD_NUMBER_H

#include "GPUAADTape.h"

class GPUAADNumber {
private:
    int tape_index_;
    GPUAADTape* tape_;

public:
    // Constructors
    GPUAADNumber(double value, GPUAADTape* tape);
    GPUAADNumber(int tape_index, GPUAADTape* tape);

    // Copy constructor and assignment
    GPUAADNumber(const GPUAADNumber& other);
    GPUAADNumber& operator=(const GPUAADNumber& other);

    // Arithmetic operators
    GPUAADNumber operator+(const GPUAADNumber& other) const;
    GPUAADNumber operator-(const GPUAADNumber& other) const;
    GPUAADNumber operator*(const GPUAADNumber& other) const;
    GPUAADNumber operator/(const GPUAADNumber& other) const;
    GPUAADNumber operator-() const;

    // Math functions
    GPUAADNumber log() const;
    GPUAADNumber exp() const;
    GPUAADNumber sqrt() const;
    GPUAADNumber norm_cdf() const;

    // Getters
    double value() const;
    double adjoint() const;
    int get_tape_index() const { return tape_index_; }
    GPUAADTape* get_tape() const { return tape_; }
};

// Math function declarations
GPUAADNumber log(const GPUAADNumber& x);
GPUAADNumber exp(const GPUAADNumber& x);
GPUAADNumber sqrt(const GPUAADNumber& x);
GPUAADNumber norm_cdf(const GPUAADNumber& x);

#endif // GPU_AAD_NUMBER_H
