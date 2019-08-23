#ifndef lattice_h
#define lattice_h
#include <iostream>
#include <cuda.h>
#include <curand.h>

__forceinline__ __device__ int _res(const int i, const int N)
{
    return (i<0)?(i+N):(i%N);
}

// lattice of spins: 2D Ising model
struct lattice
{
    const int N; // size of lattice
    int *a; // N*N spin
    double *real_dist; // (0,1) real number uniform distribution
    curandGenerator_t gen; // random number generator
    lattice(const int N);
    ~lattice();
    void initialize(char option); // initialize with random +1/-1 spins
};

// calculate observables: magnetization & energy
void S(const lattice &sigma, int *obs);

// TEST FUNCTION: show the lattice 
std::ostream & operator<<(std::ostream &os, const lattice &sigma);

#endif