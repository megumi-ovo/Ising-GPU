#ifndef test_h
#define test_h
#include "lattice.h"

// display the lattice 
void show_lattice(const int N, const double beta, const int ITER);

// calculate observables at given beta
void test_observable(const lattice &sigma, const double beta, const int ITER, std::ofstream &fout);

// calculate observables for T in [1.0,4.0]
void test_temperature(const int N, const int ITER);

#endif