# Ising-GPU
In this project, we investigate the GPU accelerated Monte Carlo simulations of the 2D Ising model. Parallel single-site flips will be applied to a large lattice up to an 1024*1024 lattice.

Two functions, `show_lattice` and `test_temperature`, are designed to test the Ising model:

```
void show_lattice(const int N, const double beta, const int ITER);
```
Display the lattice of size N for given inverse temperature beta. ITER specifies the number of iterations.

```
void test_temperature(const int N, const int ITER);
```
Compute 4 physical quantities: M (magnetization), X (susceptibility), E (energy), C (specific heat) for a lattice of size N.

## Run this program
In your command line, write
```
nvcc -std=c++11 -lcurand -o a main.cu test.cu lattice.cu Metroplis.cu
./a
```
