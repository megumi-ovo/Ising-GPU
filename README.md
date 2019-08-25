# Ising-GPU
In this project, we investigate the GPU accelerated Monte Carlo simulations of the 2D Ising model. Parallel single-site flips will be applied to a lattice of a large size up to 1024*1024.

## Two functions to test the Ising model
Two functions, `show_lattice` and `test_temperature`, are designed to test the Ising model.

```
void show_lattice(const int N, const double beta, const int ITER);
```
displays the lattice of size N for given inverse temperature beta, where ITER specifies the number of iterations.

```
void test_temperature(const int N, const int ITER);
```
computes 4 physical quantities: M (magnetization), X (susceptibility), E (energy), C (specific heat) for a lattice of size N.

These functions are defined in `test.cu` and called `main.cu`. You can modify parameters in `main.cu` and choose an arbitrary function to call.

## Test environment
System: Ubuntu 16.04
CPU: Intel(R) Xeon(R) CPU @ 2.20GHz (x4)
GPU:  Nvidia Tesla K80 (x1)
CUDA Toolkit: 10.1

## Run this program (require a CUDA-enabled GPU)
In your command line, input
```
nvcc -std=c++11 -lcurand -o a main.cu test.cu lattice.cu Metroplis.cu
./a
```
