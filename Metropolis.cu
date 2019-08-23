#include <cmath>
#include "lattice.h"

using namespace std;

__global__ void _Metropolis(int *a, const double beta, const int res, const double *real_dist)
{
    const int N = blockDim.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    if(((i+j)&1)==res)
    {
        // indexes around spin (i,j)
        int i_ = _res(i+1,N), _i = _res(i-1,N);
        int j_ = _res(j+1,N), _j = _res(j-1,N);
        int I = i*N+j;
        // calculate energy difference
        int spin_sum = a[i_*N+j]+a[_i*N+j]+a[i*N+j_]+a[i*N+_j];
        double h_diff = 2*beta*a[I]*spin_sum;
        // attempt to flip the spin
        if(h_diff<0||exp(-h_diff)>real_dist[I]) a[I] *= -1;
    }
}

void Metropolis_sweep(const lattice &sigma, const double beta)
{
    const int N = sigma.N;
    curandGenerateUniformDouble(sigma.gen, sigma.real_dist, N*N); // generate (0,1) random numbers
    _Metropolis<<<N,N>>>(sigma.a, beta, 0, sigma.real_dist); // sweep even (0) spins
    _Metropolis<<<N,N>>>(sigma.a, beta, 1, sigma.real_dist); // sweep odd  (1) spins
}