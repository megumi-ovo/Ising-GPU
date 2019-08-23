#include "lattice.h"

using namespace std;

lattice::lattice(const int N_): N(N_)
{
    cudaMalloc(&a, N*N*sizeof(int));
    cudaMalloc(&real_dist, N*N*sizeof(double));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937); // initialize rng
    curandSetPseudoRandomGeneratorSeed(gen, 0); // set seed
}

lattice::~lattice()
{
    cudaFree(a);
    cudaFree(real_dist);
}

__global__ void _initialize_1(int *a)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] = 1;
}

__global__ void _initialize_r(int *a, double *real_dist)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    a[i] = (real_dist[i]>0.5)?(1):(-1);
}

void lattice::initialize(char option)
{
    switch(option)
    {
        case '1': // all +1 spins
            _initialize_1<<<N,N>>>(a);
            return;
        case 'r': // random +1/-1 spins
            curandGenerateUniformDouble(gen, real_dist, N*N);
            _initialize_r<<<N,N>>>(a, real_dist);
            return;
    }
}

__global__ void _S(const int *a, int *obs)
{
    const int N = blockDim.x;
    int i = blockIdx.x;
    int i_ = _res(i+1,N);
    int j = threadIdx.x;
    int j_ = _res(j+1,N);
    int spin_1 = a[i*N+j];
    int spin_2 = a[i_*N+j]+a[i*N+j_];
    atomicAdd(obs+0, spin_1); // magnetization
    atomicAdd(obs+1, spin_1*spin_2); // energy
}

void S(const lattice &sigma, int *obs)
{
    const int N = sigma.N;
    cudaMemset(obs, 0, 2*sizeof(int));
    _S<<<N,N>>>(sigma.a, obs);
}

std::ostream & operator<<(std::ostream &os, const lattice &sigma)
{
    const int N = sigma.N;
    int i,j, a[N*N];
    // memory copy from device to host
    cudaMemcpy(a, sigma.a, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            os << a[i*N+j] << " ";
        }
        os << endl;
    }
    return os;
}