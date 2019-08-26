#include <fstream>
#include <chrono>
#include "lattice.h"
#include "Metropolis.h"

using namespace std;
using namespace std::chrono;

high_resolution_clock::time_point t1;
high_resolution_clock::time_point t2;

void show_lattice(const int N, const double beta, const int ITER)
{
    lattice sigma(N);
    sigma.initialize('r');
    ofstream fout;
    char filename[100];
    for(int epoch=1;epoch<=6;epoch++)
    {
        cout << "epoch " << epoch << " ..." << endl; 
        sprintf(filename, "data%d.txt", epoch);
        fout.open(filename);
        t1 = high_resolution_clock::now();
        for(int iter=0;iter<ITER;iter++){
            Metropolis_sweep(sigma, beta);
        }
        t2 = high_resolution_clock::now();
        fout << sigma;
        fout.close();
        double Time = 1.0*duration_cast<microseconds>(t2-t1).count()/1e6;
        cout << "Time: " << Time  << " second(s)" << endl;
    }
}

void test_observable(const lattice &sigma, const double beta, const int ITER, ofstream &fout)
{
    const int N = sigma.N;
    // ---- lattice warm up ---- //
    t1 = high_resolution_clock::now();
    for(int iter=0;iter<ITER;iter++){
        Metropolis_sweep(sigma, beta);
    }
    // ---- calculate observables ---- //
    double mag = 0, mag2 = 0;
    double ene = 0, ene2 = 0; 
    int obs[2]; // observables at each iteration
    for(int iter=0;iter<ITER;iter++){
        Metropolis_sweep(sigma, beta);
        S(sigma, obs); // calculate mag & ene
        mag  += obs[0];
        mag2 += obs[0]*obs[0];
        ene  += obs[1];
        ene2 += obs[1]*obs[1];
    }
    t2 = high_resolution_clock::now();
    // ---- calculate averages ---- //
    mag  /= ITER;
    mag2 /= ITER;
    ene  /= ITER;
    ene2 /= ITER;
    // ---- output results ---- //
    // 1. Magnetization per spin
    double M = mag/(N*N);
    // 2. Susceptibility
    double X = beta*(mag2-mag*mag)/(N*N);
    // 3. Energy per spin
    double E = -ene/(N*N);
    // 4. Specific heat
    double C = beta*beta*(ene2-ene*ene)/(N*N);
    cout << "M: " << M << endl;
    cout << "X: " << X << endl;
    cout << "E: " << E << endl;
    cout << "C: " << C << endl;
    fout << M << " " << X << " " << E << " " << C << endl;
    double Time = 1.0*duration_cast<microseconds>(t2-t1).count()/1e6;
    cout << "Time: " << Time  << " second(s)" << endl;
}

void test_temperature(const int N, const int ITER)
{
    lattice sigma(N);
    sigma.initialize('1');
    double T; // temperature
    double beta; // inverse temperature
    const int NUM = 1000; // number of choices of T
    ofstream fout;
    fout.open("data.txt");
    for(int i=0;i<=NUM;i++)
    {
        T = 1.0+3.0/NUM*i; // T varies in [1.0,4.0]
        beta = 1.0/T;
        cout << "Current temperature: " << T << endl;
        test_observable(sigma, beta, ITER, fout);
    }
    fout.close();
}
