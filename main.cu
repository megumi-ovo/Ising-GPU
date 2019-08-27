#include <fstream>
#include "test.h"

using namespace std;

int main()
{
    const int N = 512;
    const int ITER = 10000;
    // test_temperature(N, ITER);
    // show_lattice(N, 0.4407, ITER);
    // ---- the codes below are for test_observable at a given temperature ---- //
    lattice sigma(N);
    sigma.initialize('1');
    ofstream fout;
    fout.open("data.txt");
    test_observable(sigma, 0.45, ITER, fout);
    fout.close();
}

// nvc main.cu test.cu lattice.cu Metropolis.cu
