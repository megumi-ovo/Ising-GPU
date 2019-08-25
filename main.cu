#include <fstream>
#include "test.h"

using namespace std;

int main()
{
    const int N = 1024;
    const int ITER = 40000;
    // test_temperature(N, ITER);
    show_lattice(N, 0.4407, ITER);
    // ---- the codes below is for test_observable at a given temperature ---- //
    /*lattice sigma(N);
    sigma.initialize('r');
    ofstream fout;
    fout.open("data.txt");
    test_observable(sigma, 0.50, ITER, fout);
    fout.close();*/
}

// nvc main.cu test.cu lattice.cu Metropolis.cu
