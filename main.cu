#include <fstream>
#include "test.h"

using namespace std;

int main()
{
    const int N = 128;
    const int ITER = 10000;
    test_temperature(N, ITER);
    // show_lattice(N, 0.4407, ITER);
}

// nvc main.cu test.cu lattice.cu Metropolis.cu