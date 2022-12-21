#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <ctime>
#include <string>

#define SEED (62002)
#define EXACT_VALUE (1.0 / 24)
#define VOLUME (1.0)

using namespace std;

double F(double x, double y, double z) {
    return x * x * x * y * y * z;
}

double Integral(int n, long double s) {
    return VOLUME * s / n;
}

double randomDouble() {
    return (double)rand() / (double)RAND_MAX;
}

int main(int argc, char **argv) {
    double given_eps = 1.5e-6, eps = 1.0;
    given_eps = strtod(argv[1], NULL);
    double x, y, z;                    
    int myid, proc_amount, k = 0, total_count = 0;                  
    const int COUNT = 256;
    double total_sum = 0.0;
    double reduced_total_sum;                  

    MPI_Init(&argc, &argv);                
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);           
	MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);

    double start_time = MPI_Wtime();
    srand(myid + SEED + proc_amount * std::time(0));

    while (eps > given_eps) {
        while (k < COUNT / proc_amount) {
            x = randomDouble(); //0 <= x <= 1
            y = randomDouble(); //0 <= y <= 1
            z = randomDouble(); //0 <= z <= 1
            total_sum += F(x, y, z); //integrals are the same for 0 <= x, y, z <= 1 and -1 <= x, y, z <= 0
            ++k;
        }
        k = 0;
        total_count += COUNT;
        MPI_Allreduce(&total_sum, &reduced_total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        eps = abs(EXACT_VALUE - Integral(total_count, reduced_total_sum));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (myid == 0) {
        std::cout << "-----------------------------" << "\n\n";
        std::cout << "  Processes : " << proc_amount << "\n";
        std::cout << "  Epsilon : " << given_eps << "\n";
        std::cout << "  Execution time  : " << end_time - start_time << "\n";
        std::cout << "  Final error     : " << eps << "\n";
        std::cout << "  Total points    : " << total_count << "\n";
        std::cout << "  Calculated integral : " << Integral(total_count, reduced_total_sum) << "\n\n";
        std::cout << "-----------------------------" << "\n";
    }
    MPI_Finalize();
    return 0;
}