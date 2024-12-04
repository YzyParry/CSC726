// Homework 4 problem 3 question (a)
// Implement the binomial tree recursive halving algorithm for Scatter.

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const bool PRINT = true;

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int nProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank*12345);

    int m, n;
    m  = atoi(argv[1]); // local data volume
    // n  = atoi(argv[2]);
    int local_data_size = m;
    int global_data_size = m*nProcs;

    double* localdata = new double[local_data_size];
    double* localdata_ref = new double[local_data_size];
    double* global_data =  new double[global_data_size];
    if (rank == 0) {
        for(int j = 0; j < global_data_size; j++) {
            global_data[j] = rand() % 100;
        }
    }
    
    double time, start;
    // My_Scatter Implementation

    


    // MPI_Scatter() verifies the correctness
    start = MPI_Wtime();

    MPI_Scatter(global_data, local_data_size, MPI_DOUBLE, localdata_ref, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "MPI_Scatter: " << time << " seconds" << endl;
    }


    if (PRINT) {
        if (rank == 0)
        {
            cout << "\nRank " << rank << " start with values\n";
            for(int i = 0; i < global_data_size; i++) {
                cout << global_data[i] << " ";
            }
            cout << endl;
        }

        cout << "\nRank " << rank << " ended with values\n";
        for(int i = 0; i < local_data_size; i++) {
            cout << localdata[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
}