// Homework 4 problem 3 question (a)
// Implement the butterfly recursive doubling algorithm for All-Gather.

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const bool PRINT = true;
const bool CHECK = false;

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
    double* global_data =  new double[global_data_size];
    double* global_data_ref =  new double[global_data_size];

    for(int j = 0; j < local_data_size; j++) {
        localdata[j] = rand() % 100;
    }

    double time, start;

    // My_Allgather Implementation

    start = MPI_Wtime();

    for (int i = 1; i < nProcs; i>>=1)
    {
        MPI_Sendrecv();
    }
    

    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "My_Scatter: " << time << " seconds" << endl;
    }


    // MPI_Allgather() verifies the correctness
    start = MPI_Wtime();

    MPI_Allgather(localdata, local_data_size, MPI_DOUBLE, global_data_ref, local_data_size, MPI_DOUBLE, MPI_COMM_WORLD);
   
    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "MPI_Scatter: " << time << " seconds" << endl;
    }

    if (PRINT) {

        cout << "\nRank " << rank << " started with values\n";
        for(int i = 0; i < local_data_size; i++) {
            cout << localdata[i] << " ";
        }
        cout << endl;

        // cout << "\nRank " << rank << " ended with values\n";
        // for(int i = 0; i < global_data_size; i++) {
        //     cout << global_data[i] << " ";
        // }
        // cout << endl;

        cout << "\nRank " << rank << " should end with values\n";
        for(int i = 0; i < global_data_size; i++) {
            cout << global_data_ref[i] << " ";
        }
        cout << endl;
    }

    // if (CHECK) {
    //     for (int i = 0; i < local_data_size; i++)
    //     {
    //         if (localdata[i]!=localdata_ref[i]) {
    //             cerr << "Does not match the MPI_Scatter!" << endl;
    //         }
    //     }
    // }

    MPI_Finalize();
}