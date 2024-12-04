// Homework 4 problem 3 question (a)
// Implement the binomial tree recursive halving algorithm for Scatter.

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const bool PRINT = false;
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
    double* localdata_ref = new double[local_data_size];
    double* global_data =  new double[global_data_size];
    if (rank == 0) {
        for(int j = 0; j < global_data_size; j++) {
            global_data[j] = rand() % 100;
        }
    }
    double time, start;

    // My_Scatter Implementation

    // For example: 8 proc
    // step 1: 0->4
    // step 2: 0->2, 4->6
    // step 3: 0->1, 2->3, 4->5, 6->7
    start = MPI_Wtime();
    int data_size;
    if (rank == 0) {
        data_size = global_data_size;
    } else {
        data_size = 0;
    }

    for (int i = nProcs; i > 1; i >>= 1) {
        int dest, source;

        if (rank % i == 0) { // Sender
            dest = rank + (i >> 1);
            // cout << rank << " send msg to " << dest << " at i=" << i << endl;
            int send_count = data_size / 2;
            int offset = data_size - send_count;
            MPI_Send(global_data + offset, send_count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            // // Keep only the first half
            data_size = offset;
            
        } else if (rank % i == (i >> 1)) { // Receiver
            source = rank - (i >> 1);
            // cout << rank << " recv msg from " << source << " at i=" << i << endl;
            data_size = (global_data_size * i) / (2 * nProcs);
            int recv_count = data_size;
            MPI_Recv(global_data, recv_count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    memcpy(localdata, global_data, local_data_size * sizeof(double));

    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "My_Scatter: " << time << " seconds" << endl;
    }


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

        cout << "\nRank " << rank << " should end with values\n";
        for(int i = 0; i < local_data_size; i++) {
            cout << localdata_ref[i] << " ";
        }
        cout << endl;
    }

    if (CHECK) {
        for (int i = 0; i < local_data_size; i++)
        {
            if (localdata[i]!=localdata_ref[i]) {
                cerr << "Does not match the MPI_Scatter!" << endl;
            }
        }
    }

    MPI_Finalize();
}