// Homework 4 problem 3 question (b)
// Implement the butterfly recursive doubling algorithm for All-Gather.

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const bool PRINT = false;
const bool CHECK = true;

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
        global_data[j] = rand() % 100;
        localdata[j] = global_data[j];
    }

    double time, start;

    // My_Allgather Implementation

    start = MPI_Wtime();
    int offset, dest, source, data_size;
    for (int i = 1; i < nProcs; i<<=1)
    {
        int dest, source;
        int data_size = m * i;
        int send_offset, recv_offset;

        if ((rank % (i << 1)) < i) {
            dest = rank + i;
            source = rank + i;
            send_offset = 0;
            recv_offset = data_size; // 收到的数据接在后面
        } else {
            dest = rank - i;
            source = rank - i;
            send_offset = 0;
            recv_offset = 0; // 收到的数据应该放在前面
        }

        if ((rank % (i << 1)) >= i) {
            double* temp = new double[data_size];
            MPI_Sendrecv(global_data + send_offset, data_size, MPI_DOUBLE, dest, 0,
                         temp, data_size, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 当前 global_data 前 data_size 个元素是自己的数据，
            // 先把自己的数据段挪到后面，腾出前面的位置给 temp 中的 partner 数据。
            memmove(global_data + data_size, global_data, data_size * sizeof(double));
            memcpy(global_data, temp, data_size * sizeof(double));
            delete[] temp;
        } else {
            MPI_Sendrecv(global_data + send_offset, data_size, MPI_DOUBLE, dest, 0,
                         global_data + recv_offset, data_size, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    

    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "My_Allgather: " << time << " seconds" << endl;
    }


    // MPI_Allgather() verifies the correctness
    start = MPI_Wtime();

    MPI_Allgather(localdata, local_data_size, MPI_DOUBLE, global_data_ref, local_data_size, MPI_DOUBLE, MPI_COMM_WORLD);
   
    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "MPI_Allgather: " << time << " seconds" << endl;
    }

    if (PRINT) {

        cout << "\nRank " << rank << " started with values\n";
        for(int i = 0; i < local_data_size; i++) {
            cout << localdata[i] << " ";
        }
        cout << endl;

        cout << "\nRank " << rank << " ended with values\n";
        for(int i = 0; i < global_data_size; i++) {
            cout << global_data[i] << " ";
        }
        cout << endl;

        cout << "\nRank " << rank << " should end with values\n";
        for(int i = 0; i < global_data_size; i++) {
            cout << global_data_ref[i] << " ";
        }
        cout << endl;
    }

    if (CHECK) {
        for (int i = 0; i < global_data_size; i++)
        {
            if (global_data[i]!=global_data_ref[i]) {
                cerr << "Does not match the MPI_Scatter!" << endl;
            }
        }
    }

    MPI_Finalize();
}