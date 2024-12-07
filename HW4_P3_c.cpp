// Homework 4 problem 3 question (c)
// Implement Broadcast using Scatter followed by All-Gather.

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

    if (rank == 0) {
        for(int j = 0; j < global_data_size; j++) {
            global_data[j] = rand() % 100;
            global_data_ref[j] = global_data[j];
        }
    }

    double time, start;

    // My_Broadcast Implementation
    // My_Scatter => My_allgather

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
        cout << "My_Broadcast: " << time << " seconds" << endl;
    }


    // MPI_Bcast() verifies the correctness
    start = MPI_Wtime();

    MPI_Bcast(global_data_ref, global_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
    time = MPI_Wtime() - start;
    if (rank == 0) {
        cout << "MPI_Bcast: " << time << " seconds" << endl;
    }

    if (PRINT) {
        if (rank==0) {
            cout << "\nRank " << rank << " started with values\n";
            for(int i = 0; i < global_data_size; i++) {
                cout << global_data[i] << " ";
            }
            cout << endl;
        }
        

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