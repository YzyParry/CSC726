#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
using namespace std;

const bool DEBUG = true;

// initialize matrix and vectors (A is mxn, x is xn-vec)
void init_rand(double* a, int m, int n, double* x, int xn);
// local matvec: y = y+A*x, where A is m x n
void local_gemv(double* A, double* x, double* y, int m, int n);

int main(int argc, char** argv) {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int nProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank*12345);

    // Read dimensions and processor grid from command line arguments
    if(argc != 5) {
        cerr << "Usage: ./a.out rows cols pr pc" << endl;
        return 1;
    }
    int m, n, pr, pc;
    m  = atoi(argv[1]);
    n  = atoi(argv[2]);
    pr = atoi(argv[3]);
    pc = atoi(argv[4]);
    if(pr*pc != nProcs) {
        cerr << "Processor grid doesn't match number of processors" << endl;
        return 1;
    }
    if(m % pr || n % pc || m % nProcs || n % nProcs) {
        cerr << "Processor grid doesn't divide rows and columns evenly" << endl;
        return 1;
    }

    // Set up row and column communicators
    int ranki = rank % pr; // proc row coordinate
    int rankj = rank / pr; // proc col coordinate
    
    // Create row and column communicators using MPI_Comm_split
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, ranki, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, rankj, rank, &col_comm);

    // Check row and column communicators and proc coordinates
    int rankichk, rankjchk;
    MPI_Comm_rank(row_comm,&rankjchk);
    MPI_Comm_rank(col_comm,&rankichk);
    if(ranki != rankichk || rankj != rankjchk) {
        cerr << "Processor ranks are not as expected, check row and column communicators" << endl;
        return 1;
    }

    // Initialize matrices and vectors
    int mloc = m / pr;     // number of rows of local matrix
    int nloc = n / pc;     // number of cols of local matrix
    int ydim = m / nProcs; // number of entries of local output vector
    int xdim = n / nProcs; // number of entries of local input vector
    double* Alocal = new double[mloc*nloc];
    double* xlocal = new double[xdim];
    double* ylocal = new double[ydim];
    init_rand(Alocal, mloc, nloc, xlocal, xdim);
    memset(ylocal,0,ydim*sizeof(double));

    // start timer
    double time, start = MPI_Wtime();

    double* xglobal = new double[nloc];
    double* yglobal = new double[mloc];
    // Communicate input vector entries
    MPI_Allgather(xlocal,xdim, MPI_DOUBLE, xglobal, xdim, MPI_DOUBLE, col_comm);

    if (rank==0){
        for (int i=0;i<xdim;i++){
        cout << xglobal[i] << " ";
        }
        cout << endl;
    }

    if (rank==0){
        for (int i = 0; i < mloc; i++) {
            for (int j=0; j < nloc; j++) {
                cout << Alocal[i*nloc + j] << " ";
            }
            cout << endl;
        }
    }


    local_gemv(Alocal, xglobal, ylocal, mloc, nloc);

    // cout << mloc << nloc << endl;

    if (rank==0){
        for (int i=0;i<mloc;i++){
        cout << ylocal[i] << " ";
        }
        cout << endl;
    }

    // Communicate output vector entries
    MPI_Reduce(ylocal, yglobal, mloc, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    // int * recvcounts[pr];
    // for (int i=0;i<pr;i++) recvcounts[i]=ydim;
    // MPI_Reduce_scatter(ylocal,yglobal,&recvcounts,MPI_DOUBLE,MPI_SUM,row_comm);
    
    // if (rank==0){
    //     for (int i=0;i<mloc;i++){
    //     cout << yglobal[i] << " ";
    //     }
    //     cout << endl;
    // }

    // Redistribute the output vector to match input vector
    MPI_Scatter(yglobal,ydim,MPI_DOUBLE,ylocal,ydim,MPI_DOUBLE,0,row_comm);

    // MPI_Reduce_scatter()

    // Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - start;

    // Print results for debugging
    if(DEBUG) {
        cout << "\nProc (" << ranki << "," << rankj << ") started with x values\n";
        for(int j = 0; j < xdim; j++) {
            cout << xlocal[j] << " ";
        }
        cout << "\nProc (" << ranki << "," << rankj << ") has local matrix\n";
        for (int i = 0; i < mloc; i++) {
            for (int j=0; j < nloc; j++) {
                cout << Alocal[i*nloc + j] << " ";
            }
            cout << endl;
        }

        cout << "Proc (" << ranki << "," << rankj << ") ended with y values\n";
        for(int i = 0; i < ydim; i++) {
            cout << ylocal[i] << " ";
        }
        cout << endl; // flush now
    }

    // Print time
    if(!rank) {
        cout << "Time elapsed: " << time << " seconds" << endl;
    }

    // Clean up
    delete [] ylocal;
    delete [] xlocal;
    delete [] Alocal;
    delete [] xglobal;
    delete [] yglobal;
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
}

void local_gemv(double* a, double* x, double* y, int m, int n) {
    // order for loops to match col-major storage
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            y[i] += a[i+j*m] * x[j];
        }
    }
}

void init_rand(double* a, int m, int n, double* x, int xn) {
    // init matrix
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            a[i+j*m] = rand() % 100;
        }
    }
    // init input vector x
    for(int j = 0; j < xn; j++) {
        x[j] = rand() % 100;
    }
}
