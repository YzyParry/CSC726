#!/bin/bash
#SBATCH --job-name="yzy_pdgemv"    # Name that appears in queue
#SBATCH --partition=large          # Resource group (small/large)
#SBATCH --account=csc726           # SLURM account
#SBATCH --nodes=4                  # Number of Nodes
#SBATCH --ntasks-per-node=44       # Number of CPU Cores
#SBATCH --mem=196gb                # Requested memory
#SBATCH --time=00-02:00:00         # Job duration in DD-HH:MM:SS
#SBATCH --mail-user=yangz223@wfu.edu
#SBATCH --mail-type=END,FAIL       # Updates to provide mail-user

# Load your software
module load compilers/gcc/12.3.0
module load mpi/openmpi/4.1.6

# Go to working directory
cd /deac/csc/classes/csc726/${USER}/1106Wed

# Execute your program

mpirun -n 1 ./a.out 14400 14400 1 1 0
mpirun -n 4 ./a.out 14400 14400 2 2 0
mpirun -n 9 ./a.out 14400 14400 3 3 0
mpirun -n 25 ./a.out 14400 14400 5 5 0
mpirun -n 36 ./a.out 14400 14400 5 5 0
mpirun -n 100 ./a.out 14400 14400 10 10 0
mpirun -n 144 ./a.out 14400 14400 12 12 0
mpirun -n 176 ./a.out 17600 17600 11 16 0






