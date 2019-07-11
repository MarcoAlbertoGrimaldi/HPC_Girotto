#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <mpi.h>

int main( int argc, char * argv[] ){

    int rank = 0; // store the MPI identifier of the process
    int npes = 1; // store the number of MPI processes
    long int N = pow(2,20);

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &npes );

    long int n = N/npes;
    int rest = N%npes;

    if(rank<rest) ++n;

    matrix = (double *) malloc(sizeof(double) * n * N);

    for(int i=0; i<n; ++i){
        for(int j=n*rank; j<n*(rank+1)){
            if(i+rank*n == j) matrix[j][i]=1;
            else matrix[j][i]=0;
        }
    }

    if(rank==0) {
        FILE *output = fopen("matrix.txt", "ab+");
    }

    for(int k=1; k<npes; ++k){
        if(rank==0){
            for(i=0;i<n;i++) {
                for(int j=n*rank; j<n*(rank+1)){
                    fprintf(output,"%d ",matrix[j][i]);
                }
            fprintf(output),"\n");}
        }

        if(rank==k) MPI_Send(matrix, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(rank==0) MPI_Recv(matrix, 1, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if(rank==0) //write last chunk
}