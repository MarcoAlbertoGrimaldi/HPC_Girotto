#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main( int argc, char * argv[] ){

    int rank = 0; // store the MPI identifier of the process
    int npes = 1; // store the number of MPI processes
    long int N = pow(2,4);

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &npes );

    int n = N/npes;
    int rest = N%npes;

    double* matrix;
    if(rank<rest) ++n;

    matrix = (double *) malloc(sizeof(double) * n * N);

    for(int i=0; i<n; ++i){
        for(int j=0; j<N; ++j){
            if(rank*n + i == j) matrix[N*i+j] = 1;
            else matrix[N*i+j]=0;
        }
    }

    fprintf( stderr, "\nPprocess %d of %d: inizialization completed...\n", rank, npes);
   
    FILE *output;
    if(rank==0) output = fopen("matrix.txt", "ab+");
    
    for(int k=1; k<npes; ++k){
        if(rank==0){
            for(int i=0; i<n; ++i) {
                for(int j=0; j<N; ++j){
                    fprintf(output,"%f ",matrix[N*i+j]);
                }
                fprintf(output,"\n");
            }
            fprintf(stderr, "\nPprocess %d of %d: writing chunk %d of %d completed...\n", rank, npes, k, npes);
            free(matrix);
        }

        if(rank==k) MPI_Send(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if(rank==0) MPI_Recv(&n, 1, MPI_INT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(rank==0) matrix = (double *) malloc(sizeof(double) * n * N);

        if(rank==k)MPI_Send(matrix, N*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(rank==0)MPI_Recv(matrix, N*n, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(rank==k)free(matrix);
    }

    if(rank==0) {
        for(int i=0; i<n; ++i) {
            for(int j=0; j<N; ++j){
                fprintf(output,"%f ",matrix[N*i+j]);
            }
        fprintf(output,"\n");
        }
        fprintf(stderr, "\nPprocess %d of %d: writing chunk %d of %d completed...\n", rank, npes, npes, npes);
        fclose(output);
        free(matrix);
    }

    MPI_Finalize();

    return 0;
}