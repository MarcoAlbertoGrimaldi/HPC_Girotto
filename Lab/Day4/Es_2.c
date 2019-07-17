#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main( int argc, char * argv[] ){

    int rank = 0; // store the MPI identifier of the process
    int npes = 1; // store the number of MPI processes

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &npes );

    const size_t N = 20;

    int my_sum[N];
    int message[N];

    for (size_t j = 0; j < N; ++j){
	    my_sum[j] = 0;
	    message[j] = rank;
    }

    MPI_Request request;
    int request_complete = 0; 
 
    for(int i=0; i<npes; ++i){

        MPI_Isend(&message, N, MPI_INT, ((rank+1)%npes), rank, MPI_COMM_WORLD,&request);
        	for (size_t j = 0; j < N; ++j)
	    my_sum[j] += message[i];
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        MPI_Recv(&message, N, MPI_INT, ((rank-1)%npes), ((rank-1)%npes), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    fprintf( stderr, "\n Process %d of %d: my_sum now is: %u; \n", rank, npes, my_sum[N-1]);

    MPI_Finalize();

    return 0;
}