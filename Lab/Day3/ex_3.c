#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>


int main( int argc, char * argv[] ){

    const double a = 0;
    double const b = 1;
    double res = 0;
    double start = MPI_Wtime();;
    double stop = 0;
    int rank = 0; // store the MPI identifier of the process
    int npes = 1; // store the number of MPI processes
    double x_j = 0;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &npes );

    unsigned long int n = 40*36*32*28*24*20;
    double h = ( b - a ) / n;
    double h_2 = h/2;
    long unsigned int frac = n/npes;
    double approx = 0;
    unsigned long int j = 0;
   
    for(j = rank*frac; j < (rank+1)*frac; ++j){
        x_j = a + 2 * j * h_2 + h_2;
        approx += 1/(1 + x_j*x_j);
    }

    MPI_Reduce(&approx, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    if(rank==0){
        res = res * 4 * h;
        fprintf( stderr, "Done in %f seconds, pi approx is: %f.\n", stop-start, res);
    }
    MPI_Finalize();

    return 0;
}
