#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>

void print_usage( int * a, int N, int nthreads ) {

  int tid, i;
  for( tid = 0; tid < nthreads; ++tid ) {

    fprintf( stdout, "%d: ", tid );

    for( i = 0; i < N; ++i ) {

      if( a[ i ] == tid) fprintf( stdout, "*" );
      else fprintf( stdout, " ");
    }
    printf("\n");
  }
}

int main() {

    const int N = 120;
    int a[N];
    int nthreads;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp single
	    {
	        nthreads = omp_get_num_threads();
	    }
        //Static

        #pragma omp for schedule(static)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);    
        
        #pragma omp for schedule(static,1)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);

        #pragma omp for schedule(static,10)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);

        //Dynamic

        #pragma omp for schedule(dynamic)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);    

        #pragma omp for schedule(dynamic,1)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);

        #pragma omp for schedule(dynamic,10)
            for(int i = 0; i < N; ++i) {
                a[i] = thread_id;
            }

        #pragma omp single
        print_usage(a, N, nthreads);
    }                                                                                                                                                                                                                                                                         

    return 0;
    }
