#include <stdio.h>
#include <math.h>

#define N (2048*2048)
#define THREAD_PER_BLOCK 512

__global__ void mul( int *a, int *b, int *c) {
        int i = blockIdx.x/4;
        int j = (blockIdx.x%4) * blockDim + threadIdx.x;
        c[i*2048+j]=0;
        for(int k=0; k<N; ++k){
            c[i*2048+j] += a[i*2048+k]*a[k*2048+j];
        }
}

void random_ints(int *p, int n) {
	int i;
	for(i=0; i<n; i++) {
		p[i]=rand();
	}
}

int main( void ) {
    int *a, *b, *c, *d;               // host copies of a, b, c
    int *dev_a, *dev_b, *dev_c;   // device copies of a, b, c
    int size = N * sizeof( int ); // we need space for N   									// integers
    int i;

    // allocate device copies of a, b
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );

    a = (int*)malloc( size ); 
    b = (int*)malloc( size );
    c = (int*)malloc( size );
    d = (int*)malloc( size );

    random_ints( a, N ); 
    random_ints( b, N );

    // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    // launch an rev() kernel with N threads
    rev<<< N/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>( dev_a, dev_b);

    // copy device result back to host copy of c
   cudaMemcpy( b, dev_b, size,   cudaMemcpyDeviceToHost );

    for(i=0; i<N; i++) {
            d[i] += 0;   
    }

    for(i=0; i<2048; i++) {
        for(j=0; j<2048; j++0) {
            for(k=0; k<2048; k++0) {
                d[i*2048+j] += a[i*2048+k]*b[k*2048+j];
                if(b[i]!=d[i]) {
                    printf("error: expected %d, got %d!\n",d[i], c[i]);
                    break;
            }
        }
        if(i==N) {
            printf("correct!\n");
         }  
    }
 
    free( a ); free( b ); free( c ); free( d );
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_d );
    return 0;
}