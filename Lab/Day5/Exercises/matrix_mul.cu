#include <stdio.h>

#define N 2048
#define N2 N*N
#define BLOCK_SIZE 32

__global__ void matrix_mult( const int *dev_a, const int *dev_b, int *dev_c) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp_sum = 0;

    for(int k = 0; k < N; ++k){
        tmp_sum += dev_a[row * N + k] * dev_b[k * N + col];
    }
    
    dev_c[row * N + col] = tmp_sum;
}

void random_ints(int *p, int n) {
	int i;
	for(i=0; i<n; i++) {
		p[i]=rand();
	}
}

int main() {
    int *host_a, *host_b, *host_c, *host_d;     // host copies of host_a, host_b, host_c
    int *dev_a, *dev_b, *dev_c;                 // device copies of dev_a, dev_b, dev_c
    int size = N2 * sizeof( int );              // we need space for N   								
    int i, j, k;

    // allocate device copies of host_a, host_b, host_c
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );
    cudaMalloc( (void**)&dev_c, size );

    host_a = (int*)malloc( size ); 
    host_b = (int*)malloc( size );
    host_c = (int*)malloc( size );
    host_d = (int*)malloc( size );

    random_ints( host_a, N2 ); 
    random_ints( host_b, N2 );

    dim3 Block_Dim (BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 Grid_Dim (N/BLOCK_SIZE, N/BLOCK_SIZE, 1);

    // copy inputs to device
    cudaMemcpy( dev_a, host_a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, host_b, size, cudaMemcpyHostToDevice );

    // launch an rev() kernel with N threads
    matrix_mult<<< Grid_Dim, Block_Dim >>>(dev_a, dev_b, dev_c);

    // copy device result back to host copy of host_c
    cudaMemcpy( host_c, dev_c, size, cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();

    int sum;
    int errors = 0;

    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            sum = 0;
            for(k=0; k<N; k++) {
                sum += host_a[i*N+k]*host_b[k*N+j];
            }
    
            host_d[i*N+j] = sum;

            if(host_c[i*N+j] != host_d[i*N+j]) {
                printf(" %i \n", host_c[i*N+j]);
                printf(" %i \n", host_d[i*N+j]);
                errors += 1;
                break;
            }
        }
    }  

    if(errors==0) printf("%i errors: correct! \n", errors);
    else printf("%i errors: not correct! \n", errors);
 
    free( host_a ); free( host_b ); free( host_c ); free( host_d );
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );
    return 0;
}