#include <stdio.h>
#include <math.h>

#define N (2048*2048)
#define THREAD_PER_BLOCK 256
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNaive(int *odata, const int *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

void random_ints(int *p, int n) {
	int i;
	for(i=0; i<n; i++) {
		p[i]=rand();
	}
}

__global__ void transposeCoalesced(int *odata, const int *idata)
{
  __shared__ int tile[TILE_DIM+1][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main( void ) {
    int *a, *b, *c;               // host copies of a, b, c
    int *dev_a, *dev_b;   // device copies of a, b, c
    int size = N * sizeof( int ); // we need space for N   									// integers
    int i, j;

    // allocate device copies of a, b
    cudaMalloc( (void**)&dev_a, size );
    cudaMalloc( (void**)&dev_b, size );

    a = (int*)malloc( size ); 
    b = (int*)malloc( size );
    c = (int*)malloc( size );

    random_ints( a, N ); 
    random_ints( b, N );
    // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

    // launch an rev() kernel with N threads
    tra<<< N/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>( dev_a, dev_b);


    // copy device result back to host copy of c
   cudaMemcpy( b, dev_b, size,   cudaMemcpyDeviceToHost );

    for(i=0; i<2048; i++) {
        for(j=0; j<2048; j++) {
            c[i*2048+j] = a[j*2048+i];
            if(b[i*2048+j]!=c[i*2048+j]) {
                printf("error: expected %d, got %d!\n",c[i*2048+j], b[i*2048+j]);
                break;
            }
        }
    }

    if(i==N) {printf("correct!\n");}
 
    free( a ); free( b ); free( c );
    cudaFree( dev_a );
    cudaFree( dev_b );
    return 0;
}