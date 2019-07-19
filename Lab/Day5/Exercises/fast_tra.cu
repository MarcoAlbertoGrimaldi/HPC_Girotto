#include <stdio.h>

#define N 8192
#define BLOCK_COLUMNS 32
#define BLOCK_ROWS 8
#define TILE_DIM 32


__global__ void transposeCoalesced(float *host_out, const float *host_in)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (BLOCK_ROWS < BLOCK_COLUMNS){
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
            tile[threadIdx.y+j][threadIdx.x] = host_in[(y+j)*width + x];
        }
    } else {
        tile[threadIdx.y][threadIdx.x] = host_in[y*width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (BLOCK_ROWS < BLOCK_COLUMNS){
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
            host_out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    } else {
        host_out[y*width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int correct(float* a, float* b)
{   
    int i;
    for(i=0; i<N*N; i++)
        if(a[i]!=b[(i%N)*N + i/N]) return 0;
    return 1;
}

int main()
{

    float * host_in, * host_out, * host_test;
    float * dev_in, * dev_out;
    int size = N*N;
    int mem_syze = size * sizeof(float);
    int i;


    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //allocate memory in host and device
    host_in = (float *)malloc(mem_syze);
    host_out = (float *)malloc(mem_syze);
    host_test = (float *)malloc(mem_syze);

    cudaMalloc((void**)&dev_in, mem_syze);
    cudaMalloc((void**)&dev_out, mem_syze);

    //fill matrix in host
    for(i = 0; i<size; ++i){
        host_in[i] = i;
        host_test[i] = 0;
    }
    
    //transfer matrix from host to device
    cudaMemcpy(dev_in, host_in, mem_syze, cudaMemcpyHostToDevice);


    //transpose matrix in device
    dim3 dimGrid(N/TILE_DIM, N/TILE_DIM, 1);
    dim3 dimBlock(BLOCK_COLUMNS, BLOCK_ROWS, 1);

    cudaEventRecord(start);
    transposeCoalesced<<< dimGrid, dimBlock >>>(dev_out, dev_in);
    cudaEventRecord(stop);

    // transfer matrix from device to host
    cudaMemcpy(host_out, dev_out, mem_syze, cudaMemcpyDeviceToHost);

       // correctness test
    //printf("\ncorrecteness: %d \n", correct(host_in, host_out));
   
    //showing Bandwidth
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            host_test[j*N + i] = host_in[i*N + j];
        }
    }

    bool passed = true;
    for (int i = 0; i < N*N; i++){
        if (host_test[i] !=  host_out[i]) {
            passed = false;
        break;
        }
    }
    if (passed) {printf("Passed. \n");}
    else {printf("Not passed. \n");}

    printf("\nblock: %d x %d", dimBlock.y, dimBlock.x);
    printf("\nmilliseconds: %f", milliseconds);
    printf("\nBandwidth: %f GB/s \n", 2*mem_syze/milliseconds/1e6);

    //free memory   
    free(host_in);
    free(host_out);
    free(host_test);
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}
