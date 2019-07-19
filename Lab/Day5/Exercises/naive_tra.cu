#include <stdio.h>

#define N 2048*4
#define BLOCK_COLUMNS 8
#define BLOCK_ROWS 8

__global__ void transpose_naive(float *dev_out, const float *dev_in)
{
    int x = blockIdx.x * BLOCK_COLUMNS + threadIdx.x;
    int y = blockIdx.y * BLOCK_ROWS + threadIdx.y;

    dev_out[x*N + (y)] = dev_in[(y)*N + x];
}

int main(){

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
    dim3 dimGrid(N/BLOCK_COLUMNS, N/BLOCK_ROWS, 1);
    dim3 dimBlock(BLOCK_COLUMNS, BLOCK_ROWS, 1);

    cudaEventRecord(start);
    transpose_naive<<< dimGrid, dimBlock >>>(dev_out, dev_in);
    cudaEventRecord(stop);

    // transfer matrix from device to host
    cudaMemcpy(host_out, dev_out, mem_syze, cudaMemcpyDeviceToHost);

       // correctness test
    //printf("\ncorrecteness: %d \n", correct(host_in, host_out));
   
    //showing BandN
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
    printf("\nBandN: %f GB/s \n", 2*mem_syze/milliseconds/1e6);

    //free memory   
    free(host_in);
    free(host_out);
    free(host_test);
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}