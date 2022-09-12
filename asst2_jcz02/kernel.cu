
#include "common.h"
#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if( col < K && row < M) {
        for(int i = 0; i < N; i++) {
            sum += A[row*N+i] *B[i*K+col];
        }
        C[row * K + col] = sum;
    }

}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*M*sizeof(float));
    cudaMalloc((void**) &B_d, N*M*sizeof(float));
    cudaMalloc((void**) &C_d, N*M*sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    
    // TODO
    cudaMemcpy(A_d, A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*M*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (M + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    mm_kernel <<< numBlocks, numThreadsPerBlock >>> (A_d, B_d, C_d, N, M, K);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    

    // TODO
    cudaMemcpy(C_d, C, N*M*sizeof(float), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

