
#include "common.h"
#include "timer.h"

__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {

    // TODO
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < M) {
        double aval = a[i];
        double bval = b[i];
        c[i] = (aval > bval)?aval:bval;
    }

}

void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    // TODO
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**) &x_d, M*sizeof(float));
    cudaMalloc((void**) &y_d, M*sizeof(float));
    cudaMalloc((void**) &z_d, M*sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(x_d, x, M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, M*sizeof(float), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    // TODO
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = N/numThreadsPerBlock;
    vecMax_gpu <<< numBlocks, numThreadsPerBlock >>> (x_d, y_d, z_d, N);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(z, z_d, M*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree((void**) &x_d);
    cudaFree((void**) &y_d);
    cudaFree((void**) &z_d);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

