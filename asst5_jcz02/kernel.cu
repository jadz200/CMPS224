
#include "common.h"
#include "timer.h"

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__  char  b_s[NUM_BINS];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<NUM_BINS){
        b_s[threadIdx.x]=0;
    }
    __syncthreads();

    if(i < width*height) {
        unsigned char b = b_s[i];
        atomicAdd(&bins[b], 1);
    }
    __syncthreads();
    if(i < width*height && i<NUM_BINS) {
        unsigned char b = image[i];
        atomicAdd(&bins[b], b_s[b]);
    }
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numThreadsPerBlock=1024;
    unsigned int numBlocks=(width*heigh+numThreadsPerBlock-1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,width,height);
}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO


}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numThreadsPerBlock=512;
    unsigned int numBlocks=(width*heigh+numThreadsPerBlock-1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,width,height);
    
}

