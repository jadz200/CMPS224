
#include "common.h"
#include "timer.h"
#define COARSE_FACTOR 4
__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__  int  b_s[NUM_BINS];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x<NUM_BINS){
	b_s[threadIdx.x]=0;
    }
    __syncthreads();
    
    if(i< width*height) {
        unsigned char b = image[i];
        atomicAdd(&b_s[b], 1);
    }
    __syncthreads();
    if(threadIdx.x<NUM_BINS) {
        atomicAdd(&bins[threadIdx.x], b_s[threadIdx.x]);
    }

}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock=1024;
    unsigned int numBlocks=(width*height+numThreadsPerBlock-1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,width,height);
}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__  int  b_s[NUM_BINS];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x<NUM_BINS){
	b_s[threadIdx.x]=0;
    }
    __syncthreads();
    for(unsigned int c=0;c<COARSE_FACTOR;c++){
	if(i+blockDim.x*c<width*height){
		unsigned char b=image[i+blockDim.x*c];
		atomicAdd(&(bins[b]),1);
	}
    }
    __syncthreads();
   if(threadIdx.x< NUM_BINS && b_s[threadIdx.x]>0){
	atomicAdd(&(bins[threadIdx.x]),b_s[threadIdx.x]);
   }
	
}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock=1024;
    unsigned int numBlocks=(width*height+numThreadsPerBlock-1)/numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d,width,height);
    
}

