
#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
	// TODO
	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM]; 
	int row= blockIdx.y*OUT_TILE_DIM + threadIdx.y;
	int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
	if(outRow<height && outCol<width){
		float sum= 0.0f;
	}
	__syncthreads();














}

void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memoryi
	
    // TODO
    cudaMemcpyToSymbol(filter_c,filter,FILTER_DIM*FILTER_DIM*sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    // TODO
	dim3 numThreadsPerBlock(IN_TILE_DIM,IN_TILE_DIM);
	dim3 numBlocks((width + IN_TILE_DIM - 1) / IN_TILE_DIM ,(height + IN_TILE_DIM - 1) / IN_TILE_DIM  );
	convoltuion_tiled_kernel<<< numBlocks, numThreadsPerBlock >>>(input_d, output_d, width, height);

}

