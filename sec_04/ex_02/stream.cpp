#include <hip/hip_runtime.h>

#define N 100000000

__global__ void squareKernel(float *input, float *output){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    output[tid] = input[tid] * input[tid]; 
}

__global__ void cubeKernel(float *input, float *output){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    output[tid] = input[tid] * input[tid] * input[tid]; 
}

int main(){
    float *h_input, *h_output_sq, *h_output_cube;
    float *d_input, *d_output_sq, *d_output_cube;

    // Allocate host memory
    h_input = (float*)malloc(sizeof(float) * N);
    h_output_sq = (float*)malloc(sizeof(float) * N);
    h_output_cube = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        h_input[i] = rand() % RAND_MAX;
    }

    // Allocate device memory
    hipMalloc((void**)&d_input, sizeof(float) * N);
    hipMalloc((void**)&d_output_sq, sizeof(float) * N);
    hipMalloc((void**)&d_output_cube, sizeof(float) * N);

    // Create two streams
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    // Transfer data from host to device memory
    hipMemcpyAsync(d_input, h_input, sizeof(float) * N, hipMemcpyHostToDevice, stream1);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Executing square kernel 
    squareKernel<<<gridSize, blockSize, 0, stream1>>>(d_input, d_output_sq);

    // Transfer data back to host memory
    hipMemcpyAsync(h_output_sq, d_output_sq, 0, hipMemcpyDeviceToHost, stream1);

    // Executing cube kernel
    cubeKernel<<<gridSize, blockSize, 0, stream2>>>(d_input, d_output_cube);

    // Transfer data back to host memory
    hipMemcpyAsync(h_output_cube, d_output_cube, 0, hipMemcpyDeviceToHost, stream2);

    // Wait for all work in streams to complete
    hipStreamSynchronize(stream1);
    hipStreamSynchronize(stream2);

    hipFree(d_input);
    hipFree(d_output_sq);
    hipFree(d_output_cube);

    free(h_input);
    free(h_output_sq);
    free(h_output_cube);

    return 0;
}