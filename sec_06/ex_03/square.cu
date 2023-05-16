#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_runtime.h>

// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = a[idx] * a[idx];
}

// main routine that executes on the host
int main(void)
{
	float *a_h, *a_d;  // Pointer to host & device arrays
	const int N = 10;  // Number of elements in arrays
	size_t size = N * sizeof(float);
	a_h = (float *)malloc(size);        // Allocate array on host
	hipMalloc((void **) &a_d, size);   // Allocate array on device
	// Initialize host array and copy it to CUDA device
	for (int i=0; i<N; i++) a_h[i] = (float)i;
	hipMemcpy(a_d, a_h, size, hipMemcpyHostToDevice);
	// Do calculation on device:
	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	hipLaunchKernelGGL(square_array, n_blocks, block_size , 0, 0, a_d, N);
	// Retrieve result from device and store it in host array
	hipMemcpy(a_h, a_d, sizeof(float)*N, hipMemcpyDeviceToHost);
	// Print results
	for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
	// Cleanup
	free(a_h); hipFree(a_d);
}