#include <hip/hip_runtime.h>

#define N 100000000

__global__ void vector_add(float *c, float *a, float *b, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;

  // Allocate host memory
  a = (float *)malloc(sizeof(float) * N);
  b = (float *)malloc(sizeof(float) * N);
  c = (float *)malloc(sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Allocate device memory
  hipMalloc((void **)&d_a, sizeof(float) * N);
  hipMalloc((void **)&d_b, sizeof(float) * N);
  hipMalloc((void **)&d_c, sizeof(float) * N);

  float gpu_elapsed_time_ms;

  // Some events to count the execution time
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Transfer data from host to device memory
  hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Start to count execution time of GPU version
  hipEventRecord(start, 0);

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);

  // Time counting terminate
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  // Transfer data back to host memory
  hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

  // Compute time elapse on GPU computing
  hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on vector addition on GPU: %f ms.\n\n",
         gpu_elapsed_time_ms);

  // Deallocate device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  // Deallocate host memory
  free(a);
  free(b);
  free(c);

  return 0;
}
