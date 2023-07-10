#include <hip/hip_runtime.h>
#include <time.h>

#define N 1000000000
#define LDS_SIZE 416

__global__ void vector_add(float *c, float *a, float *b, int n) {
  __shared__ volatile float s_a[LDS_SIZE];

  int global_size = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = tid; i < n; i += global_size) {
    c[i] = a[i] + b[i];
  }

  if (a[tid] == 0.0) {
    s_a[threadIdx.x] = a[tid];
    a[tid] += 1.0;
    a[tid] = s_a[threadIdx.x];
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

  // Transfer data from host to device memory
  hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice);

  int blockSize = 64;
  int gridSize = 4800;

  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);

  // Executing kernel
  hipEventRecord(start, 0);
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);
  hipEventRecord(end, 0);

  hipEventSynchronize(end);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, end);
  printf("Time: %f ms\n", milliseconds);

  // Transfer data back to host memory
  hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

  printf("The first index of the resulting array, c[0] = %f\n", c[0]);

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
