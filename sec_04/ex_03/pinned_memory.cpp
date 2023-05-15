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
  hipHostMalloc(&a, sizeof(float) * N);
  hipHostMalloc(&b, sizeof(float) * N);
  hipHostMalloc(&c, sizeof(float) * N);

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

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);

  // Transfer data back to host memory
  hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

  printf("The first index of the resulting array, c[0], is %f\n", c[0]);

  // Deallocate device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  // Deallocate host memory
  hipHostFree(a);
  hipHostFree(b);
  hipHostFree(c);

  return 0;
}
