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

  // Allocate managed memory
  hipMallocManaged(&a, sizeof(float) * N);
  hipMallocManaged(&b, sizeof(float) * N);
  hipMallocManaged(&c, sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(c, a, b, N);

  hipDeviceSynchronize();

  printf("The first index of the resulting array, c[0], is %f\n", c[0]);

  // Deallocate device memory
  hipFree(a);
  hipFree(b);
  hipFree(c);

  return 0;
}
