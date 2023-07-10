#include <hip/hip_runtime.h>
#include <roctracer_ext.h>
#include <roctx.h>

#include <iostream>

#define WIDTH 1024
#define NUM (WIDTH * WIDTH)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4

#define THREADS_PER_BLOCK_Z 1

__global__ void matrixTranspose(float* out, float* in, const int width) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  out[y * width + x] = in[x * width + y];
}

int main() {
  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;
  float* gpuMatrix;
  float* gpuTransposeMatrix;
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << "Device name " << devProp.name << std::endl;
  int i;
  int errors;
  Matrix = (float*)malloc(NUM * sizeof(float));
  TransposeMatrix = (float*)malloc(NUM * sizeof(float));
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    Matrix[i] = (float)i * 10.0f;
  }
  // allocate the memory on the device side
  hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
  hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
  uint32_t iterations = 100;
  while (iterations-- > 0) {
    std::cout << "## Iteration (" << iterations << ")#################"
              << std::endl;
    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    roctxMark("before hipLaunchKernel");
    int rangeId = roctxRangeStart("hipLaunchKernel range");
    roctxRangePush("hipLaunchKernel");

    // Lauching kernel from host
    hipLaunchKernelGGL(
        matrixTranspose,
        dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
        gpuTransposeMatrix, gpuMatrix, WIDTH);

    roctxMark("after hipLaunchKernel");

    // Memory transfer from device to host
    roctxRangePush("hipMemcpy");
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),
              hipMemcpyDeviceToHost);
    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"
    roctxRangeStop(rangeId);
  }

  // free the resources on device side
  hipFree(gpuMatrix);
  hipFree(gpuTransposeMatrix);
  // free the resources on host side
  free(Matrix);
  free(TransposeMatrix);

  return errors;
}
