#include <hip/hip_runtime.h>

__global__ void gpuHello() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", tid);
}

int main(){
    gpuHello<<<4, 4>>>();
    hipDeviceSynchronize();

    return 0;
}