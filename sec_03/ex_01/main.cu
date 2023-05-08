#include <hip/hip_runtime.h>

__global__ void hello_world() { printf("Hello World!"); }

int main() {
  hello_world<<<1, 1>>>();
  return 0;
}
