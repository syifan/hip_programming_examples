#include <hip/hip_runtime.h>

#include "image.h"

__global__ void image_gamma(uint8_t *d_image, float gamma, int num_values) {
  int global_size = blockDim.x * gridDim.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (; idx < num_values; idx += global_size) {
    float value = d_image[idx] / 255.0f;
    value = pow(value, gamma);
    d_image[idx] = (uint8_t)(value * 255.0f);
  }
}

int main(int argc, char *argv[]) {
  int width, height, channels;
  uint8_t *data = stbi_load("test.jpg", &width, &height, &channels, 0);
  if (!data) {
    printf("Failed to load image\n");
    return 1;
  }

  printf("Width %d, Height %d, Channel %d.\n", width, height, channels);

  int num_pixels = width * height * channels;
  uint8_t *d_image;
  hipMalloc(&d_image, num_pixels * sizeof(uint8_t));
  hipMemcpy(d_image, data, num_pixels * sizeof(uint8_t), hipMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = 2048;

  float gamma = 4.0;

  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);

  hipEventRecord(start, 0);
  image_gamma<<<gridSize, blockSize>>>(d_image, gamma, num_pixels);
  hipEventRecord(end, 0);

  hipEventSynchronize(end);
  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, end);
  printf("Time: %f ms\n", milliseconds);

  hipMemcpy(data, d_image, num_pixels * sizeof(uint8_t), hipMemcpyDeviceToHost);

  // Save the image to test_out.jpg
  stbi_write_jpg("test_out.jpg", width, height, channels, data, 100);

  hipFree(d_image);
  stbi_image_free(data);

  return 0;
}