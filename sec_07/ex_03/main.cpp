#include <hip/hip_runtime.h>

#include "image.h"

#define UNROLL 64

__global__ void image_gamma(uint8_t *d_image, float gamma, int num_values) {
  int global_size = blockDim.x * gridDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  for (; i * UNROLL < num_values; i += global_size) {
    int idx = i * UNROLL;

    float value0 = d_image[idx + 0] / 255.0f;
    float value1 = d_image[idx + 1] / 255.0f;
    float value2 = d_image[idx + 2] / 255.0f;
    float value3 = d_image[idx + 3] / 255.0f;
    float value4 = d_image[idx + 4] / 255.0f;
    float value5 = d_image[idx + 5] / 255.0f;
    float value6 = d_image[idx + 6] / 255.0f;
    float value7 = d_image[idx + 7] / 255.0f;
    float value8 = d_image[idx + 8] / 255.0f;
    float value9 = d_image[idx + 9] / 255.0f;
    float value10 = d_image[idx + 10] / 255.0f;
    float value11 = d_image[idx + 11] / 255.0f;
    float value12 = d_image[idx + 12] / 255.0f;
    float value13 = d_image[idx + 13] / 255.0f;
    float value14 = d_image[idx + 14] / 255.0f;
    float value15 = d_image[idx + 15] / 255.0f;
    float value16 = d_image[idx + 16] / 255.0f;
    float value17 = d_image[idx + 17] / 255.0f;
    float value18 = d_image[idx + 18] / 255.0f;
    float value19 = d_image[idx + 19] / 255.0f;
    float value20 = d_image[idx + 20] / 255.0f;
    float value21 = d_image[idx + 21] / 255.0f;
    float value22 = d_image[idx + 22] / 255.0f;
    float value23 = d_image[idx + 23] / 255.0f;
    float value24 = d_image[idx + 24] / 255.0f;
    float value25 = d_image[idx + 25] / 255.0f;
    float value26 = d_image[idx + 26] / 255.0f;
    float value27 = d_image[idx + 27] / 255.0f;
    float value28 = d_image[idx + 28] / 255.0f;
    float value29 = d_image[idx + 29] / 255.0f;
    float value30 = d_image[idx + 30] / 255.0f;
    float value31 = d_image[idx + 31] / 255.0f;
    float value32 = d_image[idx + 32] / 255.0f;
    float value33 = d_image[idx + 33] / 255.0f;
    float value34 = d_image[idx + 34] / 255.0f;
    float value35 = d_image[idx + 35] / 255.0f;
    float value36 = d_image[idx + 36] / 255.0f;
    float value37 = d_image[idx + 37] / 255.0f;
    float value38 = d_image[idx + 38] / 255.0f;
    float value39 = d_image[idx + 39] / 255.0f;
    float value40 = d_image[idx + 40] / 255.0f;
    float value41 = d_image[idx + 41] / 255.0f;
    float value42 = d_image[idx + 42] / 255.0f;
    float value43 = d_image[idx + 43] / 255.0f;
    float value44 = d_image[idx + 44] / 255.0f;
    float value45 = d_image[idx + 45] / 255.0f;
    float value46 = d_image[idx + 46] / 255.0f;
    float value47 = d_image[idx + 47] / 255.0f;
    float value48 = d_image[idx + 48] / 255.0f;
    float value49 = d_image[idx + 49] / 255.0f;
    float value50 = d_image[idx + 50] / 255.0f;
    float value51 = d_image[idx + 51] / 255.0f;
    float value52 = d_image[idx + 52] / 255.0f;
    float value53 = d_image[idx + 53] / 255.0f;
    float value54 = d_image[idx + 54] / 255.0f;
    float value55 = d_image[idx + 55] / 255.0f;
    float value56 = d_image[idx + 56] / 255.0f;
    float value57 = d_image[idx + 57] / 255.0f;
    float value58 = d_image[idx + 58] / 255.0f;
    float value59 = d_image[idx + 59] / 255.0f;
    float value60 = d_image[idx + 60] / 255.0f;
    float value61 = d_image[idx + 61] / 255.0f;
    float value62 = d_image[idx + 62] / 255.0f;
    float value63 = d_image[idx + 63] / 255.0f;

    value0 = pow(value0, gamma);
    value1 = pow(value1, gamma);
    value2 = pow(value2, gamma);
    value3 = pow(value3, gamma);
    value4 = pow(value4, gamma);
    value5 = pow(value5, gamma);
    value6 = pow(value6, gamma);
    value7 = pow(value7, gamma);
    value8 = pow(value8, gamma);
    value9 = pow(value9, gamma);
    value10 = pow(value10, gamma);
    value11 = pow(value11, gamma);
    value12 = pow(value12, gamma);
    value13 = pow(value13, gamma);
    value14 = pow(value14, gamma);
    value15 = pow(value15, gamma);
    value16 = pow(value16, gamma);
    value17 = pow(value17, gamma);
    value18 = pow(value18, gamma);
    value19 = pow(value19, gamma);
    value20 = pow(value20, gamma);
    value21 = pow(value21, gamma);
    value22 = pow(value22, gamma);
    value23 = pow(value23, gamma);
    value24 = pow(value24, gamma);
    value25 = pow(value25, gamma);
    value26 = pow(value26, gamma);
    value27 = pow(value27, gamma);
    value28 = pow(value28, gamma);
    value29 = pow(value29, gamma);
    value30 = pow(value30, gamma);
    value31 = pow(value31, gamma);
    value32 = pow(value32, gamma);
    value33 = pow(value33, gamma);
    value34 = pow(value34, gamma);
    value35 = pow(value35, gamma);
    value36 = pow(value36, gamma);
    value37 = pow(value37, gamma);
    value38 = pow(value38, gamma);
    value39 = pow(value39, gamma);
    value40 = pow(value40, gamma);
    value41 = pow(value41, gamma);
    value42 = pow(value42, gamma);
    value43 = pow(value43, gamma);
    value44 = pow(value44, gamma);
    value45 = pow(value45, gamma);
    value46 = pow(value46, gamma);
    value47 = pow(value47, gamma);
    value48 = pow(value48, gamma);
    value49 = pow(value49, gamma);
    value50 = pow(value50, gamma);
    value51 = pow(value51, gamma);
    value52 = pow(value52, gamma);
    value53 = pow(value53, gamma);
    value54 = pow(value54, gamma);
    value55 = pow(value55, gamma);
    value56 = pow(value56, gamma);
    value57 = pow(value57, gamma);
    value58 = pow(value58, gamma);
    value59 = pow(value59, gamma);
    value60 = pow(value60, gamma);
    value61 = pow(value61, gamma);
    value62 = pow(value62, gamma);
    value63 = pow(value63, gamma);

    d_image[idx + 0] = (uint8_t)(value0 * 255.0f);
    d_image[idx + 1] = (uint8_t)(value1 * 255.0f);
    d_image[idx + 2] = (uint8_t)(value2 * 255.0f);
    d_image[idx + 3] = (uint8_t)(value3 * 255.0f);
    d_image[idx + 4] = (uint8_t)(value4 * 255.0f);
    d_image[idx + 5] = (uint8_t)(value5 * 255.0f);
    d_image[idx + 6] = (uint8_t)(value6 * 255.0f);
    d_image[idx + 7] = (uint8_t)(value7 * 255.0f);
    d_image[idx + 8] = (uint8_t)(value8 * 255.0f);
    d_image[idx + 9] = (uint8_t)(value9 * 255.0f);
    d_image[idx + 10] = (uint8_t)(value10 * 255.0f);
    d_image[idx + 11] = (uint8_t)(value11 * 255.0f);
    d_image[idx + 12] = (uint8_t)(value12 * 255.0f);
    d_image[idx + 13] = (uint8_t)(value13 * 255.0f);
    d_image[idx + 14] = (uint8_t)(value14 * 255.0f);
    d_image[idx + 15] = (uint8_t)(value15 * 255.0f);
    d_image[idx + 16] = (uint8_t)(value16 * 255.0f);
    d_image[idx + 17] = (uint8_t)(value17 * 255.0f);
    d_image[idx + 18] = (uint8_t)(value18 * 255.0f);
    d_image[idx + 19] = (uint8_t)(value19 * 255.0f);
    d_image[idx + 20] = (uint8_t)(value20 * 255.0f);
    d_image[idx + 21] = (uint8_t)(value21 * 255.0f);
    d_image[idx + 22] = (uint8_t)(value22 * 255.0f);
    d_image[idx + 23] = (uint8_t)(value23 * 255.0f);
    d_image[idx + 24] = (uint8_t)(value24 * 255.0f);
    d_image[idx + 25] = (uint8_t)(value25 * 255.0f);
    d_image[idx + 26] = (uint8_t)(value26 * 255.0f);
    d_image[idx + 27] = (uint8_t)(value27 * 255.0f);
    d_image[idx + 28] = (uint8_t)(value28 * 255.0f);
    d_image[idx + 29] = (uint8_t)(value29 * 255.0f);
    d_image[idx + 30] = (uint8_t)(value30 * 255.0f);
    d_image[idx + 31] = (uint8_t)(value31 * 255.0f);
    d_image[idx + 32] = (uint8_t)(value32 * 255.0f);
    d_image[idx + 33] = (uint8_t)(value33 * 255.0f);
    d_image[idx + 34] = (uint8_t)(value34 * 255.0f);
    d_image[idx + 35] = (uint8_t)(value35 * 255.0f);
    d_image[idx + 36] = (uint8_t)(value36 * 255.0f);
    d_image[idx + 37] = (uint8_t)(value37 * 255.0f);
    d_image[idx + 38] = (uint8_t)(value38 * 255.0f);
    d_image[idx + 39] = (uint8_t)(value39 * 255.0f);
    d_image[idx + 40] = (uint8_t)(value40 * 255.0f);
    d_image[idx + 41] = (uint8_t)(value41 * 255.0f);
    d_image[idx + 42] = (uint8_t)(value42 * 255.0f);
    d_image[idx + 43] = (uint8_t)(value43 * 255.0f);
    d_image[idx + 44] = (uint8_t)(value44 * 255.0f);
    d_image[idx + 45] = (uint8_t)(value45 * 255.0f);
    d_image[idx + 46] = (uint8_t)(value46 * 255.0f);
    d_image[idx + 47] = (uint8_t)(value47 * 255.0f);
    d_image[idx + 48] = (uint8_t)(value48 * 255.0f);
    d_image[idx + 49] = (uint8_t)(value49 * 255.0f);
    d_image[idx + 50] = (uint8_t)(value50 * 255.0f);
    d_image[idx + 51] = (uint8_t)(value51 * 255.0f);
    d_image[idx + 52] = (uint8_t)(value52 * 255.0f);
    d_image[idx + 53] = (uint8_t)(value53 * 255.0f);
    d_image[idx + 54] = (uint8_t)(value54 * 255.0f);
    d_image[idx + 55] = (uint8_t)(value55 * 255.0f);
    d_image[idx + 56] = (uint8_t)(value56 * 255.0f);
    d_image[idx + 57] = (uint8_t)(value57 * 255.0f);
    d_image[idx + 58] = (uint8_t)(value58 * 255.0f);
    d_image[idx + 59] = (uint8_t)(value59 * 255.0f);
    d_image[idx + 60] = (uint8_t)(value60 * 255.0f);
    d_image[idx + 61] = (uint8_t)(value61 * 255.0f);
    d_image[idx + 62] = (uint8_t)(value62 * 255.0f);
    d_image[idx + 63] = (uint8_t)(value63 * 255.0f);
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
  num_pixels = (((num_pixels - 1) / UNROLL) + 1) * UNROLL;
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