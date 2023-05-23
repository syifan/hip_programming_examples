#ifndef IMAGE_H
#define IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

uint8_t *read_image(char *filename, int *width, int *height, int *channels) {
  uint8_t *data = stbi_load(filename, width, height, channels, 0);

  if (data) {
    printf("%s: %d x %d\n", filename, *width, *height);
    return data;
  } else {
    printf("%s: failed to load\n", filename);
    return NULL;
  }
}

#endif  // IMAGE_H
