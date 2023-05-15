#include <hip/hip_runtime.h>

#define N 1000000000

struct TwoDimensionPoint {
    float x;
    float y;
    float distToOrigin;
};

__global__ void calculateDistToOrigin(TwoDimensionPoint *points, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        points[idx].distToOrigin = sqrt(points[idx].x * points[idx].x + points[idx].y * points[idx].y);
    }
}

int main() {
    TwoDimensionPoint *h_points;
    h_points  = (TwoDimensionPoint*)malloc(sizeof(TwoDimensionPoint) * N);

    // Randomly initialize x and y values
    for (int i = 0; i < N; i++) {
        h_points[i].x = rand() % RAND_MAX;
        h_points[i].y = rand() % RAND_MAX;        
    }

    TwoDimensionPoint *points;
    hipMalloc((void**)&points, N * sizeof(TwoDimensionPoint));

    // Transfer data from host to device memory
    hipMemcpy(points, h_points, sizeof(TwoDimensionPoint) * N, hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Kernel launch
    calculateDistToOrigin<<<gridSize, blockSize>>>(points, N);
    hipDeviceSynchronize();

    // Transfer data back to host memory
    hipMemcpy(h_points, points, sizeof(TwoDimensionPoint) * N, hipMemcpyDeviceToHost);

    // Print the calculated distToOrigin values
    for (int i = 0; i < N; i++) {
        printf("Point %d: (%f, %f), distToOrigin = %f\n", i, h_points[i].x, h_points[i].y, h_points[i].distToOrigin); 
    }

    // Deallocate device memory
    hipFree(points);

    // Deallocate host memory
    free(h_points);

    return 0;
}