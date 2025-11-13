#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ void colortoGrayscaleConvertionKernel(
    unsigned char * Pout,
    unsigned char * Pin,
    int width,
    int height
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void colortoGrayscaleConvertion(unsigned char *h_in, unsigned char *h_out, int width, int height) {
    unsigned char *d_in, *d_out;
    size_t rgbSize = width * height * CHANNELS * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    cudaMalloc((void**)&d_in, rgbSize);
    cudaMalloc((void**)&d_out, graySize);

    cudaMemcpy(d_in, h_in, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    colortoGrayscaleConvertionKernel<<<gridSize, blockSize>>>(d_out, d_in, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, graySize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    int width = 8;
    int height = 4;

    size_t colorBytes = width * height * CHANNELS;
    size_t grayBytes  = width * height;

    unsigned char *h_in  = (unsigned char*)malloc(colorBytes);
    unsigned char *h_out = (unsigned char*)malloc(grayBytes);

    for (int i = 0; i < width * height; i++) {
        h_in[i * CHANNELS + 0] = (unsigned char)(i * 3 % 256);   // R
        h_in[i * CHANNELS + 1] = (unsigned char)(i * 5 % 256);   // G
        h_in[i * CHANNELS + 2] = (unsigned char)(i * 7 % 256);   // B
    }

    colortoGrayscaleConvertion(h_in, h_out, width, height);

    printf("Original (%dx%d):\n", height, width);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int idx = (r * width + c) * CHANNELS;
            printf("(%3d,%3d,%3d) ", h_in[idx], h_in[idx+1], h_in[idx+2]);
        }
        printf("\n");
    }

    printf("Grayscale (%dx%d):\n", height, width);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%3d ", h_out[r * width + c]);
        }
        printf("\n");
    }

    free(h_in);
    free(h_out);
    return 0;
}