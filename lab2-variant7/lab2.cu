#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <exception>

void fail(const char* text) {
    fprintf(stderr, "ERROR: %s\n", text);
    exit(0);
}

void cuda_check(cudaError_t err) {
    if (err != cudaSuccess) {
        fail(cudaGetErrorString(err));
    }
}

__device__ float get_gray(const uchar4* image, int x, int y,int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    uchar4 pixel = image[(size_t)y * width + x];
    return 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}

__global__ void sobel_filter(const uchar4* image, uchar4* result,int width, int height) {
    int start_x = blockIdx.x * blockDim.x + threadIdx.x;
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;
    int step_x = blockDim.x * gridDim.x;
    int step_y = blockDim.y * gridDim.y;

    for (int y = start_y; y < height; y += step_y) {
        for (int x = start_x; x < width; x += step_x) {
            float p1 = get_gray(image, x - 1, y - 1, width, height);
            float p2 = get_gray(image, x,     y - 1, width, height);
            float p3 = get_gray(image, x + 1, y - 1, width, height);
            float p4 = get_gray(image, x - 1, y, width, height);
            float p6 = get_gray(image, x + 1, y, width, height);
            float p7 = get_gray(image, x - 1, y + 1, width, height);
            float p8 = get_gray(image, x,     y + 1, width, height);
            float p9 = get_gray(image, x + 1, y + 1, width, height);
            float gx = -p1 + p3 - 2.0f * p4 + 2.0f * p6 - p7 + p9;
            float gy = -p1 - 2.0f * p2 - p3 + p7 + 2.0f * p8 + p9;
            float value = sqrtf(gx * gx + gy * gy);
            if (value > 255.0f) {
                value = 255.0f;
            }
            unsigned char color = (unsigned char)value;
            size_t index = (size_t)y * width + x;
            result[index] = make_uchar4(color, color, color, 0);
        }
    }
}

int program() {
    std::string input_name;
    std::string output_name;
    std::getline(std::cin, input_name);
    std::getline(std::cin, output_name);
    if (input_name.empty() || output_name.empty()) {
        fail("wrong file name");
    }
    std::ifstream input(input_name, std::ios::binary);
    if (!input) {
        fail("cannot open input file");
    }
    int width;
    int height;
    input.read((char*)&width, sizeof(int));
    input.read((char*)&height, sizeof(int));
    if (!input || width <= 0 || height <= 0) {
        fail("wrong image size");
    }

    size_t pixel_count = (size_t)width * height;
    size_t data_size = pixel_count * sizeof(uchar4);
    std::vector<uchar4> image(pixel_count);

    input.read((char*)image.data(), data_size);
    if (!input) {
        fail("cannot read image");
    }
    input.close();

    uchar4* gpu_image;
    cuda_check(cudaMalloc(&gpu_image, data_size));
    cuda_check(cudaMemcpy(gpu_image, image.data(), data_size, cudaMemcpyHostToDevice));
    uchar4* gpu_result;
    cuda_check(cudaMalloc(&gpu_result, data_size));
    dim3 blocks(32, 32);
    dim3 threads(16, 16);
    sobel_filter<<<blocks, threads>>>(gpu_image, gpu_result, width, height);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpy(image.data(), gpu_result, data_size, cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(gpu_result));
    cuda_check(cudaFree(gpu_image));

    std::ofstream output(output_name, std::ios::binary);
    if (!output) {
        fail("cannot open output file");
    }
    output.write((char*)&width, sizeof(int));
    output.write((char*)&height, sizeof(int));
    output.write((char*)image.data(), data_size);
    output.close();
    if (!output) {
        fail("cannot write output image");
    }
    return 0;
}

int main() {
    try {
        return program();
    }
    catch (const std::exception& e) {
        fail(e.what());
    }
    catch (...) {
        fail("unknown error");
    }
    return 0;
}
