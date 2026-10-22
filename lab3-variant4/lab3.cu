#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

__constant__ double avg[32][3];

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(0);
    }
}

__global__ void classify(uchar4* image, int size, int nc) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (index < size) {
        uchar4 pixel = image[index];
        int best_class = 0;
        double best = pixel.x * avg[0][0] + pixel.y * avg[0][1] + pixel.z * avg[0][2];
        for (int i = 1; i < nc; i++) {
            double value = pixel.x * avg[i][0] + pixel.y * avg[i][1] + pixel.z * avg[i][2];
            if (value > best) {
                best = value;
                best_class = i;
            }
        }
        image[index].w = best_class;
        index += offset;
    }
}

int main() {
    std::string input_name, output_name;
    int nc;
    std::cin >> input_name >> output_name >> nc;
    std::ifstream input(input_name, std::ios::binary);
    int width, height;
    input.read((char*)&width, sizeof(int));
    input.read((char*)&height, sizeof(int));
    int size = width * height;
    std::vector<uchar4> image(size);
    input.read((char*)image.data(), size * sizeof(uchar4));
    input.close();

    double host_avg[32][3] = {};
    for (int i = 0; i < nc; i++) {
        int np;
        std::cin >> np;
        for (int j = 0; j < np; j++) {
            int x, y;
            std::cin >> x >> y;
            uchar4 pixel = image[y * width + x];
            host_avg[i][0] += pixel.x;
            host_avg[i][1] += pixel.y;
            host_avg[i][2] += pixel.z;
        }
        for (int j = 0; j < 3; j++) {
            host_avg[i][j] /= np;
        }
        double length = sqrt(host_avg[i][0] * host_avg[i][0] + host_avg[i][1] * host_avg[i][1] + host_avg[i][2] * host_avg[i][2]);
        for (int j = 0; j < 3; j++) {
            host_avg[i][j] /= length;
        }
    }

    check(cudaMemcpyToSymbol(avg, host_avg, nc * 3 * sizeof(double)));
    uchar4* device_image;
    check(cudaMalloc(&device_image, size * sizeof(uchar4)));
    check(cudaMemcpy(device_image, image.data(), size * sizeof(uchar4),cudaMemcpyHostToDevice));
    classify<<<256, 256>>>(device_image, size, nc);
    check(cudaGetLastError());
    check(cudaMemcpy(image.data(), device_image, size * sizeof(uchar4), cudaMemcpyDeviceToHost));
    check(cudaFree(device_image));
    std::ofstream output(output_name, std::ios::binary);
    output.write((char*)&width, sizeof(int));
    output.write((char*)&height, sizeof(int));
    output.write((char*)image.data(), size * sizeof(uchar4));
    output.close();
    return 0;
}
