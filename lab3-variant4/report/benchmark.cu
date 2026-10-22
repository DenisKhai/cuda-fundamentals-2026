#define main lab_main
#include "../lab3.cu"
#undef main
#include "cpu.h"
#include <chrono>
#include <cstdint>
#include <iomanip>

bool correct(const std::vector<Pixel>& result, const std::vector<Pixel>& expected) {
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i].x != expected[i].x || result[i].y != expected[i].y || result[i].z != expected[i].z || result[i].w != expected[i].w) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int width = argc > 1 ? std::stoi(argv[1]) : 1024;
    int height = argc > 2 ? std::stoi(argv[2]) : width;
    if (width < 32 || height < 32 || (int64_t)width * height > 100000000) return 1;
    int size = width * height;
    size_t bytes = size * sizeof(Pixel);
    const int nc = 5;
    const int repeats = 100;
    std::vector<Pixel> image(size), expected(size), actual(size);
    uint32_t state = 7;
    for (int i = 0; i < size; i++) {
        state = state * 1664525u + 1013904223u;
        image[i] = {(unsigned char)(state >> 24), (unsigned char)(state >> 16), (unsigned char)(state >> 8), 255};
    }

    double host_avg[32][3] = {};
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < 16; j++) {
            Pixel pixel = image[i * 16 + j];
            host_avg[i][0] += pixel.x;
            host_avg[i][1] += pixel.y;
            host_avg[i][2] += pixel.z;
        }
        for (int j = 0; j < 3; j++) host_avg[i][j] /= 16;
        double length = sqrt(host_avg[i][0] * host_avg[i][0] + host_avg[i][1] * host_avg[i][1] + host_avg[i][2] * host_avg[i][2]);
        for (int j = 0; j < 3; j++) host_avg[i][j] /= length;
    }

    expected = image;
    cpu_classify(expected.data(), size, host_avg, nc);
    double cpu_ms = 0;
    for (int repeat = 0; repeat < repeats; repeat++) {
        actual = image;
        auto start = std::chrono::steady_clock::now();
        cpu_classify(actual.data(), size, host_avg, nc);
        auto stop = std::chrono::steady_clock::now();
        cpu_ms += std::chrono::duration<double, std::milli>(stop - start).count();
        if (!correct(actual, expected)) return 1;
    }
    cpu_ms /= repeats;

    uchar4* device_image;
    check(cudaMalloc(&device_image, bytes));
    check(cudaMemcpyToSymbol(avg, host_avg, nc * 3 * sizeof(double)));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    int configs[10][2] = {{1, 32}, {8, 64}, {16, 128}, {32, 128}, {64, 128}, {128, 256}, {256, 256}, {512, 512}, {1024, 512}, {1024, 1024}};

    std::cout << "width,height,classes,blocks,threads,repeats,cpu_checks,gpu_checks,cpu_ms,gpu_ms,speedup,errors\n";
    for (auto& config : configs) {
        int blocks = config[0];
        int threads = config[1];
        check(cudaMemcpy(device_image, image.data(), bytes, cudaMemcpyHostToDevice));
        classify<<<blocks, threads>>>(device_image, size, nc);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        double gpu_ms = 0;
        for (int repeat = 0; repeat < repeats; repeat++) {
            check(cudaMemcpy(device_image, image.data(), bytes, cudaMemcpyHostToDevice));
            check(cudaEventRecord(start));
            classify<<<blocks, threads>>>(device_image, size, nc);
            check(cudaGetLastError());
            check(cudaEventRecord(stop));
            check(cudaEventSynchronize(stop));
            float elapsed;
            check(cudaEventElapsedTime(&elapsed, start, stop));
            gpu_ms += elapsed;
            check(cudaMemcpy(actual.data(), device_image, bytes, cudaMemcpyDeviceToHost));
            if (!correct(actual, expected)) {
                std::cerr << "ERROR: CPU/GPU mismatch\n";
                return 1;
            }
        }
        gpu_ms /= repeats;
        std::cout << std::setprecision(10) << width << ',' << height << ',' << nc << ',' << blocks << ',' << threads << ',' << repeats << ',' << repeats << ',' << repeats << ',' << cpu_ms << ',' << gpu_ms << ',' << cpu_ms / gpu_ms << ",0" << std::endl;
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(device_image));
    return 0;
}
