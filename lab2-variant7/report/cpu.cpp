#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Pixel {
    unsigned char x, y, z, w;
};

float cpu_gray(const Pixel* image, int width, int height, int x, int y) {
    x = std::max(0, std::min(width - 1, x));
    y = std::max(0, std::min(height - 1, y));
    Pixel p = image[(size_t)y * width + x];
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

void cpu_sobel(const Pixel* image, Pixel* result, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float p1 = cpu_gray(image, width, height, x - 1, y - 1);
            float p2 = cpu_gray(image, width, height, x, y - 1);
            float p3 = cpu_gray(image, width, height, x + 1, y - 1);
            float p4 = cpu_gray(image, width, height, x - 1, y);
            float p6 = cpu_gray(image, width, height, x + 1, y);
            float p7 = cpu_gray(image, width, height, x - 1, y + 1);
            float p8 = cpu_gray(image, width, height, x, y + 1);
            float p9 = cpu_gray(image, width, height, x + 1, y + 1);
            float gx = -p1 + p3 - 2 * p4 + 2 * p6 - p7 + p9;
            float gy = -p1 - 2 * p2 - p3 + p7 + 2 * p8 + p9;
            float value = std::min(255.0f, std::sqrt(gx * gx + gy * gy));
            unsigned char c = (unsigned char)value;
            result[(size_t)y * width + x] = {c, c, c, 0};
        }
    }
}

#ifndef CPU_NO_MAIN
int main() {
    try {
        std::string src, dst;
        std::getline(std::cin, src);
        std::getline(std::cin, dst);
        std::ifstream input(src, std::ios::binary);
        int32_t width = 0, height = 0;
        input.read((char*)&width, 4);
        input.read((char*)&height, 4);
        if (!input || width <= 0 || height <= 0 || width >= 65536 || height >= 65536 ||
            (uint64_t)width * height > 100000000) {
            std::cerr << "ERROR: invalid image\n";
            return 0;
        }
        size_t count = (size_t)width * height;
        std::vector<Pixel> image(count), result(count);
        input.read((char*)image.data(), count * sizeof(Pixel));
        if (!input) {
            std::cerr << "ERROR: truncated image\n";
            return 0;
        }
        cpu_sobel(image.data(), result.data(), width, height);
        std::ofstream output(dst, std::ios::binary);
        output.write((char*)&width, 4);
        output.write((char*)&height, 4);
        output.write((char*)result.data(), count * sizeof(Pixel));
        output.close();
        if (!output) std::cerr << "ERROR: cannot write image\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
    }
    return 0;
}
#endif
