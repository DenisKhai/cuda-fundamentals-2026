#include "cpu.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void cpu_classify(Pixel* image, int size, const double avg[][3], int nc) {
    for (int index = 0; index < size; index++) {
        Pixel pixel = image[index];
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
    }
}

#ifndef CPU_NO_MAIN
int main() {
    std::string input_name, output_name;
    int nc;
    std::cin >> input_name >> output_name >> nc;
    std::ifstream input(input_name, std::ios::binary);
    int width, height;
    input.read((char*)&width, sizeof(int));
    input.read((char*)&height, sizeof(int));
    if (!input || width <= 0 || height <= 0 || nc < 1 || nc > 32) return 1;
    int size = width * height;
    std::vector<Pixel> image(size);
    input.read((char*)image.data(), size * sizeof(Pixel));
    if (!input) return 1;
    input.close();

    double avg[32][3] = {};
    for (int i = 0; i < nc; i++) {
        int np;
        std::cin >> np;
        if (np <= 0) return 1;
        for (int j = 0; j < np; j++) {
            int x, y;
            std::cin >> x >> y;
            if (!std::cin || x < 0 || x >= width || y < 0 || y >= height) return 1;
            Pixel pixel = image[y * width + x];
            avg[i][0] += pixel.x;
            avg[i][1] += pixel.y;
            avg[i][2] += pixel.z;
        }
        for (int j = 0; j < 3; j++) avg[i][j] /= np;
        double length = sqrt(avg[i][0] * avg[i][0] + avg[i][1] * avg[i][1] + avg[i][2] * avg[i][2]);
        if (length == 0) return 1;
        for (int j = 0; j < 3; j++) avg[i][j] /= length;
    }

    cpu_classify(image.data(), size, avg, nc);
    std::ofstream output(output_name, std::ios::binary);
    output.write((char*)&width, sizeof(int));
    output.write((char*)&height, sizeof(int));
    output.write((char*)image.data(), size * sizeof(Pixel));
    output.close();
    return output ? 0 : 1;
}
#endif
