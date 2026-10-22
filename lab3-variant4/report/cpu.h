#pragma once

struct Pixel {
    unsigned char x, y, z, w;
};

void cpu_classify(Pixel* image, int size, const double avg[][3], int nc);
