#include "cpu.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

void cpu_sort(uint32_t* values, uint32_t* temporary, size_t n) {
    for (int shift = 0; shift < 32; shift += 8) {
        size_t counts[256] = {}, offsets[256];
        for (size_t i = 0; i < n; i++) counts[(values[i] >> shift) & 255]++;
        size_t sum = 0;
        for (int i = 0; i < 256; i++) { offsets[i] = sum; sum += counts[i]; }
        for (size_t i = 0; i < n; i++) temporary[offsets[(values[i] >> shift) & 255]++] = values[i];
        uint32_t* swap = values; values = temporary; temporary = swap;
    }
}

#ifndef CPU_NO_MAIN
int main() {
    try {
        uint32_t n;
        if (fread(&n, sizeof(n), 1, stdin) != 1 || n > 128000000u)
            throw std::runtime_error("invalid array size");
        std::vector<uint32_t> values(n), temporary(n);
        if (fread(values.data(), sizeof(uint32_t), n, stdin) != n)
            throw std::runtime_error("incomplete input");
        cpu_sort(values.data(), temporary.data(), n);
        if (fwrite(values.data(), sizeof(uint32_t), n, stdout) != n || fflush(stdout) != 0)
            throw std::runtime_error("cannot write output");
    } catch (const std::exception& error) {
        fprintf(stderr, "ERROR: %s\n", error.what());
    }
    return 0;
}
#endif
