#define main lab5_main
#include "../lab5.cu"
#undef main
#include "cpu.h"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 8) throw std::runtime_error("benchmark n repeats warmup warmup_size [--profile grid block]");
        uint n = std::stoul(argv[1]);
        int repeats = std::stoi(argv[2]), warmup = std::stoi(argv[3]);
        uint warmup_size = std::min(n, uint(std::stoul(argv[4])));
        bool profile = argc == 8 && std::string(argv[5]) == "--profile";
        if (n < 1 || n > 128000000 || repeats < 1 || warmup < 0 || !warmup_size || (argc != 5 && !profile))
            throw std::runtime_error("invalid benchmark arguments");
        std::vector<uint> original(n), expected(n), values(n), temporary(n);
        uint state = 508;
        for (uint& value : original) { state = 1664525u * state + 1013904223u; value = state; }
        original[0] = 0; original[n - 1] = 0xffffffffu;
        expected = original;
        cpu_sort(expected.data(), temporary.data(), n);
        if (!std::is_sorted(expected.begin(), expected.end())) throw std::runtime_error("CPU reference is not sorted");
        double cpu_ms = 0;
        if (!profile) {
            for (int repeat = -1; repeat < repeats; repeat++) {
                values = original;
                auto start = Clock::now();
                cpu_sort(values.data(), temporary.data(), n);
                auto stop = Clock::now();
                if (values != expected) throw std::runtime_error("CPU mismatch");
                if (repeat >= 0) cpu_ms += std::chrono::duration<double, std::milli>(stop - start).count();
            }
            cpu_ms /= repeats;
        }
        DeviceArray input(n), buffer(n);
        const int configs[][2] = {{1, 32}, {32, 128}, {128, 256}, {256, 256}, {1024, 1024}};
        if (!profile) std::cout << "n,grid,block,repeats,warmup,warmup_size,cpu_ms,gpu_ms,speedup,cpu_checks,gpu_checks,errors\n";
        int count = profile ? 1 : 5;
        for (int config = 0; config < count; config++) {
            int grid = profile ? std::stoi(argv[6]) : configs[config][0];
            int block = profile ? std::stoi(argv[7]) : configs[config][1];
            RadixSort sorter(n, grid, block);
            std::vector<uint> warm_values(original.begin(), original.begin() + warmup_size), warm_expected(warm_values);
            cpu_sort(warm_expected.data(), temporary.data(), warmup_size);
            for (int repeat = 0; repeat < warmup; repeat++) {
                check(cudaMemcpy(input.data, original.data(), warmup_size * sizeof(uint), cudaMemcpyHostToDevice));
                sorter.sort(input.data, buffer.data, warmup_size);
                check(cudaMemcpy(warm_values.data(), input.data, warmup_size * sizeof(uint), cudaMemcpyDeviceToHost));
                if (warm_values != warm_expected) throw std::runtime_error("warmup mismatch");
            }
            double gpu_ms = 0;
            int measured = profile ? 1 : repeats;
            for (int repeat = 0; repeat < measured; repeat++) {
                check(cudaMemcpy(input.data, original.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
                check(cudaDeviceSynchronize());
                if (profile) check(cudaProfilerStart());
                auto start = Clock::now();
                sorter.sort(input.data, buffer.data, n);
                check(cudaDeviceSynchronize());
                auto stop = Clock::now();
                if (profile) check(cudaProfilerStop());
                gpu_ms += std::chrono::duration<double, std::milli>(stop - start).count();
                check(cudaMemcpy(values.data(), input.data, n * sizeof(uint), cudaMemcpyDeviceToHost));
                if (values != expected) throw std::runtime_error("GPU mismatch");
            }
            gpu_ms /= measured;
            if (profile) std::cout << "Profile passed: output equals CPU result\n";
            else std::cout << std::setprecision(12) << n << ',' << grid << ',' << block << ',' << repeats << ','
                           << warmup << ',' << warmup_size << ',' << cpu_ms << ',' << gpu_ms << ','
                           << cpu_ms / gpu_ms << ',' << repeats + 2 << ',' << repeats + warmup << ",0\n" << std::flush;
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
