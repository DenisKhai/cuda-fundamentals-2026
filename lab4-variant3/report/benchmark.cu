#ifndef BENCHMARK_CPU_ONLY
#define main lab4_main
#include "../lab4.cu"
#undef main
#include <cuda_profiler_api.h>
#endif
#include "cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Result {
    double ms = 0, error = 0, residual = 0;
};

void make_system(std::vector<double>& a, std::vector<double>& x, int n) {
    uint32_t state = 7;
    for (int i = 0; i < n; i++) {
        double sum = 0;
        x[i] = (i % 2 ? -1.0 : 1.0) * (1.0 + (i % 7) / 8.0);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            state = state * 1664525u + 1013904223u;
            double value = (double(state >> 8) / 16777216.0 - 0.5) / n;
            a[j * n + n - 1 - i] = value;
            sum += std::abs(value);
        }
        a[i * n + n - 1 - i] = 2.0 + sum;
    }
    for (int i = 0; i < n; i++) {
        long double b = 0;
        for (int j = 0; j < n; j++) b += (long double)a[j * n + i] * x[j];
        a[n * n + i] = b;
    }
}

void validate(const std::vector<double>& a, const std::vector<double>& expected,
              const std::vector<double>& x, Result& result) {
    int n = x.size();
    long double error = 0, residual = 0, norm_a = 0, norm_b = 0, norm_x = 0, norm_expected = 0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(x[i])) throw std::runtime_error("non-finite solution");
        error = std::max(error, std::abs((long double)x[i] - expected[i]));
        norm_x = std::max(norm_x, std::abs((long double)x[i]));
        norm_expected = std::max(norm_expected, std::abs((long double)expected[i]));
        long double b = 0, sum = 0;
        for (int j = 0; j < n; j++) {
            b += (long double)a[j * n + i] * x[j];
            sum += std::abs((long double)a[j * n + i]);
        }
        residual = std::max(residual, std::abs(b - a[n * n + i]));
        norm_a = std::max(norm_a, sum);
        norm_b = std::max(norm_b, std::abs((long double)a[n * n + i]));
    }
    double relative = error / norm_expected;
    double scaled = residual / (norm_a * norm_x + norm_b);
    if (!std::isfinite(relative) || !std::isfinite(scaled) || relative > 1e-10 || scaled > 1e-10) {
        throw std::runtime_error("accuracy check failed");
    }
    result.error = std::max(result.error, relative);
    result.residual = std::max(result.residual, scaled);
}

int integer(const char* text, int minimum) {
    size_t end;
    int value = std::stoi(text, &end);
    if (text[end] || value < minimum || value > 10000) throw std::runtime_error("invalid argument");
    return value;
}

int main(int argc, char** argv) {
    try {
        if (argc > 5 || (argc == 5 && std::string(argv[4]) != "--profile")) {
            throw std::runtime_error("usage: benchmark [n [repeats [warmup [--profile]]]]");
        }
        int n = argc > 1 ? integer(argv[1], 1) : 256;
        int repeats = argc > 2 ? integer(argv[2], 1) : 5;
        int warmup = argc > 3 ? integer(argv[3], 0) : 100;
        bool profile = argc == 5;
        std::vector<double> original(n * (n + 1)), expected(n), a(original.size()), x(n);
        make_system(original, expected, n);
        int count = 0;
#ifndef BENCHMARK_CPU_ONLY
        count = 10;
        int configs[][4] = {
            {1, 1, 8, 8}, {4, 4, 8, 8}, {8, 8, 16, 8}, {16, 16, 16, 8}, {16, 16, 32, 8},
            {32, 32, 32, 8}, {16, 16, 16, 16}, {32, 32, 16, 16}, {16, 16, 32, 16}, {32, 32, 32, 16}
        };
        size_t bytes = original.size() * sizeof(double);
        double *device_a, *device_x;
        check(cudaMalloc(&device_a, bytes));
        check(cudaMalloc(&device_x, n * sizeof(double)));
        if (profile) {
            for (int repeat = 0; repeat < warmup; repeat++) {
                check(cudaMemcpy(device_a, original.data(), bytes, cudaMemcpyHostToDevice));
                gpu_solve(device_a, device_x, n);
                check(cudaDeviceSynchronize());
            }
            check(cudaMemcpy(device_a, original.data(), bytes, cudaMemcpyHostToDevice));
            check(cudaDeviceSynchronize());
            check(cudaProfilerStart());
            gpu_solve(device_a, device_x, n);
            check(cudaDeviceSynchronize());
            check(cudaProfilerStop());
            check(cudaMemcpy(x.data(), device_x, n * sizeof(double), cudaMemcpyDeviceToHost));
            Result result;
            validate(original, expected, x, result);
            std::cout << "Profiled solver passed: relative_error=" << result.error
                      << ", scaled_residual=" << result.residual << '\n';
        } else
#else
        if (profile) throw std::runtime_error("profiling requires the CUDA benchmark");
#endif
        {
            std::cout << std::setprecision(12)
                      << "mode,n,grid_x,grid_y,block_x,block_y,repeats,warmup,cpu_checks,gpu_checks,"
                         "cpu_ms,gpu_ms,speedup,cpu_rel_error,cpu_scaled_residual,gpu_rel_error,gpu_scaled_residual,errors\n";
            Result cpu;
            for (int config = -1; config < count; config++) {
                Result result;
                for (int repeat = -warmup; repeat < repeats; repeat++) {
#ifndef BENCHMARK_CPU_ONLY
                    if (config >= 0) {
                        check(cudaMemcpy(device_a, original.data(), bytes, cudaMemcpyHostToDevice));
                        check(cudaDeviceSynchronize());
                    } else
#endif
                    a = original;
                    auto start = std::chrono::steady_clock::now();
#ifndef BENCHMARK_CPU_ONLY
                    if (config >= 0) {
                        auto& c = configs[config];
                        gpu_solve(device_a, device_x, n, dim3(c[0], c[1]), dim3(c[2], c[3]));
                        check(cudaDeviceSynchronize());
                    } else
#endif
                    cpu_solve(a.data(), x.data(), n);
                    auto stop = std::chrono::steady_clock::now();
                    if (repeat >= 0) result.ms += std::chrono::duration<double, std::milli>(stop - start).count();
#ifndef BENCHMARK_CPU_ONLY
                    if (config >= 0) check(cudaMemcpy(x.data(), device_x, n * sizeof(double), cudaMemcpyDeviceToHost));
#endif
                    validate(original, expected, x, result);
                }
                result.ms /= repeats;
                if (config == -1) cpu = result;
#ifndef BENCHMARK_CPU_ONLY
                else {
                    std::cout << "gpu," << n;
                    for (int value : configs[config]) std::cout << ',' << value;
                    std::cout << ',' << repeats << ',' << warmup << ',' << repeats + warmup << ',' << repeats + warmup
                              << ',' << cpu.ms << ',' << result.ms << ',' << cpu.ms / result.ms
                              << ',' << cpu.error << ',' << cpu.residual << ',' << result.error << ',' << result.residual << ",0\n";
                }
#else
                std::cout << "cpu," << n << ",,,,," << repeats << ',' << warmup << ',' << repeats + warmup
                          << ",0," << cpu.ms << ",,," << cpu.error << ',' << cpu.residual << ",,,0\n";
#endif
            }
        }
#ifndef BENCHMARK_CPU_ONLY
        check(cudaFree(device_x));
        check(cudaFree(device_a));
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
