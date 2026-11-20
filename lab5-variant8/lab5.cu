#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using uint = uint32_t;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ int padded(int i) { return i + (i >> 5) + (i >> 10); }

__global__ void scan_blocks(const uint* input, uint* prefix, uint* sums, uint n, int bit) {
    extern __shared__ uint shared[];
    int t = threadIdx.x, width = 2 * blockDim.x;
    for (uint tile = blockIdx.x; tile < (n + width - 1) / width; tile += gridDim.x) {
        uint start = tile * width;
        for (int i = t; i < width; i += blockDim.x) {
            uint value = start + i < n ? input[start + i] : 0;
            shared[padded(i)] = start + i >= n ? 0 : (bit < 0 ? value : 1u - ((value >> bit) & 1u));
        }
        __syncthreads();
        // Прямой проход Blelloch: суммы поддеревьев.
        for (int step = 1; step < width; step *= 2) {
            int i = (t + 1) * 2 * step - 1;
            if (i < width) shared[padded(i)] += shared[padded(i - step)];
            __syncthreads();
        }
        if (t == 0) {
            sums[tile] = shared[padded(width - 1)];
            shared[padded(width - 1)] = 0;
        }
        __syncthreads();
        // Обратный проход: исключающие префиксные суммы.
        for (int step = width / 2; step > 0; step /= 2) {
            int i = (t + 1) * 2 * step - 1;
            if (i < width) {
                uint left = shared[padded(i - step)];
                shared[padded(i - step)] = shared[padded(i)];
                shared[padded(i)] += left;
            }
            __syncthreads();
        }
        for (int i = t; i < width; i += blockDim.x)
            if (start + i < n) prefix[start + i] = shared[padded(i)];
        __syncthreads();
    }
}

__global__ void add_offsets(uint* prefix, const uint* offsets, uint n) {
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        prefix[i] += offsets[i / (2 * blockDim.x)];
}

__global__ void scatter(const uint* input, uint* output, const uint* prefix, uint n, int bit) {
    uint zeros = prefix[n - 1] + 1u - ((input[n - 1] >> bit) & 1u);
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        uint value = input[i];
        uint position = ((value >> bit) & 1u) ? zeros + i - prefix[i] : prefix[i];
        output[position] = value;
    }
}

struct DeviceArray {
    uint* data = nullptr;
    explicit DeviceArray(size_t n) { check(cudaMalloc(&data, n * sizeof(uint))); }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

struct ScanLevel {
    DeviceArray sums, prefix;
    explicit ScanLevel(size_t n) : sums(n), prefix(n) {}
};

class RadixSort {
    uint capacity;
    int blocks, threads;
    DeviceArray prefix;
    std::vector<ScanLevel*> levels;

    void scan(const uint* input, uint* output, uint n, int bit, size_t level) {
        uint tiles = (n + 2 * threads - 1) / (2 * threads);
        uint* sums = levels[level]->sums.data;
        int width = 2 * threads;
        size_t shared = (width + (width >> 5) + (width >> 10)) * sizeof(uint);
        scan_blocks<<<blocks, threads, shared>>>(input, output, sums, n, bit);
        check(cudaGetLastError());
        if (tiles > 1) {
            uint* offsets = levels[level]->prefix.data;
            scan(sums, offsets, tiles, -1, level + 1);
            add_offsets<<<blocks, threads>>>(output, offsets, n);
            check(cudaGetLastError());
        }
    }

public:
    explicit RadixSort(uint n, int grid = 256, int block = 256)
        : capacity(n), blocks(grid), threads(block), prefix(n ? n : 1) {
        if (grid < 1 || grid > 1024 || block < 32 || block > 1024 || (block & (block - 1)))
            throw std::runtime_error("invalid launch configuration");
        try {
            do {
                n = (n + 2 * threads - 1) / (2 * threads);
                levels.push_back(new ScanLevel(n ? n : 1));
            } while (n > 1);
        } catch (...) {
            for (auto level : levels) delete level;
            throw;
        }
    }
    ~RadixSort() { for (auto level : levels) delete level; }

    void sort(uint* input, uint* temporary, uint n) {
        if (n > capacity) throw std::runtime_error("buffer is too small");
        if (n < 2) return;
        for (int bit = 0; bit < 32; bit++) {
            scan(input, prefix.data, n, bit, 0);
            scatter<<<blocks, threads>>>(input, temporary, prefix.data, n, bit);
            check(cudaGetLastError());
            uint* swap = input; input = temporary; temporary = swap;
        }
    }
};

int main() {
    try {
        static_assert(sizeof(uint) == 4, "32-bit integers required");
        uint n;
        if (fread(&n, sizeof(n), 1, stdin) != 1 || n > 128000000u)
            throw std::runtime_error("invalid array size");
        std::vector<uint> values(n);
        if (fread(values.data(), sizeof(uint), n, stdin) != n)
            throw std::runtime_error("incomplete input");
        if (n > 1) {
            DeviceArray input(n), temporary(n);
            RadixSort sorter(n);
            check(cudaMemcpy(input.data, values.data(), n * sizeof(uint), cudaMemcpyHostToDevice));
            sorter.sort(input.data, temporary.data, n);
            check(cudaMemcpy(values.data(), input.data, n * sizeof(uint), cudaMemcpyDeviceToHost));
        }
        if (fwrite(values.data(), sizeof(uint), n, stdout) != n || fflush(stdout) != 0)
            throw std::runtime_error("cannot write output");
    } catch (const std::exception& error) {
        fprintf(stderr, "ERROR: %s\n", error.what());
    }
    return 0;
}
