#define main lab_main
#include "../lab2.cu"
#undef main
#define CPU_NO_MAIN
#include "cpu.cpp"
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    try {
        int device = 0;
        cudaDeviceProp p;
        cuda_check(cudaGetDevice(&device));
        cuda_check(cudaGetDeviceProperties(&p, device));
        if (argc == 2 && std::string(argv[1]) == "--info") {
            std::cout << "name," << p.name << '\n'
                      << "compute_capability," << p.major << '.' << p.minor << '\n'
                      << "global_memory_bytes," << p.totalGlobalMem << '\n'
                      << "shared_memory_per_block_bytes," << p.sharedMemPerBlock << '\n'
                      << "constant_memory_bytes," << p.totalConstMem << '\n'
                      << "registers_per_block," << p.regsPerBlock << '\n'
                      << "max_threads_per_block," << p.maxThreadsPerBlock << '\n'
                      << "max_threads_dimensions," << p.maxThreadsDim[0] << 'x' << p.maxThreadsDim[1] << 'x' << p.maxThreadsDim[2] << '\n'
                      << "max_grid_dimensions," << p.maxGridSize[0] << 'x' << p.maxGridSize[1] << 'x' << p.maxGridSize[2] << '\n'
                      << "multiprocessors," << p.multiProcessorCount << '\n';
            return 0;
        }
        int width = 32;
        if (argc > 1) {
            width = std::stoi(argv[1]);
        }
        int height = width;
        if (argc > 2) {
            height = std::stoi(argv[2]);
        }
        if (width <= 0 || height <= 0 || width >= 65536 || height >= 65536 || (uint64_t)width * height > 100000000) {
            std::cerr << "ERROR: invalid image size\n";
            return 1;
        }
        size_t count = (size_t)width * height;
        size_t bytes = count * sizeof(Pixel);
        std::vector<Pixel> image(count);
        std::vector<Pixel> expected(count);
        std::vector<Pixel> actual(count);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Pixel pixel;
                pixel.x = (x * 17 + y * 3) % 256;
                pixel.y = (x * 5 + y * 11) % 256;
                pixel.z = (x ^ y) % 256;
                pixel.w = 0;
                image[(size_t)y * width + x] = pixel;
            }
        }
        cpu_sobel(image.data(), expected.data(), width, height);
        double cpu_ms = 0;
        int repeats = 100;
        for (int i = 0; i < repeats; ++i) {
            auto start = std::chrono::steady_clock::now();
            cpu_sobel(image.data(), expected.data(), width, height);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            cpu_ms += elapsed.count();
        }
        cpu_ms /= repeats;
        uchar4* gpu_image;
        cuda_check(cudaMalloc(&gpu_image, bytes));
        cuda_check(cudaMemcpy(gpu_image, image.data(), bytes, cudaMemcpyHostToDevice));
        uchar4* output;
        cuda_check(cudaMalloc(&output, bytes));
        cudaEvent_t start;
        cudaEvent_t end;
        cuda_check(cudaEventCreate(&start));
        cuda_check(cudaEventCreate(&end));

        int configurations[7][4] = {
            {1, 1, 32, 1},
            {4, 4, 8, 8},
            {8, 8, 16, 16},
            {16, 16, 16, 16},
            {32, 32, 16, 16},
            {32, 32, 32, 16},
            {32, 32, 32, 32}
        };
        std::cout << "width,height,blocks,threads,grid,block,repeats,cpu_ms,gpu_ms,speedup,max_error\n";
        for (int config = 0; config < 7; ++config) {
            dim3 grid(configurations[config][0], configurations[config][1]);
            dim3 block(configurations[config][2], configurations[config][3]);
            sobel_filter<<<grid, block>>>(gpu_image, output, width, height);
            cuda_check(cudaGetLastError());
            cuda_check(cudaDeviceSynchronize());
            double gpu_ms = 0;
            for (int i = 0; i < repeats; ++i) {
                cuda_check(cudaEventRecord(start));
                sobel_filter<<<grid, block>>>(gpu_image, output, width, height);
                cuda_check(cudaGetLastError());
                cuda_check(cudaEventRecord(end));
                cuda_check(cudaEventSynchronize(end));
                float elapsed;
                cuda_check(cudaEventElapsedTime(&elapsed, start, end));
                gpu_ms += elapsed;
            }
            gpu_ms /= repeats;
            cuda_check(cudaMemcpy(actual.data(), output, bytes, cudaMemcpyDeviceToHost));
            int max_error = 0;
            for (size_t i = 0; i < count; ++i) {
                int error = std::abs((int)actual[i].x - expected[i].x);
                if (error > max_error) {
                    max_error = error;
                }
                if (actual[i].x != actual[i].y || actual[i].x != actual[i].z || actual[i].w != 0) {
                    throw std::runtime_error("invalid output channels");
                }
            }
            if (max_error > 1) {
                throw std::runtime_error("CPU/GPU mismatch");
            }
            std::cout << std::setprecision(9) << width << ',' << height << ','
                      << grid.x * grid.y << ',' << block.x * block.y << ','
                      << grid.x << 'x' << grid.y << ',' << block.x << 'x' << block.y << ','
                      << repeats << ',' << cpu_ms << ',' << gpu_ms << ',' << cpu_ms / gpu_ms << ',' << max_error << std::endl;
        }
        cuda_check(cudaEventDestroy(start));
        cuda_check(cudaEventDestroy(end));
        cuda_check(cudaFree(output));
        cuda_check(cudaFree(gpu_image));
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
