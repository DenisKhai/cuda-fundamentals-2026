#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaDeviceProp gpu;
    cudaError_t error = cudaGetDeviceProperties(&gpu, 0);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }
    int runtime = 0, driver = 0;
    error = cudaRuntimeGetVersion(&runtime);
    if (error == cudaSuccess) error = cudaDriverGetVersion(&driver);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }
    std::printf("Модель\t%s\n", gpu.name);
    std::printf("Compute capability\t%d.%d\n", gpu.major, gpu.minor);
    std::printf("CUDA runtime\t%d\nCUDA driver\t%d\n", runtime, driver);
    std::printf("Глобальная память, байт\t%zu\n", gpu.totalGlobalMem);
    std::printf("Разделяемая память на блок, байт\t%zu\n", gpu.sharedMemPerBlock);
    std::printf("Константная память, байт\t%zu\n", gpu.totalConstMem);
    std::printf("Регистров на блок\t%d\n", gpu.regsPerBlock);
    std::printf("Максимум потоков на блок\t%d\n", gpu.maxThreadsPerBlock);
    std::printf("Размеры блока (x, y, z)\t%d x %d x %d\n", gpu.maxThreadsDim[0], gpu.maxThreadsDim[1], gpu.maxThreadsDim[2]);
    std::printf("Размеры сетки (x, y, z)\t%d x %d x %d\n", gpu.maxGridSize[0], gpu.maxGridSize[1], gpu.maxGridSize[2]);
    std::printf("Мультипроцессоров\t%d\n", gpu.multiProcessorCount);
    return 0;
}
