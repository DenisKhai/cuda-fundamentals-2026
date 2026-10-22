#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaDeviceProp gpu;
    cudaError_t error = cudaGetDeviceProperties(&gpu, 0);
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("Модель\t%s\n", gpu.name);
    printf("Compute capability\t%d.%d\n", gpu.major, gpu.minor);
    printf("Глобальная память\t%zu\n", gpu.totalGlobalMem);
    printf("Разделяемая память на блок\t%zu\n", gpu.sharedMemPerBlock);
    printf("Константная память\t%zu\n", gpu.totalConstMem);
    printf("Регистров на блок\t%d\n", gpu.regsPerBlock);
    printf("Максимум потоков на блок\t%d\n", gpu.maxThreadsPerBlock);
    printf("Размеры блока (x, y, z)\t%d × %d × %d; суммарно не более %d потоков\n", gpu.maxThreadsDim[0], gpu.maxThreadsDim[1], gpu.maxThreadsDim[2], gpu.maxThreadsPerBlock);
    printf("Размеры сетки (x, y, z)\t%d × %d × %d блоков\n", gpu.maxGridSize[0], gpu.maxGridSize[1], gpu.maxGridSize[2]);
    printf("Мультипроцессоров\t%d\n", gpu.multiProcessorCount);
    return 0;
}
