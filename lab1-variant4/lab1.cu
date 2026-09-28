#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(0);
    }
}

__global__ void vector_min(double* a, const double* b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += step) {
        if (a[i] > b[i]) {
            a[i] = b[i];
        }
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1 || n < 0 || n >= (1 << 25)) {
        fprintf(stderr, "ERROR: invalid vector size\n");
        return 0;
    }
    if (n == 0) return 0;

    size_t bytes = n * sizeof(double);
    double* a;
    double* b;
    check(cudaMallocManaged(&a, bytes));
    check(cudaMallocManaged(&b, bytes));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &a[i]) != 1) {
            fprintf(stderr, "ERROR: invalid first vector\n");
            check(cudaFree(a));
            check(cudaFree(b));
            return 0;
        }
    }
    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &b[i]) != 1) {
            fprintf(stderr, "ERROR: invalid second vector\n");
            check(cudaFree(a));
            check(cudaFree(b));
            return 0;
        }
    }

    vector_min<<<256, 256>>>(a, b, n);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    for (int i = 0; i < n; i++) {
        printf("%.10e ", a[i]);
    }
    printf("\n");
    check(cudaFree(a));
    check(cudaFree(b));
    return 0;
}
