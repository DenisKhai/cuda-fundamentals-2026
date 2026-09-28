#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(0);
    }
}

__global__ void bubble_sort(float* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                float temp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n;

    if (scanf("%d", &n) != 1 || n < 0) {
        fprintf(stderr, "ERROR: invalid size\n");
        return 0;
    }

    if (n == 0) return 0;

    size_t bytes = n * sizeof(float);
    float* a = (float*)malloc(bytes);

    if (a == NULL) {
        fprintf(stderr, "ERROR: not enough memory\n");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        if (scanf("%f", &a[i]) != 1) {
            fprintf(stderr, "ERROR: invalid number\n");
            free(a);
            return 0;
        }
    }

    float* d_a;
    check(cudaMalloc(&d_a, bytes));
    check(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    bubble_sort<<<1, 1>>>(d_a, n);
    check(cudaGetLastError());
    check(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(d_a));

    for (int i = 0; i < n; i++) {
        printf("%.6e ", a[i]);
    }
    
    printf("\n");
    free(a);
    return 0;
}
