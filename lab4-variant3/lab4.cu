#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void fail(const char* message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(1);
}

void check(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}

struct AbsoluteLess {
    __host__ __device__ bool operator()(double a, double b) const {
        return fabs(a) < fabs(b);
    }
};

__global__ void swap_rows(double* a, int n, int k, int pivot) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int j = k + index; j <= n; j += step) {
        double value = a[j * n + k];
        a[j * n + k] = a[j * n + pivot];
        a[j * n + pivot] = value;
    }
}

__global__ void compute_factors(double* a, int n, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = k + 1 + index; i < n; i += step) {
        a[k * n + i] /= a[k * n + k];
    }
}

__global__ void eliminate(double* a, int n, int k) {
    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int col = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int row_step = blockDim.x * gridDim.x;
    int col_step = blockDim.y * gridDim.y;
    for (int j = col; j <= n; j += col_step) {
        for (int i = row; i < n; i += row_step) {
            a[j * n + i] -= a[k * n + i] * a[j * n + k];
        }
    }
}

__global__ void back_substitute(double* a, double* x, int n, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    double value = a[n * n + k] / a[k * n + k];
    if (index == 0) x[k] = value;
    for (int i = index; i < k; i += step) {
        a[n * n + i] -= a[k * n + i] * value;
    }
}

void gpu_solve(double* a, double* x, int n,
               dim3 grid = dim3(32, 32), dim3 block = dim3(32, 8)) {
    int blocks = (n + 255) / 256;
    for (int k = 0; k < n; k++) {
        thrust::device_ptr<double> column(a + k * n);
        auto pivot = thrust::max_element(column + k, column + n, AbsoluteLess{});
        double value;
        check(cudaMemcpy(&value, pivot.get(), sizeof(double), cudaMemcpyDeviceToHost));
        if (value == 0.0 || !std::isfinite(value)) fail("singular or non-finite matrix");
        int row = pivot - column;
        if (row != k) {
            swap_rows<<<blocks, 256>>>(a, n, k, row);
            check(cudaGetLastError());
        }
        if (k + 1 < n) {
            compute_factors<<<blocks, 256>>>(a, n, k);
            check(cudaGetLastError());
            eliminate<<<grid, block>>>(a, n, k);
            check(cudaGetLastError());
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        back_substitute<<<blocks, 256>>>(a, x, n, k);
        check(cudaGetLastError());
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1 || n < 1 || n > 10000) fail("invalid matrix size");
    double *a, *x;
    check(cudaMallocManaged(&a, n * (n + 1) * sizeof(double)));
    check(cudaMallocManaged(&x, n * sizeof(double)));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (scanf("%lf", &a[j * n + i]) != 1 || !std::isfinite(a[j * n + i])) {
                fail("invalid matrix element");
            }
        }
    }
    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &a[n * n + i]) != 1 || !std::isfinite(a[n * n + i])) {
            fail("invalid right-hand side");
        }
    }
    gpu_solve(a, x, n);
    check(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(x[i])) fail("non-finite solution");
    }
    for (int i = 0; i < n; i++) printf("%.10e%s", x[i], i + 1 == n ? "\n" : " ");
    check(cudaFree(x));
    check(cudaFree(a));
    return 0;
}
