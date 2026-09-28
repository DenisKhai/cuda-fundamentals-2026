#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct Answer {
    int count;
    float x1;
    float x2;
};


void check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(0);
    }
}


__global__ void solve(float a, float b, float c, Answer* ans) {
    if (a == 0.0f) {
        if (b == 0.0f) {
            if (c == 0.0f)
                ans->count = -1;
            else
                ans->count = -2;
        } else {
            ans->count = 1;
            ans->x1 = -c / b;
            if (!isfinite(ans->x1)) {
                ans->count = -3;
            } else if (ans->x1 == 0.0f) {
                ans->x1 = 0.0f;
            }
        }
        return;
    }

    float d = b * b - 4.0f * a * c;
    if (!isfinite(d)) {
        ans->count = -3;
        return;
    }

    if (d < 0.0f) {
        ans->count = 0;
    } else if (d == 0.0f) {
        ans->count = 1;
        ans->x1 = -b / (2.0f * a);
        if (!isfinite(ans->x1)) {
            ans->count = -3;
        } else if (ans->x1 == 0.0f) {
            ans->x1 = 0.0f;
        }
    } else {
        float sqrt_d = sqrtf(d);
        ans->count = 2;
        ans->x1 = (-b + sqrt_d) / (2.0f * a);
        ans->x2 = (-b - sqrt_d) / (2.0f * a);
        if (!isfinite(ans->x1) || !isfinite(ans->x2)) {
            ans->count = -3;
        } else {
            if (ans->x1 == 0.0f)
                ans->x1 = 0.0f;
            if (ans->x2 == 0.0f)
                ans->x2 = 0.0f;
        }
    }
}


int main() {
    float a, b, c;
    if (scanf("%f %f %f", &a, &b, &c) != 3) {
        fprintf(stderr, "ERROR: invalid input\n");
        return 0;
    }
    if (!isfinite(a) || !isfinite(b) || !isfinite(c)) {
        fprintf(stderr, "ERROR: invalid input\n");
        return 0;
    }

    Answer* ans;
    check(cudaMallocManaged(&ans, sizeof(Answer)));
    solve<<<1, 1>>>(a, b, c, ans);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    if (ans->count == 2) {
        printf("%.6f %.6f\n", ans->x1, ans->x2);
    } else if (ans->count == 1) {
        printf("%.6f\n", ans->x1);
    } else if (ans->count == 0) {
        printf("imaginary\n");
    } else if (ans->count == -1) {
        printf("any\n");
    } else if (ans->count == -3) {
        fprintf(stderr, "ERROR: calculation overflow\n");
    } else {
        printf("incorrect\n");
    }

    check(cudaFree(ans));
    return 0;
}
