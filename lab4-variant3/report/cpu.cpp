#include "cpu.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void cpu_solve(double* matrix, double* solution, int n) {
    if (n < 1 || n > 10000) throw std::runtime_error("n must be between 1 and 10000");
    const std::size_t stride = static_cast<std::size_t>(n);
    for (int k = 0; k < n; ++k) {
        double* column = matrix + stride * k;
        int pivot_row = k;
        for (int row = k + 1; row < n; ++row) {
            if (std::abs(column[row]) > std::abs(column[pivot_row])) pivot_row = row;
        }
        const double pivot = column[pivot_row];
        if (!std::isfinite(pivot) || pivot == 0.0) {
            throw std::runtime_error("singular or non-finite matrix");
        }
        if (pivot_row != k) {
            for (int col = k; col <= n; ++col) {
                std::swap(matrix[stride * col + k], matrix[stride * col + pivot_row]);
            }
        }
        for (int row = k + 1; row < n; ++row) column[row] /= pivot;
        for (int col = k + 1; col <= n; ++col) {
            double* current = matrix + stride * col;
            const double upper = current[k];
            for (int row = k + 1; row < n; ++row) current[row] -= column[row] * upper;
        }
    }
    double* rhs = matrix + stride * n;
    for (int k = n - 1; k >= 0; --k) {
        const double value = rhs[k] / matrix[stride * k + k];
        if (!std::isfinite(value)) throw std::runtime_error("non-finite solution");
        solution[k] = value;
        for (int row = 0; row < k; ++row) rhs[row] -= matrix[stride * k + row] * value;
    }
}

#ifndef CPU_NO_MAIN
int main() {
    try {
        int n;
        if (!(std::cin >> n) || n < 1 || n > 10000) {
            throw std::runtime_error("expected n between 1 and 10000");
        }
        const std::size_t stride = static_cast<std::size_t>(n);
        std::vector<double> matrix(stride * (stride + 1)), solution(n);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                double& value = matrix[stride * col + row];
                if (!(std::cin >> value) || !std::isfinite(value)) {
                    throw std::runtime_error("invalid matrix element");
                }
            }
        }
        for (int row = 0; row < n; ++row) {
            double& value = matrix[stride * n + row];
            if (!(std::cin >> value) || !std::isfinite(value)) {
                throw std::runtime_error("invalid right-hand side");
            }
        }
        cpu_solve(matrix.data(), solution.data(), n);
        std::cout << std::scientific << std::setprecision(10);
        for (int i = 0; i < n; ++i) std::cout << (i ? " " : "") << solution[i];
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
#endif
