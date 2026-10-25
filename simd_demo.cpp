#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>

float sum_linear(const float* arr, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}

float sum_simd(const float* arr, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    __m256 v_sum = _mm256_setzero_ps();

    for (; i + 7 < n; i += 8) {
        __m256 v_arr = _mm256_loadu_ps(&arr[i]);
        v_sum = _mm256_add_ps(v_sum, v_arr);
    }

    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, v_sum);

    for (int j = 0; j < 8; ++j) {
        sum += temp[j];
    }

    for (; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}

int main() {
    const size_t size = 10000000;
    const int iterations = 10;

    std::vector<float> data(size, 1.5f);
    std::vector<double> linear_times;
    std::vector<double> simd_times;

    linear_times.reserve(iterations);
    simd_times.reserve(iterations);

    volatile float dummy_sink = 0.0f;
    dummy_sink = sum_linear(data.data(), size);
    dummy_sink = sum_simd(data.data(), size);

    for (int i = 0; i < iterations; ++i) {
        auto start_linear = std::chrono::high_resolution_clock::now();
        dummy_sink = sum_linear(data.data(), size);
        auto end_linear = std::chrono::high_resolution_clock::now();

        auto start_simd = std::chrono::high_resolution_clock::now();
        dummy_sink = sum_simd(data.data(), size);
        auto end_simd = std::chrono::high_resolution_clock::now();

        linear_times.push_back(std::chrono::duration<double, std::micro>(end_linear - start_linear).count());
        simd_times.push_back(std::chrono::duration<double, std::micro>(end_simd - start_simd).count());
    }

    std::sort(linear_times.begin(), linear_times.end());
    std::sort(simd_times.begin(), simd_times.end());

    double p50_linear = linear_times[iterations / 2];
    double p90_linear = linear_times[(iterations * 90) / 100 - 1];

    double p50_simd = simd_times[iterations / 2];
    double p90_simd = simd_times[(iterations * 90) / 100 - 1];

    std::cout << std::left << std::setw(12) << "method"
              << std::setw(15) << "p50 (us)"
              << std::setw(15) << "p90 (us)" << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::left << std::setw(12) << "Linear"
              << std::setw(15) << std::fixed << std::setprecision(2) << p50_linear
              << std::setw(15) << std::fixed << std::setprecision(2) << p90_linear << "\n";
    std::cout << std::left << std::setw(12) << "SIMD"
              << std::setw(15) << std::fixed << std::setprecision(2) << p50_simd
              << std::setw(15) << std::fixed << std::setprecision(2) << p90_simd << "\n";

    return 0;
}
