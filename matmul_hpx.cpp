// HPX parallel matrix multiplication demo
// Author: Rahul Surya <s2894842@ed.ac.uk>

#include <hpx/hpx_main.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

// Row-major flat storage: A[i][j] = A[i*N + j]
static void matmul_serial(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    std::size_t N)
{
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

static void matmul_parallel(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    std::size_t N)
{
    hpx::for_loop(hpx::execution::par, std::size_t(0), N,
        [&](std::size_t i) {
            for (std::size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < N; ++k)
                    sum += A[i * N + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
        });
}

static double checksum(const std::vector<double>& M)
{
    return std::accumulate(M.begin(), M.end(), 0.0);
}

int hpx_main(int argc, char* argv[])
{
    const std::size_t N =
        (argc > 1) ? static_cast<std::size_t>(std::atoi(argv[1])) : 512;

    std::cout << "Matrix size : " << N << " x " << N << "\n";

    std::vector<double> A(N * N), B(N * N);
    std::vector<double> C_serial(N * N, 0.0), C_par(N * N, 0.0);

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            A[i * N + j] = static_cast<double>(i + 1) / static_cast<double>(N);
            B[i * N + j] = static_cast<double>(j + 1) / static_cast<double>(N);
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_serial(A, B, C_serial, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    matmul_parallel(A, B, C_par, N);
    auto t3 = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "Serial   time : " << serial_ms << " ms\n";
    std::cout << "Parallel time : " << par_ms    << " ms\n";
    std::cout << "Speedup        : " << (serial_ms / par_ms) << "x\n";
    std::cout << "Checksum (ser) : " << checksum(C_serial) << "\n";
    std::cout << "Checksum (par) : " << checksum(C_par)    << "\n";
    std::cout << "Max diff       : " << std::abs(checksum(C_serial) - checksum(C_par)) << "\n";

    return hpx::finalize();
}
