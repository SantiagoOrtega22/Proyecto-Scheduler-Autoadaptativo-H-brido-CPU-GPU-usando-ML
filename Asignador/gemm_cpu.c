// blas_gemm_bench.c
// OpenBLAS: gcc -O3 -o gemm_cpu gemm_cpu.c  -I/usr/include/openblas -lopenblas -lm
// MKL:      gcc -O3 -o gemm_cpu gemm_cpu.c -lmkl_rt -lm

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <bits/time.h>
#include <linux/time.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

void benchmark_gemm(int N, int warmup, int iters) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }
    double elapsed = (get_time_ms() - start) / iters;

    double flops  = 2.0 * (double)N * N * N;
    double gflops = flops / (elapsed * 1e6);

    printf("N=%5d | tiempo=%.3f ms | GFLOPS=%.1f\n", N, elapsed, gflops);

    free(A); free(B); free(C);
}

int main(int argc, char** argv) {
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup  = 3;
    int iters   = 10;

    if (argc == 2) {
        int N = atoi(argv[1]);
        benchmark_gemm(N, warmup, iters);
        return 0;
    }

    printf("%-8s %-14s %s\n", "N", "Tiempo (ms)", "GFLOPS");
    printf("-----------------------------------\n");
    for (int i = 0; i < n_sizes; i++)
        benchmark_gemm(sizes[i], warmup, iters);

    return 0;
}