// cublas_gemm_bench.cu
// Compilar: nvcc -O3 -o cublas_bench cublas_gemm_bench.cu -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) \
    do { cudaError_t e = (call); if(e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

#define CHECK_CUBLAS(call) \
    do { cublasStatus_t s = (call); if(s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); exit(1); } } while(0)

void benchmark_gemm(int N, int warmup, int iters) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double flops  = 2.0 * (double)N * N * N;
    double gflops = flops / (ms * 1e6);

    printf("N=%5d | tiempo=%.3f ms | GFLOPS=%.1f\n", N, ms, gflops);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
}

int main(int argc, char** argv) {
    // Tamaños a evaluar (puedes editarlos o pasarlos como argumentos)
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup  = 3;
    int iters   = 10;

    // Si se pasa N como argumento: ./cublas_bench 2048
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