// cufft_bench.cu
// Compilar: $ nvcc -O3 -o fft_gpu fft_gpu.cu -lcufft

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) \
    do { cudaError_t e = (call); if(e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

#define CHECK_CUFFT(call) \
    do { cufftResult r = (call); if(r != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, r); exit(1); } } while(0)

void benchmark_fft(int N, int warmup, int iters) {
    size_t in_bytes  = (size_t)N * sizeof(cufftComplex);
    size_t out_bytes = (size_t)N * sizeof(cufftComplex);

    cufftComplex *h_in = (cufftComplex*)malloc(in_bytes);
    for (int i = 0; i < N; i++) {
        h_in[i].x = (float)rand() / RAND_MAX;
        h_in[i].y = 0.0f;
    }

    cufftComplex *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  in_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Warmup
    for (int i = 0; i < warmup; i++)
        CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iters; i++)
        CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    // GFLOPS teoricos FFT: 5 * N * log2(N)
    double flops  = 5.0 * N * log2((double)N);
    double gflops = flops / (ms * 1e6);

    printf("N=%7d | tiempo=%.3f ms | GFLOPS=%.2f\n", N, ms, gflops);

    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
}

int main(int argc, char** argv) {
    int sizes[] = {4096,16384, 65536, 262144, 1048576};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup  = 3;
    int iters   = 10;

    if (argc == 2) {
        int N = atoi(argv[1]);
        benchmark_fft(N, warmup, iters);
        return 0;
    }

    printf("%-10s %-16s %s\n", "N", "Tiempo (ms)", "GFLOPS");
    printf("------------------------------------------\n");
    for (int i = 0; i < n_sizes; i++)
        benchmark_fft(sizes[i], warmup, iters);

    return 0;
}