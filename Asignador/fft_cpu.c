// fftw_bench.c
// Compilar: gcc -O3 -o fftw_bench fftw_bench.c -lfftw3 -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

void benchmark_fft(int N, int warmup, int iters) {
    fftw_complex *in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    for (int i = 0; i < N; i++) {
        in[i][0] = (double)rand() / RAND_MAX;  // parte real
        in[i][1] = 0.0;                         // parte imaginaria
    }

    // FFTW_MEASURE busca el plan optimo (tarda mas al inicio pero es mas rapido)
    // Cambia a FFTW_ESTIMATE si quieres que el plan se cree instantaneamente
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);

    // Warmup
    for (int i = 0; i < warmup; i++)
        fftw_execute(plan);

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iters; i++)
        fftw_execute(plan);
    double elapsed = (get_time_ms() - start) / iters;

    double flops  = 5.0 * N * log2((double)N);
    double gflops = flops / (elapsed * 1e6);

    printf("N=%7d | tiempo=%.3f ms | GFLOPS=%.2f\n", N, elapsed, gflops);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

int main(int argc, char** argv) {
    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
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