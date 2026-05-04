/*
 * gemm_cpu_mkl.c
 *
 * Benchmark de GEMM para CPU basado en BLAS, compatible con MKL u OpenBLAS.
 * Soporta las cuatro precisiones usadas en el proyecto:
 *   S -> float real
 *   D -> double real
 *   C -> float compleja
 *   Z -> double compleja
 *
 * El objetivo es mantener la misma interfaz que el binario CUDA `GEMMparametros`
 * para que `benchmark_runner.py` pueda ejecutar CPU o GPU con el mismo esquema
 * de parametros y leer la misma salida parseable.
 *
 * ------------------------------------------------------------
 * GUIA DE COMPILACION
 * ------------------------------------------------------------
 * Opcion 1: MKL
 *   gcc -O3 -march=native -o gemm_cpu_mkl gemm_cpu_mkl.c -lmkl_rt -lpthread -lm
 *
 * Opcion 2: OpenBLAS
 *   gcc -O3 -march=native -o gemm_cpu_mkl gemm_cpu_mkl.c -I/usr/include/openblas -lopenblas -lm
 *
 * Si tu sistema no encuentra cblas.h, revisa que el paquete de desarrollo de BLAS
 * este instalado o ajusta la ruta de include con -I.
 *
 * ------------------------------------------------------------
 * GUIA DE EJECUCION
 * ------------------------------------------------------------
 * Sintaxis general:
 *   ./gemm_cpu_mkl M N K <S|D|C|Z> [OpA] [OpB]
 *
 * Argumentos:
 *   M, N, K   -> dimensiones del producto C = A * B
 *   Precision -> S, D, C o Z
 *   OpA       -> N, T o C para la matriz A
 *   OpB       -> N, T o C para la matriz B
 *
 * Operaciones:
 *   N = No transpose
 *   T = Transpose
 *   C = Conjugate transpose
 *
 * Ejemplos de uso:
 *   ./gemm_cpu_mkl 128 128 128 S N N
 *   ./gemm_cpu_mkl 256 256 256 D T N
 *   ./gemm_cpu_mkl 512 256 128 C N T
 *   ./gemm_cpu_mkl 1024 1024 1024 Z C C
 *
 * Ejemplo para barrido desde el runner:
 *   python3 benchmark_runner.py --device cpu --binary ./gemm_cpu_mkl \
 *       --sizes 128,256,512 --precisions S,D,C,Z --output cpu_results.csv
 *
 * ------------------------------------------------------------
 * SALIDA ESPERADA
 * ------------------------------------------------------------
 * El programa imprime una sola linea con este formato:
 *   M=... N=... K=... Precision=... OpA=... OpB=... Time_sec=...
 *
 * Esa linea es la que usa el runner para calcular GFLOPS, energia y EDP.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <cblas.h>
#include <complex.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static CBLAS_TRANSPOSE parse_op(char c) {
    if (c == 'T' || c == 't') return CblasTrans;
    if (c == 'C' || c == 'c') return CblasConjTrans;
    return CblasNoTrans;
}

int run_sgemm(int M, int N, int K, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int warmup, int iters, double *out_time_sec) {
    float *A = malloc((size_t)M * K * sizeof(float));
    float *B = malloc((size_t)K * N * sizeof(float));
    float *C = malloc((size_t)M * N * sizeof(float));
    if (!A || !B || !C) return -1;

    for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0f;

    const float alpha = 1.0f, beta = 0.0f;
    int lda = K, ldb = N, ldc = N;

    for (int i = 0; i < warmup; ++i) {
        cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iters; ++i) {
        cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    double t1 = get_time_sec();
    *out_time_sec = (t1 - t0) / iters;

    free(A); free(B); free(C);
    return 0;
}

int run_dgemm(int M, int N, int K, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int warmup, int iters, double *out_time_sec) {
    double *A = malloc((size_t)M * K * sizeof(double));
    double *B = malloc((size_t)K * N * sizeof(double));
    double *C = malloc((size_t)M * N * sizeof(double));
    if (!A || !B || !C) return -1;

    for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0;

    const double alpha = 1.0, beta = 0.0;
    int lda = K, ldb = N, ldc = N;

    for (int i = 0; i < warmup; ++i) {
        cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    double t1 = get_time_sec();
    *out_time_sec = (t1 - t0) / iters;

    free(A); free(B); free(C);
    return 0;
}

int run_cgemm(int M, int N, int K, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int warmup, int iters, double *out_time_sec) {
    float complex *A = malloc((size_t)M * K * sizeof(float complex));
    float complex *B = malloc((size_t)K * N * sizeof(float complex));
    float complex *C = malloc((size_t)M * N * sizeof(float complex));
    if (!A || !B || !C) return -1;

    for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = (float)rand() / RAND_MAX + (float)rand() / RAND_MAX * I;
    for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = (float)rand() / RAND_MAX + (float)rand() / RAND_MAX * I;
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0f + 0.0f * I;

    float complex alpha = 1.0f + 0.0f * I;
    float complex beta = 0.0f + 0.0f * I;
    int lda = K, ldb = N, ldc = N;

    for (int i = 0; i < warmup; ++i) {
        cblas_cgemm(CblasRowMajor, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iters; ++i) {
        cblas_cgemm(CblasRowMajor, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    double t1 = get_time_sec();
    *out_time_sec = (t1 - t0) / iters;

    free(A); free(B); free(C);
    return 0;
}

int run_zgemm(int M, int N, int K, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int warmup, int iters, double *out_time_sec) {
    double complex *A = malloc((size_t)M * K * sizeof(double complex));
    double complex *B = malloc((size_t)K * N * sizeof(double complex));
    double complex *C = malloc((size_t)M * N * sizeof(double complex));
    if (!A || !B || !C) return -1;

    for (size_t i = 0; i < (size_t)M * K; ++i) A[i] = (double)rand() / RAND_MAX + (double)rand() / RAND_MAX * I;
    for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = (double)rand() / RAND_MAX + (double)rand() / RAND_MAX * I;
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0 + 0.0 * I;

    double complex alpha = 1.0 + 0.0 * I;
    double complex beta = 0.0 + 0.0 * I;
    int lda = K, ldb = N, ldc = N;

    for (int i = 0; i < warmup; ++i) {
        cblas_zgemm(CblasRowMajor, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iters; ++i) {
        cblas_zgemm(CblasRowMajor, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    double t1 = get_time_sec();
    *out_time_sec = (t1 - t0) / iters;

    free(A); free(B); free(C);
    return 0;
}

int main(int argc, char **argv) {
    int warmup = 3;
    int iters = 10;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s M N K <S|D|C|Z> [OpA] [OpB]\n", argv[0]);
        fprintf(stderr, "Example: %s 512 512 512 S N N\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    char prec = toupper((unsigned char)argv[4][0]);
    char opA = 'N';
    char opB = 'N';
    if (argc >= 6) opA = toupper((unsigned char)argv[5][0]);
    if (argc >= 7) opB = toupper((unsigned char)argv[6][0]);

    CBLAS_TRANSPOSE tA = parse_op(opA);
    CBLAS_TRANSPOSE tB = parse_op(opB);

    double time_sec = 0.0;
    int rc = 0;

    switch (prec) {
        case 'S': rc = run_sgemm(M, N, K, tA, tB, warmup, iters, &time_sec); break;
        case 'D': rc = run_dgemm(M, N, K, tA, tB, warmup, iters, &time_sec); break;
        case 'C': rc = run_cgemm(M, N, K, tA, tB, warmup, iters, &time_sec); break;
        case 'Z': rc = run_zgemm(M, N, K, tA, tB, warmup, iters, &time_sec); break;
        default:
            fprintf(stderr, "Precision invalida: %c\n", prec);
            return 2;
    }

    if (rc != 0) {
        fprintf(stderr, "Fallo al ejecutar GEMM (posible fallo de asignacion de memoria)\n");
        return 3;
    }

    printf("M=%d N=%d K=%d Precision=%c OpA=%c OpB=%c Time_sec=%.9f\n", M, N, K, prec, opA, opB, time_sec);
    return 0;
}
