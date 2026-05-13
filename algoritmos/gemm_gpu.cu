#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/*
 ============================================================================
                        GUIA DE COMPILACION Y EJECUCION
 ============================================================================

 1. COMPILACION:
    Desde la raiz del proyecto, ejecutar:
    $ nvcc -O3 -o algoritmos/gemm_gpu algoritmos/gemm_gpu.cu -lcublas

 2. EJECUCION - CASO INDIVIDUAL:
    Sintaxis: ./algoritmos/gemm_gpu M N K Precision [OpA] [OpB] [matrix_file]
    
    Parametros:
      M, N, K      : Dimensiones de la matriz (numeros positivos)
      Precision     : S (float), D (double), C (complex), Z (complex double)
      OpA, OpB      : [Opcional] Operaciones sobre A,B:
                      N = No-op (sin transformacion, usa matriz tal cual)
                      T = Transpose (transpone la matriz)
                      C = Conjugate (para complejos: invierte signo parte imaginaria)
                      Si se omiten, usa N,N por defecto
      matrix_file   : [Opcional] Archivo binario con las matrices de entrada generadas por el benchmark_runner

    Operadores (afectan calculo: C = alpha*OpA(A)*OpB(B) + beta*C):
      - N,N: A(MxK) y B(KxN) sin cambios
      - T,N: A transpuesta (KxM) y B sin cambios (KxN)
      - N,T: A sin cambios (MxK) y B transpuesta (NxK)
      - T,T: Ambas transpuestas A(KxM) y B(NxK)
      - C,C: Ambas conjugadas (util para datos complejos)

    Ejemplos:
      # Sin transposicion (N,N) - caso base
      $ ./algoritmos/gemm_gpu 256 256 256 D
      Salida: M=256 N=256 K=256 Precision=D OpA=N OpB=N Time_sec=X.XXXXX

      # Single precision, transpuesta en A solamente
      $ ./algoritmos/gemm_gpu 512 512 512 S T N
      Salida: M=512 N=512 K=512 Precision=S OpA=T OpB=N Time_sec=X.XXXXX

      # Complex double, transposicion en ambas
      $ ./algoritmos/gemm_gpu 1024 1024 1024 Z T T
      Salida: M=1024 N=1024 K=1024 Precision=Z OpA=T OpB=T Time_sec=X.XXXXX

      # Complex precision, conjugadas en ambas
      $ ./algoritmos/gemm_gpu 512 512 512 C C C
      Salida: M=512 N=512 K=512 Precision=C OpA=C OpB=C Time_sec=X.XXXXX

 3. BARRIDO BASELINE (via orchestrador Python):
    Ejecutar todas las combinaciones de tamanos (128-4096) x precisiones (S,D,C,Z)
    con operaciones fijas a N,N (sin transposicion):
    $ python3 benchmark_runner.py
    Genera: benchmark_results.csv con ~96 filas (6 tamanos × 4 precisiones × 4 tamanos)

 4. BARRIDO COMPLETO CON TRANSPOSICIONES:
    Incluir todas las combinaciones de operaciones (N,T,C):
    $ python3 benchmark_runner.py --sweep-transpose --op-a-list N,T --op-b-list N,T
    Genera: benchmark_results_transpose_full.csv con ~384 filas
    (6 tamanos × 4 precisiones × 4 combinaciones OpA,OpB)

 5. SALIDA CSV:
    Columnas: M, N, K, Precision, OpA, OpB, Time_sec, GFLOPS, Avg_Power_W, Energy_J, EDP
    - Time_sec: Tiempo de ejecucion GEMM (milisegundos)
    - GFLOPS: Rendimiento (operaciones en punto flotante por segundo / 1e9)
    - Avg_Power_W: Potencia GPU promedio (W)
    - Energy_J: Energia = Potencia × Tiempo (Joules)
    - EDP: Energy-Delay Product = Energy × Time (Joules × segundos)

 ============================================================================
*/

// Compilar desde la raiz del proyecto:
// nvcc gemm_gpu.cu -O3 -lcublas -o gemm_gpu

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"           \
                      << std::endl;                                              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t st = (call);                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            std::cerr << "cuBLAS error code " << st                             \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"           \
                      << std::endl;                                              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

static bool g_use_random = false;
static bool g_loaded_from_file = false;
static std::vector<float> g_file_a_f32;
static std::vector<float> g_file_b_f32;
static std::vector<float> g_file_c_f32;
static std::vector<double> g_file_a_f64;
static std::vector<double> g_file_b_f64;
static std::vector<double> g_file_c_f64;
static std::vector<cuComplex> g_file_a_c32;
static std::vector<cuComplex> g_file_b_c32;
static std::vector<cuComplex> g_file_c_c32;
static std::vector<cuDoubleComplex> g_file_a_c64;
static std::vector<cuDoubleComplex> g_file_b_c64;
static std::vector<cuDoubleComplex> g_file_c_c64;

static void init_seed_from_env() {
    const char *env = std::getenv("BENCH_SEED");
    if (!env || !*env) {
        return;
    }
    char *end = nullptr;
    unsigned long val = std::strtoul(env, &end, 10);
    if (end == env) {
        return;
    }
    std::srand(static_cast<unsigned int>(val));
    g_use_random = true;
}

static int load_gemm_matrices_from_file(const char *filename, int *M, int *N, int *K, char *precision) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        std::perror("fopen");
        return -1;
    }

    int m = 0, n = 0, k = 0;
    char p = '\0';
    if (fread(&m, sizeof(int), 1, f) != 1 ||
        fread(&n, sizeof(int), 1, f) != 1 ||
        fread(&k, sizeof(int), 1, f) != 1 ||
        fread(&p, sizeof(char), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    g_loaded_from_file = true;
    g_use_random = false;
    g_file_a_f32.clear(); g_file_b_f32.clear(); g_file_c_f32.clear();
    g_file_a_f64.clear(); g_file_b_f64.clear(); g_file_c_f64.clear();
    g_file_a_c32.clear(); g_file_b_c32.clear(); g_file_c_c32.clear();
    g_file_a_c64.clear(); g_file_b_c64.clear(); g_file_c_c64.clear();

    size_t size_a = static_cast<size_t>(m) * static_cast<size_t>(k);
    size_t size_b = static_cast<size_t>(k) * static_cast<size_t>(n);
    size_t size_c = static_cast<size_t>(m) * static_cast<size_t>(n);

    switch (p) {
        case 'S': {
            g_file_a_f32.resize(size_a);
            g_file_b_f32.resize(size_b);
            g_file_c_f32.resize(size_c);
            if (fread(g_file_a_f32.data(), sizeof(float), size_a, f) != size_a ||
                fread(g_file_b_f32.data(), sizeof(float), size_b, f) != size_b ||
                fread(g_file_c_f32.data(), sizeof(float), size_c, f) != size_c) {
                fclose(f);
                return -1;
            }
            break;
        }
        case 'D': {
            g_file_a_f64.resize(size_a);
            g_file_b_f64.resize(size_b);
            g_file_c_f64.resize(size_c);
            if (fread(g_file_a_f64.data(), sizeof(double), size_a, f) != size_a ||
                fread(g_file_b_f64.data(), sizeof(double), size_b, f) != size_b ||
                fread(g_file_c_f64.data(), sizeof(double), size_c, f) != size_c) {
                fclose(f);
                return -1;
            }
            break;
        }
        case 'C': {
            g_file_a_c32.resize(size_a);
            g_file_b_c32.resize(size_b);
            g_file_c_c32.resize(size_c);
            for (size_t i = 0; i < size_a; ++i) {
                float re = 0.0f, im = 0.0f;
                if (fread(&re, sizeof(float), 1, f) != 1 || fread(&im, sizeof(float), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_a_c32[i] = make_cuComplex(re, im);
            }
            for (size_t i = 0; i < size_b; ++i) {
                float re = 0.0f, im = 0.0f;
                if (fread(&re, sizeof(float), 1, f) != 1 || fread(&im, sizeof(float), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_b_c32[i] = make_cuComplex(re, im);
            }
            for (size_t i = 0; i < size_c; ++i) {
                float re = 0.0f, im = 0.0f;
                if (fread(&re, sizeof(float), 1, f) != 1 || fread(&im, sizeof(float), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_c_c32[i] = make_cuComplex(re, im);
            }
            break;
        }
        case 'Z': {
            g_file_a_c64.resize(size_a);
            g_file_b_c64.resize(size_b);
            g_file_c_c64.resize(size_c);
            for (size_t i = 0; i < size_a; ++i) {
                double re = 0.0, im = 0.0;
                if (fread(&re, sizeof(double), 1, f) != 1 || fread(&im, sizeof(double), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_a_c64[i] = make_cuDoubleComplex(re, im);
            }
            for (size_t i = 0; i < size_b; ++i) {
                double re = 0.0, im = 0.0;
                if (fread(&re, sizeof(double), 1, f) != 1 || fread(&im, sizeof(double), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_b_c64[i] = make_cuDoubleComplex(re, im);
            }
            for (size_t i = 0; i < size_c; ++i) {
                double re = 0.0, im = 0.0;
                if (fread(&re, sizeof(double), 1, f) != 1 || fread(&im, sizeof(double), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                g_file_c_c64[i] = make_cuDoubleComplex(re, im);
            }
            break;
        }
        default:
            fclose(f);
            return -1;
    }

    fclose(f);
    *M = m;
    *N = n;
    *K = k;
    *precision = p;
    return 0;
}

static void fill_real(std::vector<double> &buf) {
    if (g_loaded_from_file && !g_file_a_f64.empty() && buf.size() == g_file_a_f64.size()) {
        if (&buf == nullptr) {
            return;
        }
        std::copy(g_file_a_f64.begin(), g_file_a_f64.end(), buf.begin());
        return;
    }
    if (g_use_random) {
        for (double &v : buf) {
            v = static_cast<double>(std::rand()) / RAND_MAX;
        }
        return;
    }
    std::fill(buf.begin(), buf.end(), 1.0);
}

static void fill_real(std::vector<float> &buf) {
    if (g_loaded_from_file && !g_file_a_f32.empty() && buf.size() == g_file_a_f32.size()) {
        if (&buf == nullptr) {
            return;
        }
        std::copy(g_file_a_f32.begin(), g_file_a_f32.end(), buf.begin());
        return;
    }
    if (g_use_random) {
        for (float &v : buf) {
            v = static_cast<float>(std::rand()) / RAND_MAX;
        }
        return;
    }
    std::fill(buf.begin(), buf.end(), 1.0f);
}

static void fill_complex(std::vector<cuComplex> &buf) {
    if (g_loaded_from_file && !g_file_a_c32.empty() && buf.size() == g_file_a_c32.size()) {
        if (&buf == nullptr) {
            return;
        }
        std::copy(g_file_a_c32.begin(), g_file_a_c32.end(), buf.begin());
        return;
    }
    if (g_use_random) {
        for (cuComplex &v : buf) {
            float re = static_cast<float>(std::rand()) / RAND_MAX;
            float im = static_cast<float>(std::rand()) / RAND_MAX;
            v = make_cuComplex(re, im);
        }
        return;
    }
    std::fill(buf.begin(), buf.end(), make_cuComplex(1.0f, 0.0f));
}

static void fill_complex(std::vector<cuDoubleComplex> &buf) {
    if (g_loaded_from_file && !g_file_a_c64.empty() && buf.size() == g_file_a_c64.size()) {
        if (&buf == nullptr) {
            return;
        }
        std::copy(g_file_a_c64.begin(), g_file_a_c64.end(), buf.begin());
        return;
    }
    if (g_use_random) {
        for (cuDoubleComplex &v : buf) {
            double re = static_cast<double>(std::rand()) / RAND_MAX;
            double im = static_cast<double>(std::rand()) / RAND_MAX;
            v = make_cuDoubleComplex(re, im);
        }
        return;
    }
    std::fill(buf.begin(), buf.end(), make_cuDoubleComplex(1.0, 0.0));
}

double benchmark_dgemm(cublasHandle_t handle, int M, int N, int K,
                       cublasOperation_t opA, cublasOperation_t opB) {
    size_t size_a = static_cast<size_t>(M) * K;
    size_t size_b = static_cast<size_t>(K) * N;
    size_t size_c = static_cast<size_t>(M) * N;
    int lda = (opA == CUBLAS_OP_N) ? M : K;
    int ldb = (opB == CUBLAS_OP_N) ? K : N;
    int ldc = M;

    std::vector<double> h_a(size_a);
    std::vector<double> h_b(size_b);
    std::vector<double> h_c(size_c, 0.0);

    if (g_loaded_from_file && !g_file_a_f64.empty()) {
        std::copy(g_file_a_f64.begin(), g_file_a_f64.end(), h_a.begin());
        std::copy(g_file_b_f64.begin(), g_file_b_f64.end(), h_b.begin());
    } else {
        fill_real(h_a);
        fill_real(h_b);
    }

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), size_c * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0;
    const double beta = 0.0;

    CHECK_CUBLAS(cublasDgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasDgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return static_cast<double>(elapsed_ms) / 1000.0;
}

// SGEMM: misma logica de medicion, pero usando float como tipo base.
double benchmark_sgemm(cublasHandle_t handle, int M, int N, int K,
                       cublasOperation_t opA, cublasOperation_t opB) {
    size_t size_a = static_cast<size_t>(M) * K;
    size_t size_b = static_cast<size_t>(K) * N;
    size_t size_c = static_cast<size_t>(M) * N;
    int lda = (opA == CUBLAS_OP_N) ? M : K;
    int ldb = (opB == CUBLAS_OP_N) ? K : N;
    int ldc = M;

    std::vector<float> h_a(size_a);
    std::vector<float> h_b(size_b);
    std::vector<float> h_c(size_c, 0.0f);

    if (g_loaded_from_file && !g_file_a_f32.empty()) {
        std::copy(g_file_a_f32.begin(), g_file_a_f32.end(), h_a.begin());
        std::copy(g_file_b_f32.begin(), g_file_b_f32.end(), h_b.begin());
    } else {
        fill_real(h_a);
        fill_real(h_b);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), size_c * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return static_cast<double>(elapsed_ms) / 1000.0;
}
// CGEMM: usa cuComplex para matrices complejas en precision simple.

double benchmark_cgemm(cublasHandle_t handle, int M, int N, int K,
                       cublasOperation_t opA, cublasOperation_t opB) {
    size_t size_a = static_cast<size_t>(M) * K;
    size_t size_b = static_cast<size_t>(K) * N;
    size_t size_c = static_cast<size_t>(M) * N;
    int lda = (opA == CUBLAS_OP_N) ? M : K;
    int ldb = (opB == CUBLAS_OP_N) ? K : N;
    int ldc = M;

    std::vector<cuComplex> h_a(size_a);
    std::vector<cuComplex> h_b(size_b);
    std::vector<cuComplex> h_c(size_c, make_cuComplex(0.0f, 0.0f));

    if (g_loaded_from_file && !g_file_a_c32.empty()) {
        std::copy(g_file_a_c32.begin(), g_file_a_c32.end(), h_a.begin());
        std::copy(g_file_b_c32.begin(), g_file_b_c32.end(), h_b.begin());
    } else {
        fill_complex(h_a);
        fill_complex(h_b);
    }

    cuComplex* d_a = nullptr;
    cuComplex* d_b = nullptr;
    cuComplex* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(cuComplex)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), size_c * sizeof(cuComplex), cudaMemcpyHostToDevice));

    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);

    CHECK_CUBLAS(cublasCgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasCgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return static_cast<double>(elapsed_ms) / 1000.0;
}
// ZGEMM: usa cuDoubleComplex para matrices complejas en precision doble.

double benchmark_zgemm(cublasHandle_t handle, int M, int N, int K,
                       cublasOperation_t opA, cublasOperation_t opB) {
    size_t size_a = static_cast<size_t>(M) * K;
    size_t size_b = static_cast<size_t>(K) * N;
    size_t size_c = static_cast<size_t>(M) * N;
    int lda = (opA == CUBLAS_OP_N) ? M : K;
    int ldb = (opB == CUBLAS_OP_N) ? K : N;
    int ldc = M;

    std::vector<cuDoubleComplex> h_a(size_a);
    std::vector<cuDoubleComplex> h_b(size_b);
    std::vector<cuDoubleComplex> h_c(size_c, make_cuDoubleComplex(0.0, 0.0));

    if (g_loaded_from_file && !g_file_a_c64.empty()) {
        std::copy(g_file_a_c64.begin(), g_file_a_c64.end(), h_a.begin());
        std::copy(g_file_b_c64.begin(), g_file_b_c64.end(), h_b.begin());
    } else {
        fill_complex(h_a);
        fill_complex(h_b);
    }

    cuDoubleComplex* d_a = nullptr;
    cuDoubleComplex* d_b = nullptr;
    cuDoubleComplex* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(cuDoubleComplex)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), size_c * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    CHECK_CUBLAS(cublasZgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaDeviceSynchronize());

    // CUDA Events delimitan la unica ejecucion que se usa para calcular el tiempo.
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasZgemm(handle, opA, opB, M, N, K,
                             &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return static_cast<double>(elapsed_ms) / 1000.0;
}

// Muestra la ayuda cuando faltan argumentos o el formato de entrada es incorrecto.
void print_usage(const char* program) {
    std::cerr << "Uso: " << program << " <M> <N> <K> <precision> [opA opB] [matrix_file]" << std::endl;
    std::cerr << "precision: S (float), D (double), C (complex float), Z (complex double)" << std::endl;
    std::cerr << "opA/opB: N (no transpuesta), T (transpuesta), C (conjugada)" << std::endl;
}
// Convierte el caracter de la CLI al enum que cuBLAS espera.

cublasOperation_t parse_op(char op) {
    char upper = static_cast<char>(std::toupper(op));
    if (upper == 'N') {
        return CUBLAS_OP_N;
    }
    if (upper == 'T') {
        return CUBLAS_OP_T;
    }
    if (upper == 'C') {
        return CUBLAS_OP_C;
    }

    std::cerr << "Error: op debe ser N, T o C." << std::endl;
    std::exit(EXIT_FAILURE);
}
// El caso base usa NxN sin transposicion; opA/opB solo se leen si se pasan.

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7 && argc != 8) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    init_seed_from_env();

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    std::string precision = argv[4];
    std::string op_a_str = "N";
    std::string op_b_str = "N";
    const char *matrix_file = nullptr;

    if (argc == 7) {
        op_a_str = argv[5];
        op_b_str = argv[6];
    } else if (argc == 8) {
        op_a_str = argv[5];
        op_b_str = argv[6];
        matrix_file = argv[7];
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "Error: M, N, K deben ser positivos." << std::endl;
        return EXIT_FAILURE;
    }

    if (precision.size() != 1) {
        std::cerr << "Error: precision invalida." << std::endl;
        return EXIT_FAILURE;
    }
    if (op_a_str.size() != 1 || op_b_str.size() != 1) {
        std::cerr << "Error: opA y opB deben ser un caracter (N/T/C)." << std::endl;
        return EXIT_FAILURE;
    }

    if (matrix_file) {
        char file_prec = '\0';
        if (load_gemm_matrices_from_file(matrix_file, &M, &N, &K, &file_prec) != 0) {
            std::cerr << "Error: no se pudo cargar el archivo de matrices." << std::endl;
            return EXIT_FAILURE;
        }
        precision[0] = static_cast<char>(std::toupper(file_prec));
    }
// Normaliza las operaciones y las convierte al formato de cuBLAS.
    
    char opA_char = static_cast<char>(std::toupper(op_a_str[0]));
    char opB_char = static_cast<char>(std::toupper(op_b_str[0]));
    cublasOperation_t opA = parse_op(opA_char);
    cublasOperation_t opB = parse_op(opB_char);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    double time_sec = 0.0;
    char p = static_cast<char>(std::toupper(precision[0]));
// Selecciona la rutina GEMM segun la precision pedida en la CLI.
    
    switch (p) {
        case 'S':
            time_sec = benchmark_sgemm(handle, M, N, K, opA, opB);
            break;
        case 'D':
            time_sec = benchmark_dgemm(handle, M, N, K, opA, opB);
            break;
        case 'C':
            time_sec = benchmark_cgemm(handle, M, N, K, opA, opB);
            break;
        case 'Z':
            time_sec = benchmark_zgemm(handle, M, N, K, opA, opB);
            break;
        default:
            std::cerr << "Error: precision debe ser S, D, C o Z." << std::endl;
            CHECK_CUBLAS(cublasDestroy(handle));
            return EXIT_FAILURE;
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    std::cout << "M=" << M
              << " N=" << N
              << " K=" << K
              << " Precision=" << p
              << " OpA=" << opA_char
              << " OpB=" << opB_char
              << " Time_sec=" << std::fixed << std::setprecision(9) << time_sec
              << std::endl;

    return EXIT_SUCCESS;
}