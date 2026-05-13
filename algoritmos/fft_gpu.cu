// fft_gpu.cu
// Compilar: nvcc -O3 -o fft_gpu fft_gpu.cu -lcufft

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) \
    do { cudaError_t e = (call); if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

#define CHECK_CUFFT(call) \
    do { cufftResult r = (call); if (r != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, r); exit(1); } } while(0)

typedef struct {
    int nx;
    int ny;
    int nz;
    int batch;
    int warmup;
    int iters;
    char precision;
    char domain[4];
    char direction;
    char layout;
    char plan;
} FftConfig;

static int g_fft_loaded_from_file = 0;
static char g_fft_file_precision = '\0';
static char g_fft_file_domain[4] = {0};
static size_t g_fft_input_count = 0;
static float *g_fft_input_f32 = NULL;
static double *g_fft_input_f64 = NULL;
static cufftComplex *g_fft_input_c32 = NULL;
static cufftDoubleComplex *g_fft_input_c64 = NULL;

static void clear_loaded_fft_inputs(void) {
    free(g_fft_input_f32);
    free(g_fft_input_f64);
    free(g_fft_input_c32);
    free(g_fft_input_c64);
    g_fft_input_f32 = NULL;
    g_fft_input_f64 = NULL;
    g_fft_input_c32 = NULL;
    g_fft_input_c64 = NULL;
    g_fft_input_count = 0;
    g_fft_loaded_from_file = 0;
    g_fft_file_precision = '\0';
    g_fft_file_domain[0] = '\0';
}

static int load_fft_from_file(const char *filename, FftConfig *cfg) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        return -1;
    }

    int nx = 0, ny = 0, nz = 0, batch = 0;
    char precision = '\0';
    char domain[4] = {0};

    if (fread(&nx, sizeof(int), 1, f) != 1 ||
        fread(&ny, sizeof(int), 1, f) != 1 ||
        fread(&nz, sizeof(int), 1, f) != 1 ||
        fread(&batch, sizeof(int), 1, f) != 1 ||
        fread(&precision, sizeof(char), 1, f) != 1 ||
        fread(domain, sizeof(char), 3, f) != 3) {
        fclose(f);
        return -1;
    }
    domain[3] = '\0';

    clear_loaded_fft_inputs();
    g_fft_loaded_from_file = 1;
    g_fft_file_precision = precision;
    strncpy(g_fft_file_domain, domain, sizeof(g_fft_file_domain) - 1);
    g_fft_file_domain[sizeof(g_fft_file_domain) - 1] = '\0';

    cfg->nx = nx;
    cfg->ny = ny;
    cfg->nz = nz;
    cfg->batch = batch;
    cfg->precision = precision;
    strncpy(cfg->domain, domain, sizeof(cfg->domain) - 1);
    cfg->domain[sizeof(cfg->domain) - 1] = '\0';

    int dims[3] = {0, 0, 0};
    int rank = setup_dims(cfg, dims);
    size_t nreal = product_dims(rank, dims);
    size_t ncomplex = r2c_complex_elems(rank, dims);

    if (strcmp(domain, "C2C") == 0) {
        g_fft_input_count = ncomplex * (size_t)batch;
    } else if (strcmp(domain, "R2C") == 0) {
        g_fft_input_count = nreal * (size_t)batch;
    } else if (strcmp(domain, "C2R") == 0) {
        g_fft_input_count = ncomplex * (size_t)batch;
    } else {
        fclose(f);
        return -1;
    }

    if (precision == 'S') {
        if (strcmp(domain, "C2C") == 0 || strcmp(domain, "C2R") == 0) {
            g_fft_input_c32 = (cufftComplex *)malloc(sizeof(cufftComplex) * g_fft_input_count);
            if (!g_fft_input_c32) { fclose(f); return -1; }
            for (size_t i = 0; i < g_fft_input_count; ++i) {
                if (fread(&g_fft_input_c32[i].x, sizeof(float), 1, f) != 1 ||
                    fread(&g_fft_input_c32[i].y, sizeof(float), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
            }
        } else {
            g_fft_input_f32 = (float *)malloc(sizeof(float) * g_fft_input_count);
            if (!g_fft_input_f32) { fclose(f); return -1; }
            if (fread(g_fft_input_f32, sizeof(float), g_fft_input_count, f) != g_fft_input_count) {
                fclose(f);
                return -1;
            }
        }
    } else {
        if (strcmp(domain, "C2C") == 0 || strcmp(domain, "C2R") == 0) {
            g_fft_input_c64 = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * g_fft_input_count);
            if (!g_fft_input_c64) { fclose(f); return -1; }
            for (size_t i = 0; i < g_fft_input_count; ++i) {
                if (fread(&g_fft_input_c64[i].x, sizeof(double), 1, f) != 1 ||
                    fread(&g_fft_input_c64[i].y, sizeof(double), 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
            }
        } else {
            g_fft_input_f64 = (double *)malloc(sizeof(double) * g_fft_input_count);
            if (!g_fft_input_f64) { fclose(f); return -1; }
            if (fread(g_fft_input_f64, sizeof(double), g_fft_input_count, f) != g_fft_input_count) {
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}
static int seed_from_env(unsigned int *out_seed) {
    const char *env = getenv("BENCH_SEED");
    if (!env || !*env) {
        return 0;
    }
    char *end = NULL;
    unsigned long val = strtoul(env, &end, 10);
    if (end == env) {
        return 0;
    }
    *out_seed = (unsigned int)val;
    return 1;
}

static void upper_string(char *s) {
    for (; *s; ++s) {
        *s = (char)toupper((unsigned char)*s);
    }
}

static int setup_dims(const FftConfig *cfg, int dims[3]) {
    if (cfg->nz > 0) {
        dims[0] = cfg->nx;
        dims[1] = cfg->ny;
        dims[2] = cfg->nz;
        return 3;
    }
    if (cfg->ny > 0) {
        dims[0] = cfg->nx;
        dims[1] = cfg->ny;
        return 2;
    }
    dims[0] = cfg->nx;
    return 1;
}

static size_t product_dims(int rank, const int dims[3]) {
    size_t total = 1;
    for (int i = 0; i < rank; ++i) {
        total *= (size_t)dims[i];
    }
    return total;
}

static size_t r2c_complex_elems(int rank, const int dims[3]) {
    int last = dims[rank - 1];
    size_t outer = 1;
    for (int i = 0; i < rank - 1; ++i) {
        outer *= (size_t)dims[i];
    }
    return outer * (size_t)(last / 2 + 1);
}

static size_t r2c_real_inplace_elems(int rank, const int dims[3]) {
    int last = dims[rank - 1];
    size_t outer = 1;
    for (int i = 0; i < rank - 1; ++i) {
        outer *= (size_t)dims[i];
    }
    return outer * (size_t)(last + 2);
}

static double sum_log2_dims(int rank, const int dims[3]) {
    double sum = 0.0;
    for (int i = 0; i < rank; ++i) {
        sum += log2((double)dims[i]);
    }
    return sum;
}

static void fill_real_float(float *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buf[i] = (float)rand() / (float)RAND_MAX;
    }
}

static void fill_real_double(double *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buf[i] = (double)rand() / RAND_MAX;
    }
}

static void fill_complex_float(cufftComplex *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buf[i].x = (float)rand() / (float)RAND_MAX;
        buf[i].y = 0.0f;
    }
}

static void fill_complex_double(cufftDoubleComplex *buf, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        buf[i].x = (double)rand() / RAND_MAX;
        buf[i].y = 0.0;
    }
}

static void print_result(const FftConfig *cfg, double time_sec, double gflops) {
    double time_ms = time_sec * 1e3;
    printf(
        "Nx=%d Ny=%d Nz=%d Batch=%d Precision=%c Domain=%s Direction=%c Layout=%c | "
        "tiempo=%.3f ms | GFLOPS=%.2f | Time_sec=%.9f\n",
        cfg->nx,
        cfg->ny,
        cfg->nz,
        cfg->batch,
        cfg->precision,
        cfg->domain,
        cfg->direction,
        cfg->layout,
        time_ms,
        gflops,
        time_sec
    );
}

static void benchmark_fft_float(const FftConfig *cfg) {
    int dims[3] = {0, 0, 0};
    int rank = setup_dims(cfg, dims);
    size_t nreal = product_dims(rank, dims);
    size_t ncomplex = r2c_complex_elems(rank, dims);
    size_t nreal_inplace = r2c_real_inplace_elems(rank, dims);
    size_t total_real = nreal * (size_t)cfg->batch;
    size_t total_complex = ncomplex * (size_t)cfg->batch;
    size_t total_real_inplace = nreal_inplace * (size_t)cfg->batch;

    int warmup = cfg->warmup > 0 ? cfg->warmup : 0;
    int iters = cfg->iters > 0 ? cfg->iters : 0;
    int dir = (cfg->direction == 'I') ? CUFFT_INVERSE : CUFFT_FORWARD;

    double sum_log2 = sum_log2_dims(rank, dims);
    double flops = (strcmp(cfg->domain, "C2C") == 0 ? 5.0 : 2.5) * (double)nreal * sum_log2;

    int n[3] = {dims[0], dims[1], dims[2]};
    int inembed[3] = {0, 0, 0};
    int onembed[3] = {0, 0, 0};
    int *inembed_ptr = NULL;
    int *onembed_ptr = NULL;
    int idist = 0;
    int odist = 0;
    cufftType type;

    if (strcmp(cfg->domain, "C2C") == 0) {
        type = CUFFT_C2C;
        idist = (int)nreal;
        odist = (int)nreal;
    } else if (strcmp(cfg->domain, "R2C") == 0) {
        type = CUFFT_R2C;
        if (cfg->layout == 'I') {
            for (int i = 0; i < rank; ++i) {
                inembed[i] = dims[i];
                onembed[i] = dims[i];
            }
            inembed[rank - 1] = dims[rank - 1] + 2;
            onembed[rank - 1] = dims[rank - 1] / 2 + 1;
            inembed_ptr = inembed;
            onembed_ptr = onembed;
            idist = (int)nreal_inplace;
            odist = (int)ncomplex;
        } else {
            idist = (int)nreal;
            odist = (int)ncomplex;
        }
    } else {
        type = CUFFT_C2R;
        if (cfg->layout == 'I') {
            for (int i = 0; i < rank; ++i) {
                inembed[i] = dims[i];
                onembed[i] = dims[i];
            }
            inembed[rank - 1] = dims[rank - 1] / 2 + 1;
            onembed[rank - 1] = dims[rank - 1] + 2;
            inembed_ptr = inembed;
            onembed_ptr = onembed;
            idist = (int)ncomplex;
            odist = (int)nreal_inplace;
        } else {
            idist = (int)ncomplex;
            odist = (int)nreal;
        }
    }

    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n,
                              inembed_ptr, 1, idist,
                              onembed_ptr, 1, odist,
                              type, cfg->batch));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    if (strcmp(cfg->domain, "C2C") == 0) {
        cufftComplex *h_in = (cufftComplex *)malloc(sizeof(cufftComplex) * total_complex);
        if (g_fft_loaded_from_file && g_fft_input_c32 != NULL) {
            memcpy(h_in, g_fft_input_c32, sizeof(cufftComplex) * total_complex);
        } else {
            fill_complex_float(h_in, total_complex);
        }

        cufftComplex *d_in = NULL;
        cufftComplex *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftComplex) * total_complex));
            d_out = d_in;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftComplex) * total_complex));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftComplex) * total_complex));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(cufftComplex) * total_complex, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, dir));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, dir));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        cudaFree(d_in);
        if (cfg->layout != 'I') {
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    if (strcmp(cfg->domain, "R2C") == 0) {
        float *h_in = (float *)malloc(sizeof(float) * total_real);
        if (g_fft_loaded_from_file && g_fft_input_f32 != NULL) {
            memcpy(h_in, g_fft_input_f32, sizeof(float) * total_real);
            if (cfg->layout == 'I' && total_real_inplace > total_real) {
                memset(h_in + total_real, 0, sizeof(float) * (total_real_inplace - total_real));
            }
        } else {
            fill_real_float(h_in, total_real);
        }

        float *d_in = NULL;
        cufftComplex *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * total_real_inplace));
            d_out = (cufftComplex *)d_in;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * total_real));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftComplex) * total_complex));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(float) * total_real, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        cudaFree(d_in);
        if (cfg->layout != 'I') {
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    if (strcmp(cfg->domain, "C2R") == 0) {
        cufftComplex *h_in = (cufftComplex *)malloc(sizeof(cufftComplex) * total_complex);
        if (g_fft_loaded_from_file && g_fft_input_c32 != NULL) {
            memcpy(h_in, g_fft_input_c32, sizeof(cufftComplex) * total_complex);
            if (cfg->layout == 'I' && total_real_inplace > (size_t)(2 * total_complex)) {
                memset(((float *)out) + (2 * total_complex), 0, sizeof(float) * (total_real_inplace - (2 * total_complex)));
            }
        } else {
            fill_complex_float(h_in, total_complex);
        }

        cufftComplex *d_in = NULL;
        float *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * total_real_inplace));
            d_in = (cufftComplex *)d_out;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftComplex) * total_complex));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * total_real));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(cufftComplex) * total_complex, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecC2R(plan, d_in, d_out));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecC2R(plan, d_in, d_out));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        if (cfg->layout == 'I') {
            cudaFree(d_out);
        } else {
            cudaFree(d_in);
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }
}

static void benchmark_fft_double(const FftConfig *cfg) {
    int dims[3] = {0, 0, 0};
    int rank = setup_dims(cfg, dims);
    size_t nreal = product_dims(rank, dims);
    size_t ncomplex = r2c_complex_elems(rank, dims);
    size_t nreal_inplace = r2c_real_inplace_elems(rank, dims);
    size_t total_real = nreal * (size_t)cfg->batch;
    size_t total_complex = ncomplex * (size_t)cfg->batch;
    size_t total_real_inplace = nreal_inplace * (size_t)cfg->batch;

    int warmup = cfg->warmup > 0 ? cfg->warmup : 0;
    int iters = cfg->iters > 0 ? cfg->iters : 0;
    int dir = (cfg->direction == 'I') ? CUFFT_INVERSE : CUFFT_FORWARD;

    double sum_log2 = sum_log2_dims(rank, dims);
    double flops = (strcmp(cfg->domain, "C2C") == 0 ? 5.0 : 2.5) * (double)nreal * sum_log2;

    int n[3] = {dims[0], dims[1], dims[2]};
    int inembed[3] = {0, 0, 0};
    int onembed[3] = {0, 0, 0};
    int *inembed_ptr = NULL;
    int *onembed_ptr = NULL;
    int idist = 0;
    int odist = 0;
    cufftType type;

    if (strcmp(cfg->domain, "C2C") == 0) {
        type = CUFFT_Z2Z;
        idist = (int)nreal;
        odist = (int)nreal;
    } else if (strcmp(cfg->domain, "R2C") == 0) {
        type = CUFFT_D2Z;
        if (cfg->layout == 'I') {
            for (int i = 0; i < rank; ++i) {
                inembed[i] = dims[i];
                onembed[i] = dims[i];
            }
            inembed[rank - 1] = dims[rank - 1] + 2;
            onembed[rank - 1] = dims[rank - 1] / 2 + 1;
            inembed_ptr = inembed;
            onembed_ptr = onembed;
            idist = (int)nreal_inplace;
            odist = (int)ncomplex;
        } else {
            idist = (int)nreal;
            odist = (int)ncomplex;
        }
    } else {
        type = CUFFT_Z2D;
        if (cfg->layout == 'I') {
            for (int i = 0; i < rank; ++i) {
                inembed[i] = dims[i];
                onembed[i] = dims[i];
            }
            inembed[rank - 1] = dims[rank - 1] / 2 + 1;
            onembed[rank - 1] = dims[rank - 1] + 2;
            inembed_ptr = inembed;
            onembed_ptr = onembed;
            idist = (int)ncomplex;
            odist = (int)nreal_inplace;
        } else {
            idist = (int)ncomplex;
            odist = (int)nreal;
        }
    }

    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(&plan, rank, n,
                              inembed_ptr, 1, idist,
                              onembed_ptr, 1, odist,
                              type, cfg->batch));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    if (strcmp(cfg->domain, "C2C") == 0) {
        cufftDoubleComplex *h_in = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * total_complex);
        if (g_fft_loaded_from_file && g_fft_input_c64 != NULL) {
            memcpy(h_in, g_fft_input_c64, sizeof(cufftDoubleComplex) * total_complex);
        } else {
            fill_complex_double(h_in, total_complex);
        }

        cufftDoubleComplex *d_in = NULL;
        cufftDoubleComplex *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * total_complex));
            d_out = d_in;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * total_complex));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftDoubleComplex) * total_complex));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(cufftDoubleComplex) * total_complex, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecZ2Z(plan, d_in, d_out, dir));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecZ2Z(plan, d_in, d_out, dir));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        cudaFree(d_in);
        if (cfg->layout != 'I') {
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    if (strcmp(cfg->domain, "R2C") == 0) {
        double *h_in = (double *)malloc(sizeof(double) * total_real);
        if (g_fft_loaded_from_file && g_fft_input_f64 != NULL) {
            memcpy(h_in, g_fft_input_f64, sizeof(double) * total_real);
            if (cfg->layout == 'I' && total_real_inplace > total_real) {
                memset(h_in + total_real, 0, sizeof(double) * (total_real_inplace - total_real));
            }
        } else {
            fill_real_double(h_in, total_real);
        }

        double *d_in = NULL;
        cufftDoubleComplex *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(double) * total_real_inplace));
            d_out = (cufftDoubleComplex *)d_in;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(double) * total_real));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftDoubleComplex) * total_complex));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(double) * total_real, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecD2Z(plan, d_in, d_out));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecD2Z(plan, d_in, d_out));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        cudaFree(d_in);
        if (cfg->layout != 'I') {
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    if (strcmp(cfg->domain, "C2R") == 0) {
        cufftDoubleComplex *h_in = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * total_complex);
        if (g_fft_loaded_from_file && g_fft_input_c64 != NULL) {
            memcpy(h_in, g_fft_input_c64, sizeof(cufftDoubleComplex) * total_complex);
            if (cfg->layout == 'I' && total_real_inplace > (size_t)(2 * total_complex)) {
                memset(((double *)out) + (2 * total_complex), 0, sizeof(double) * (total_real_inplace - (2 * total_complex)));
            }
        } else {
            fill_complex_double(h_in, total_complex);
        }

        cufftDoubleComplex *d_in = NULL;
        double *d_out = NULL;

        if (cfg->layout == 'I') {
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(double) * total_real_inplace));
            d_in = (cufftDoubleComplex *)d_out;
        } else {
            CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * total_complex));
            CHECK_CUDA(cudaMalloc(&d_out, sizeof(double) * total_real));
        }

        CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(cufftDoubleComplex) * total_complex, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) {
            CHECK_CUFFT(cufftExecZ2D(plan, d_in, d_out));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        if (iters == 0) {
            print_result(cfg, 0.0, 0.0);
        } else {
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                CHECK_CUFFT(cufftExecZ2D(plan, d_in, d_out));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;
            double time_sec = (double)ms / 1e3;
            double gflops = flops / (time_sec * 1e9);

            print_result(cfg, time_sec, gflops);
        }

        cufftDestroy(plan);
        if (cfg->layout == 'I') {
            cudaFree(d_out);
        } else {
            cudaFree(d_in);
            cudaFree(d_out);
        }
        free(h_in);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }
}

static int parse_config(int argc, char **argv, FftConfig *cfg) {
    cfg->nx = 4096;
    cfg->ny = 0;
    cfg->nz = 0;
    cfg->batch = 1;
    cfg->warmup = 3;
    cfg->iters = 10;
    cfg->precision = 'S';
    strncpy(cfg->domain, "C2C", sizeof(cfg->domain) - 1);
    cfg->domain[sizeof(cfg->domain) - 1] = '\0';
    cfg->direction = 'F';
    cfg->layout = 'O';
    cfg->plan = 'M';

    if (argc == 2) {
        cfg->nx = atoi(argv[1]);
        return 0;
    }

    if (argc < 9) {
        return -1;
    }

    cfg->nx = atoi(argv[1]);
    cfg->ny = atoi(argv[2]);
    cfg->nz = atoi(argv[3]);
    cfg->batch = atoi(argv[4]);
    cfg->precision = argv[5][0];
    strncpy(cfg->domain, argv[6], sizeof(cfg->domain) - 1);
    cfg->domain[sizeof(cfg->domain) - 1] = '\0';
    cfg->direction = argv[7][0];
    cfg->layout = argv[8][0];

    if (argc >= 10) {
        cfg->warmup = atoi(argv[9]);
    }
    if (argc >= 11) {
        cfg->iters = atoi(argv[10]);
    }
    if (argc >= 12) {
        cfg->plan = argv[11][0];
    }

    if (argc >= 13) {
        if (load_fft_from_file(argv[12], cfg) != 0) {
            return -1;
        }
    }

    upper_string(cfg->domain);
    cfg->precision = (char)toupper((unsigned char)cfg->precision);
    cfg->direction = (char)toupper((unsigned char)cfg->direction);
    cfg->layout = (char)toupper((unsigned char)cfg->layout);
    cfg->plan = (char)toupper((unsigned char)cfg->plan);

    return 0;
}

static void print_usage(const char *prog) {
    printf("Uso:\n");
    printf("  %s N\n", prog);
    printf("  %s Nx Ny Nz Batch Precision Domain Direction Layout [Warmup] [Iters] [Plan]\n", prog);
    printf("\nEjemplo:\n");
    printf("  %s 1024 0 0 4 S C2C F I 3 10 E\n", prog);
}

int main(int argc, char **argv) {
    FftConfig cfg;
    if (parse_config(argc, argv, &cfg) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    unsigned int seed = 0;
    if (seed_from_env(&seed)) {
        srand(seed);
    }

    if (cfg.nx <= 0 || cfg.batch <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(cfg.domain, "R2C") == 0 && cfg.direction != 'F') {
        cfg.direction = 'F';
    }
    if (strcmp(cfg.domain, "C2R") == 0 && cfg.direction != 'I') {
        cfg.direction = 'I';
    }

    if (cfg.precision == 'D') {
        benchmark_fft_double(&cfg);
    } else {
        benchmark_fft_float(&cfg);
    }

    clear_loaded_fft_inputs();

    return 0;
}