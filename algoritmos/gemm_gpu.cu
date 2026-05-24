#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define CHECK_CUDA(call)                                                             \
	do {                                                                             \
		cudaError_t cuda_error = (call);                                             \
		if (cuda_error != cudaSuccess) {                                             \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
					cudaGetErrorString(cuda_error));                                  \
			return 1;                                                                \
		}                                                                            \
	} while (0)

#define CHECK_CUBLAS(call)                                                           \
	do {                                                                             \
		cublasStatus_t cublas_status = (call);                                       \
		if (cublas_status != CUBLAS_STATUS_SUCCESS) {                                \
			fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,         \
					(int)cublas_status);                                              \
			return 1;                                                                \
		}                                                                            \
	} while (0)

#define GEMM_WARMUP_RUNS 4
#define GEMM_MEASURE_ITERS 1

typedef struct {
	int m;
	int n;
	int k;
	char precision;
	char op_a;
	char op_b;
	const char *source_path;
	int warmup_runs;
	int iters;
} GemmCli;

typedef struct {
	int m;
	int n;
	int k;
	char precision;
	void *a;
	void *b;
	void *c;
} GemmInput;

static double monotonic_time_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char normalize_precision(char precision) {
	precision = (char)toupper((unsigned char)precision);
	if (precision != 'S' && precision != 'D' && precision != 'C' && precision != 'Z') {
		return '\0';
	}
	return precision;
}

static char normalize_op(char op) {
	op = (char)toupper((unsigned char)op);
	if (op != 'N' && op != 'T' && op != 'C') {
		return '\0';
	}
	return op;
}

static cublasOperation_t to_cublas_op(char op, char precision) {
	op = normalize_op(op);
	if (op == '\0') {
		return CUBLAS_OP_N;
	}
	if (op == 'C' && precision != 'C' && precision != 'Z') {
		return CUBLAS_OP_T;
	}
	if (op == 'N') {
		return CUBLAS_OP_N;
	}
	if (op == 'T') {
		return CUBLAS_OP_T;
	}
	return CUBLAS_OP_C;
}

static void free_gemm_input(GemmInput *input) {
	if (!input) {
		return;
	}
	free(input->a);
	free(input->b);
	free(input->c);
	input->a = NULL;
	input->b = NULL;
	input->c = NULL;
}

static int read_exact(FILE *file, void *ptr, size_t size, size_t count) {
	return fread(ptr, size, count, file) == count ? 0 : -1;
}

static int load_gemm_input_from_file(const char *path, GemmInput *input) {
	FILE *file = fopen(path, "rb");
	if (!file) {
		perror("No se pudo abrir el archivo de matrices");
		return -1;
	}

	int m = 0;
	int n = 0;
	int k = 0;
	char precision = '\0';

	if (read_exact(file, &m, sizeof(int), 1) != 0 ||
		read_exact(file, &n, sizeof(int), 1) != 0 ||
		read_exact(file, &k, sizeof(int), 1) != 0 ||
		read_exact(file, &precision, sizeof(char), 1) != 0) {
		fclose(file);
		fprintf(stderr, "Error al leer el encabezado del archivo de matrices\n");
		return -1;
	}

	precision = normalize_precision(precision);
	if (precision == '\0') {
		fclose(file);
		fprintf(stderr, "Precision invalida en el archivo de matrices\n");
		return -1;
	}

	size_t a_count = (size_t)m * (size_t)k;
	size_t b_count = (size_t)k * (size_t)n;
	size_t c_count = (size_t)m * (size_t)n;

	input->m = m;
	input->n = n;
	input->k = k;
	input->precision = precision;
	input->a = NULL;
	input->b = NULL;
	input->c = NULL;

	if (precision == 'S') {
		input->a = malloc(a_count * sizeof(float));
		input->b = malloc(b_count * sizeof(float));
		input->c = malloc(c_count * sizeof(float));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}
		if (read_exact(file, input->a, sizeof(float), a_count) != 0 ||
			read_exact(file, input->b, sizeof(float), b_count) != 0 ||
			read_exact(file, input->c, sizeof(float), c_count) != 0) {
			fclose(file);
			free_gemm_input(input);
			fprintf(stderr, "Error al leer datos float del archivo de matrices\n");
			return -1;
		}
	} else if (precision == 'D') {
		input->a = malloc(a_count * sizeof(double));
		input->b = malloc(b_count * sizeof(double));
		input->c = malloc(c_count * sizeof(double));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}
		if (read_exact(file, input->a, sizeof(double), a_count) != 0 ||
			read_exact(file, input->b, sizeof(double), b_count) != 0 ||
			read_exact(file, input->c, sizeof(double), c_count) != 0) {
			fclose(file);
			free_gemm_input(input);
			fprintf(stderr, "Error al leer datos double del archivo de matrices\n");
			return -1;
		}
	} else if (precision == 'C') {
		input->a = malloc(a_count * sizeof(cuComplex));
		input->b = malloc(b_count * sizeof(cuComplex));
		input->c = malloc(c_count * sizeof(cuComplex));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}

		cuComplex *a = (cuComplex *)input->a;
		cuComplex *b = (cuComplex *)input->b;
		cuComplex *c = (cuComplex *)input->c;
		for (size_t i = 0; i < a_count; ++i) {
			if (read_exact(file, &a[i].x, sizeof(float), 1) != 0 ||
				read_exact(file, &a[i].y, sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de A\n");
				return -1;
			}
		}
		for (size_t i = 0; i < b_count; ++i) {
			if (read_exact(file, &b[i].x, sizeof(float), 1) != 0 ||
				read_exact(file, &b[i].y, sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de B\n");
				return -1;
			}
		}
		for (size_t i = 0; i < c_count; ++i) {
			if (read_exact(file, &c[i].x, sizeof(float), 1) != 0 ||
				read_exact(file, &c[i].y, sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de C\n");
				return -1;
			}
		}
	} else {
		input->a = malloc(a_count * sizeof(cuDoubleComplex));
		input->b = malloc(b_count * sizeof(cuDoubleComplex));
		input->c = malloc(c_count * sizeof(cuDoubleComplex));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}

		cuDoubleComplex *a = (cuDoubleComplex *)input->a;
		cuDoubleComplex *b = (cuDoubleComplex *)input->b;
		cuDoubleComplex *c = (cuDoubleComplex *)input->c;
		for (size_t i = 0; i < a_count; ++i) {
			if (read_exact(file, &a[i].x, sizeof(double), 1) != 0 ||
				read_exact(file, &a[i].y, sizeof(double), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos dobles de A\n");
				return -1;
			}
		}
		for (size_t i = 0; i < b_count; ++i) {
			if (read_exact(file, &b[i].x, sizeof(double), 1) != 0 ||
				read_exact(file, &b[i].y, sizeof(double), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos dobles de B\n");
				return -1;
			}
		}
		for (size_t i = 0; i < c_count; ++i) {
			if (read_exact(file, &c[i].x, sizeof(double), 1) != 0 ||
				read_exact(file, &c[i].y, sizeof(double), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos dobles de C\n");
				return -1;
			}
		}
	}

	fclose(file);
	return 0;
}

static int parse_function_name(const char *name, char *out_precision) {
	if (!name || !out_precision) {
		return -1;
	}

	if (strcasecmp(name, "sgemm") == 0 || strcasecmp(name, "s") == 0) {
		*out_precision = 'S';
		return 0;
	}
	if (strcasecmp(name, "dgemm") == 0 || strcasecmp(name, "d") == 0) {
		*out_precision = 'D';
		return 0;
	}
	if (strcasecmp(name, "cgemm") == 0 || strcasecmp(name, "c") == 0) {
		*out_precision = 'C';
		return 0;
	}
	if (strcasecmp(name, "zgemm") == 0 || strcasecmp(name, "z") == 0) {
		*out_precision = 'Z';
		return 0;
	}

	return -1;
}

static int parse_cli(int argc, char **argv, GemmCli *cli) {
	memset(cli, 0, sizeof(*cli));
	cli->precision = '\0';
	cli->op_a = 'N';
	cli->op_b = 'N';
	cli->warmup_runs = GEMM_WARMUP_RUNS;
	cli->iters = GEMM_MEASURE_ITERS;

	const char *positionals[8];
	int positional_count = 0;

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			return 2;
		}
		if (strcmp(argv[i], "--function") == 0 && i + 1 < argc) {
			if (parse_function_name(argv[++i], &cli->precision) != 0) {
				fprintf(stderr, "Funcion invalida: %s\n", argv[i]);
				return -1;
			}
			continue;
		}
		if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
			cli->precision = normalize_precision(argv[++i][0]);
			if (cli->precision == '\0') {
				fprintf(stderr, "Precision invalida: %s\n", argv[i]);
				return -1;
			}
			continue;
		}
		if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
			cli->m = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
			cli->n = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
			cli->k = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
			int size = atoi(argv[++i]);
			cli->m = size;
			cli->n = size;
			cli->k = size;
			continue;
		}
		if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
			cli->source_path = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "--matrix-file") == 0 && i + 1 < argc) {
			cli->source_path = argv[++i];
			continue;
		}
		if (strcmp(argv[i], "--op-a") == 0 && i + 1 < argc) {
			cli->op_a = normalize_op(argv[++i][0]);
			if (cli->op_a == '\0') {
				fprintf(stderr, "OpA invalida: %s\n", argv[i]);
				return -1;
			}
			continue;
		}
		if (strcmp(argv[i], "--op-b") == 0 && i + 1 < argc) {
			cli->op_b = normalize_op(argv[++i][0]);
			if (cli->op_b == '\0') {
				fprintf(stderr, "OpB invalida: %s\n", argv[i]);
				return -1;
			}
			continue;
		}
		if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
			cli->warmup_runs = atoi(argv[++i]);
			continue;
		}
		if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
			cli->iters = atoi(argv[++i]);
			continue;
		}

		if (argv[i][0] == '-' && argv[i][1] == '-') {
			fprintf(stderr, "Opcion desconocida: %s\n", argv[i]);
			return -1;
		}

		if (positional_count < 8) {
			positionals[positional_count++] = argv[i];
		}
	}

	if (positional_count > 0) {
		if (positional_count < 4) {
			fprintf(stderr, "Uso legacy: %s M N K <S|D|C|Z> [OpA] [OpB] [matrix_file]\n", argv[0]);
			return -1;
		}
		cli->m = atoi(positionals[0]);
		cli->n = atoi(positionals[1]);
		cli->k = atoi(positionals[2]);
		cli->precision = normalize_precision(positionals[3][0]);
		if (cli->precision == '\0') {
			fprintf(stderr, "Precision invalida: %s\n", positionals[3]);
			return -1;
		}
		if (positional_count >= 5) {
			cli->op_a = normalize_op(positionals[4][0]);
			if (cli->op_a == '\0') {
				fprintf(stderr, "OpA invalida: %s\n", positionals[4]);
				return -1;
			}
		}
		if (positional_count >= 6) {
			cli->op_b = normalize_op(positionals[5][0]);
			if (cli->op_b == '\0') {
				fprintf(stderr, "OpB invalida: %s\n", positionals[5]);
				return -1;
			}
		}
		if (positional_count >= 7) {
			cli->source_path = positionals[6];
		}
	}

	if (cli->m <= 0 || cli->n <= 0 || cli->k <= 0) {
		fprintf(stderr, "Las dimensiones M, N y K deben ser positivas\n");
		return -1;
	}
	if (cli->precision == '\0') {
		fprintf(stderr, "Debes indicar la precision o la funcion GEMM\n");
		return -1;
	}
	if (!cli->source_path || !*cli->source_path) {
		fprintf(stderr, "Debes indicar el archivo de origen con --source o como argumento final\n");
		return -1;
	}
	if (cli->warmup_runs < 0 || cli->iters <= 0) {
		fprintf(stderr, "Warmup e iters deben ser positivos\n");
		return -1;
	}

	return 0;
}

static int run_sgemm_case(
	cublasHandle_t handle,
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	float *h_a = (float *)input->a;
	float *h_b = (float *)input->b;
	float *h_c = (float *)input->c;
	float *d_a = NULL;
	float *d_b = NULL;
	float *d_c = NULL;
	size_t a_bytes = (size_t)input->m * (size_t)input->k * sizeof(float);
	size_t b_bytes = (size_t)input->k * (size_t)input->n * sizeof(float);
	size_t c_bytes = (size_t)input->m * (size_t)input->n * sizeof(float);
	cublasOperation_t c_op_a = to_cublas_op(op_a, input->precision);
	cublasOperation_t c_op_b = to_cublas_op(op_b, input->precision);
	const float alpha = 1.0f;
	const float beta = 0.0f;

	CHECK_CUDA(cudaMalloc((void **)&d_a, a_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_b, b_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_c, c_bytes));

	for (int i = 0; i < warmup_runs; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasSgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}

	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasSgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

static int run_dgemm_case(
	cublasHandle_t handle,
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	double *h_a = (double *)input->a;
	double *h_b = (double *)input->b;
	double *h_c = (double *)input->c;
	double *d_a = NULL;
	double *d_b = NULL;
	double *d_c = NULL;
	size_t a_bytes = (size_t)input->m * (size_t)input->k * sizeof(double);
	size_t b_bytes = (size_t)input->k * (size_t)input->n * sizeof(double);
	size_t c_bytes = (size_t)input->m * (size_t)input->n * sizeof(double);
	cublasOperation_t c_op_a = to_cublas_op(op_a, input->precision);
	cublasOperation_t c_op_b = to_cublas_op(op_b, input->precision);
	const double alpha = 1.0;
	const double beta = 0.0;

	CHECK_CUDA(cudaMalloc((void **)&d_a, a_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_b, b_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_c, c_bytes));

	for (int i = 0; i < warmup_runs; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasDgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}

	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasDgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

static int run_cgemm_case(
	cublasHandle_t handle,
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	cuComplex *h_a = (cuComplex *)input->a;
	cuComplex *h_b = (cuComplex *)input->b;
	cuComplex *h_c = (cuComplex *)input->c;
	cuComplex *d_a = NULL;
	cuComplex *d_b = NULL;
	cuComplex *d_c = NULL;
	size_t a_bytes = (size_t)input->m * (size_t)input->k * sizeof(cuComplex);
	size_t b_bytes = (size_t)input->k * (size_t)input->n * sizeof(cuComplex);
	size_t c_bytes = (size_t)input->m * (size_t)input->n * sizeof(cuComplex);
	cublasOperation_t c_op_a = to_cublas_op(op_a, input->precision);
	cublasOperation_t c_op_b = to_cublas_op(op_b, input->precision);
	const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
	const cuComplex beta = make_cuComplex(0.0f, 0.0f);

	CHECK_CUDA(cudaMalloc((void **)&d_a, a_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_b, b_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_c, c_bytes));

	for (int i = 0; i < warmup_runs; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasCgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}

	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasCgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

static int run_zgemm_case(
	cublasHandle_t handle,
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	cuDoubleComplex *h_a = (cuDoubleComplex *)input->a;
	cuDoubleComplex *h_b = (cuDoubleComplex *)input->b;
	cuDoubleComplex *h_c = (cuDoubleComplex *)input->c;
	cuDoubleComplex *d_a = NULL;
	cuDoubleComplex *d_b = NULL;
	cuDoubleComplex *d_c = NULL;
	size_t a_bytes = (size_t)input->m * (size_t)input->k * sizeof(cuDoubleComplex);
	size_t b_bytes = (size_t)input->k * (size_t)input->n * sizeof(cuDoubleComplex);
	size_t c_bytes = (size_t)input->m * (size_t)input->n * sizeof(cuDoubleComplex);
	cublasOperation_t c_op_a = to_cublas_op(op_a, input->precision);
	cublasOperation_t c_op_b = to_cublas_op(op_b, input->precision);
	const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
	const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

	CHECK_CUDA(cudaMalloc((void **)&d_a, a_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_b, b_bytes));
	CHECK_CUDA(cudaMalloc((void **)&d_c, c_bytes));

	for (int i = 0; i < warmup_runs; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasZgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}

	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		CHECK_CUDA(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasZgemm(handle, c_op_b, c_op_a, input->n, input->m, input->k,
								 &alpha, d_b, input->n, d_a, input->k, &beta, d_c, input->n));
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

static int print_usage(const char *program) {
	fprintf(stderr, "Uso legacy: %s M N K <S|D|C|Z> [OpA] [OpB] [matrix_file]\n", program);
	fprintf(stderr, "Modo flags: %s --m M --n N --k K --precision S --op-a N --op-b N --source matrices.bin\n", program);
	fprintf(stderr, "Alias de funcion: --function sgemm|dgemm|cgemm|zgemm\n");
	return 2;
}

int main(int argc, char **argv) {
	GemmCli cli;
	int parse_rc = parse_cli(argc, argv, &cli);
	if (parse_rc == 2) {
		return print_usage(argv[0]);
	}
	if (parse_rc != 0) {
		return print_usage(argv[0]);
	}

	GemmInput input;
	if (load_gemm_input_from_file(cli.source_path, &input) != 0) {
		return 1;
	}

	if (cli.m != input.m || cli.n != input.n || cli.k != input.k) {
		fprintf(stderr, "Aviso: las dimensiones del archivo de matrices sobreescriben M/N/K del CLI\n");
	}
	if (cli.precision != input.precision) {
		fprintf(stderr, "Aviso: la precision del archivo de matrices sobreescribe la del CLI\n");
	}

	cublasHandle_t handle = NULL;
	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUBLAS(cublasCreate(&handle));

	double time_sec = 0.0;
	int rc = 0;

	switch (input.precision) {
		case 'S':
			rc = run_sgemm_case(handle, &input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'D':
			rc = run_dgemm_case(handle, &input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'C':
			rc = run_cgemm_case(handle, &input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'Z':
			rc = run_zgemm_case(handle, &input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		default:
			fprintf(stderr, "Precision invalida: %c\n", input.precision);
			rc = 1;
			break;
	}

	int final_m = input.m;
	int final_n = input.n;
	int final_k = input.k;
	char final_precision = input.precision;
	char final_op_a = cli.op_a;
	char final_op_b = cli.op_b;

	if (handle) {
		cublasDestroy(handle);
	}
	free_gemm_input(&input);

	if (rc != 0) {
		return 1;
	}

	printf("M=%d N=%d K=%d Precision=%c OpA=%c OpB=%c Time_sec=%.9f\n",
		   final_m,
		   final_n,
		   final_k,
		   final_precision,
		   final_op_a,
		   final_op_b,
		   time_sec);
	return 0;
}
