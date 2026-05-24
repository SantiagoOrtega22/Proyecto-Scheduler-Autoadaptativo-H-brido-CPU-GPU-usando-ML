/**
 * @file gemm_cpu.c
 * @brief Benchmark implementation for GEMM (General Matrix Multiplication) on CPU.
 *
 * This file contains the implementation of a C-based benchmark that leverages
 * BLAS (e.g., OpenBLAS or Intel MKL) to execute General Matrix Multiplication (GEMM)
 * for different data types and precisions:
 *   - SGEMM (Single Precision Real Float)
 *   - DGEMM (Double Precision Real Float)
 *   - CGEMM (Single Precision Complex Float)
 *   - ZGEMM (Double Precision Complex Float)
 *
 * The benchmark enforces rigorous HPC measurement protocols, including:
 *   - Warm-up executions to stabilize CPU caches and execution pipelines.
 *   - Clean resource management to isolate execution costs.
 *
 * Compilation instructions (OpenBLAS):
 *   gcc -O3 -march=native -o algoritmos/gemm_cpu algoritmos/gemm_cpu.c -I/usr/include/openblas -lopenblas -lm
 *
 * Compilation instructions (MKL):
 *   gcc -O3 -march=native -o algoritmos/gemm_cpu algoritmos/gemm_cpu.c -lmkl_rt -lpthread -lm
 *
 * Usage:
 *   ./algoritmos/gemm_cpu --m <M> --n <N> --k <K> --precision <S|D|C|Z> --source <matrix_file>
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <time.h>
#include <cblas.h>
#include <complex.h>

#define GEMM_WARMUP_RUNS 4
#define GEMM_MEASURE_ITERS 1

/**
 * @struct GemmCli
 * @brief Configuration parameters parsed from command-line arguments.
 *
 * Holds all user-specified benchmark execution choices, including matrix
 * dimensions, precision types, transposition settings, and execution options.
 */
typedef struct {
	int m;                  /**< Row count of matrices A and C. */
	int n;                  /**< Column count of matrices B and C. */
	int k;                  /**< Column count of matrix A and row count of matrix B. */
	char precision;         /**< Precision code ('S', 'D', 'C', 'Z'). */
	char op_a;              /**< Matrix A transpose setting ('N', 'T', 'C'). */
	char op_b;              /**< Matrix B transpose setting ('N', 'T', 'C'). */
	const char *source_path;/**< Binary file path to load matrix data from. */
	int warmup_runs;        /**< Number of warm-up iterations to run. */
	int iters;              /**< Number of measured timing iterations. */
} GemmCli;

/**
 * @struct GemmInput
 * @brief Memory representation of the loaded input matrices on the host.
 *
 * Contains matrix sizes, precision identifier, and pointers to the raw matrix
 * data buffers allocated in host memory.
 */
typedef struct {
	int m;                  /**< Matrix dimension M. */
	int n;                  /**< Matrix dimension N. */
	int k;                  /**< Matrix dimension K. */
	char precision;         /**< Loaded precision code ('S', 'D', 'C', 'Z'). */
	void *a;                /**< Host memory pointer for matrix A. */
	void *b;                /**< Host memory pointer for matrix B. */
	void *c;                /**< Host memory pointer for matrix C. */
} GemmInput;

/**
 * Retrieves the current monotonic system time in seconds.
 *
 * Uses the POSIX high-resolution CLOCK_MONOTONIC timer to acquire accurate
 * host-side timings.
 *
 * Args:
 *     None.
 *
 * Returns:
 *     double: Monotonic time in seconds.
 */
static double monotonic_time_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/**
 * Validates and normalizes the precision character to uppercase.
 *
 * Supported character codes:
 *   - 'S': Single precision float.
 *   - 'D': Double precision float.
 *   - 'C': Complex single precision float.
 *   - 'Z': Complex double precision float.
 *
 * Args:
 *     precision (char): Raw precision code from command-line.
 *
 * Returns:
 *     char: Uppercase validated code, or '\0' if invalid.
 */
static char normalize_precision(char precision) {
	precision = (char)toupper((unsigned char)precision);
	if (precision != 'S' && precision != 'D' && precision != 'C' && precision != 'Z') {
		return '\0';
	}
	return precision;
}

/**
 * Validates and normalizes the matrix transposition operation code.
 *
 * Supported character codes:
 *   - 'N': Normal (no transpose).
 *   - 'T': Transpose.
 *   - 'C': Conjugate transpose.
 *
 * Args:
 *     op (char): Raw operation character code.
 *
 * Returns:
 *     char: Uppercase validated operation, or '\0' if invalid.
 */
static char normalize_op(char op) {
	op = (char)toupper((unsigned char)op);
	if (op != 'N' && op != 'T' && op != 'C') {
		return '\0';
	}
	return op;
}

/**
 * Maps the operation and precision character to the corresponding CBLAS enum.
 *
 * Real floating point precisions ('S', 'D') do not contain imaginary components,
 * meaning a conjugate transpose ('C') is mapped to standard transpose (CblasTrans).
 *
 * Args:
 *     op (char): Matrix transposition operation setting ('N', 'T', 'C').
 *     precision (char): Matrix precision identifier ('S', 'D', 'C', 'Z').
 *
 * Returns:
 *     enum CBLAS_TRANSPOSE: Associated CBLAS operation enum type.
 */
static enum CBLAS_TRANSPOSE to_cblas_op(char op, char precision) {
	op = normalize_op(op);
	if (op == '\0') {
		return CblasNoTrans;
	}
	if (op == 'C' && precision != 'C' && precision != 'Z') {
		return CblasTrans;
	}
	if (op == 'N') {
		return CblasNoTrans;
	}
	if (op == 'T') {
		return CblasTrans;
	}
	return CblasConjTrans;
}

/**
 * Frees host memory allocations inside a GemmInput structure.
 *
 * Deallocates host buffers for matrices A, B, and C, and nullifies pointers to
 * prevent dangling references.
 *
 * Args:
 *     input (GemmInput *): Pointer to the input structure containing allocations.
 *
 * Returns:
 *     void.
 */
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

/**
 * Reads an exact count of items from the binary matrix file.
 *
 * Wrapper around standard fread to guarantee that all requested elements are
 * successfully read before returning.
 *
 * Args:
 *     file (FILE *): Pointer to file stream.
 *     ptr (void *): Memory buffer to write data to.
 *     size (size_t): Byte size of each element.
 *     count (size_t): Number of elements to read.
 *
 * Returns:
 *     int: 0 on success, -1 if the exact element count cannot be read.
 */
static int read_exact(FILE *file, void *ptr, size_t size, size_t count) {
	return fread(ptr, size, count, file) == count ? 0 : -1;
}

/**
 * Parses and loads the GEMM dimensions and data from a binary input file.
 *
 * Reads header specifications (M, N, K, and precision), validates metadata,
 * allocates appropriate host-side arrays, and streams the binary matrices into
 * memory.
 *
 * Args:
 *     path (const char *): Path to the matrix binary file.
 *     input (GemmInput *): Pointer to structural container to hold dimensions and pointers.
 *
 * Returns:
 *     int: 0 on success, -1 on allocation, validation, or stream read error.
 */
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
		input->a = malloc(a_count * 2 * sizeof(float));
		input->b = malloc(b_count * 2 * sizeof(float));
		input->c = malloc(c_count * 2 * sizeof(float));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}

		float *a = (float *)input->a;
		float *b = (float *)input->b;
		float *c = (float *)input->c;
		for (size_t i = 0; i < a_count; ++i) {
			if (read_exact(file, &a[2*i], sizeof(float), 1) != 0 ||
				read_exact(file, &a[2*i+1], sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de A\n");
				return -1;
			}
		}
		for (size_t i = 0; i < b_count; ++i) {
			if (read_exact(file, &b[2*i], sizeof(float), 1) != 0 ||
				read_exact(file, &b[2*i+1], sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de B\n");
				return -1;
			}
		}
		for (size_t i = 0; i < c_count; ++i) {
			if (read_exact(file, &c[2*i], sizeof(float), 1) != 0 ||
				read_exact(file, &c[2*i+1], sizeof(float), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos simples de C\n");
				return -1;
			}
		}
	} else {
		input->a = malloc(a_count * 2 * sizeof(double));
		input->b = malloc(b_count * 2 * sizeof(double));
		input->c = malloc(c_count * 2 * sizeof(double));
		if (!input->a || !input->b || !input->c) {
			fclose(file);
			free_gemm_input(input);
			return -1;
		}

		double *a = (double *)input->a;
		double *b = (double *)input->b;
		double *c = (double *)input->c;
		for (size_t i = 0; i < a_count; ++i) {
			if (read_exact(file, &a[2*i], sizeof(double), 1) != 0 ||
				read_exact(file, &a[2*i+1], sizeof(double), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos dobles de A\n");
				return -1;
			}
		}
		for (size_t i = 0; i < b_count; ++i) {
			if (read_exact(file, &b[2*i], sizeof(double), 1) != 0 ||
				read_exact(file, &b[2*i+1], sizeof(double), 1) != 0) {
				fclose(file);
				free_gemm_input(input);
				fprintf(stderr, "Error al leer datos complejos dobles de B\n");
				return -1;
			}
		}
		for (size_t i = 0; i < c_count; ++i) {
			if (read_exact(file, &c[2*i], sizeof(double), 1) != 0 ||
				read_exact(file, &c[2*i+1], sizeof(double), 1) != 0) {
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

/**
 * Parses the precision character from a GEMM variation name.
 *
 * Maps function names like "sgemm" or prefix flags to their uppercase precision char.
 *
 * Args:
 *     name (const char *): GEMM function name string.
 *     out_precision (char *): Pointer to output character destination.
 *
 * Returns:
 *     int: 0 on success, -1 on invalid name.
 */
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

/**
 * Parses CLI configuration parameters.
 *
 * Orchestrates parsing flag options and legacy positional options into the
 * config object.
 *
 * Args:
 *     argc (int): Argument count.
 *     argv (char **): Command line arguments.
 *     cli (GemmCli *): CLI config target structure.
 *
 * Returns:
 *     int: 0 on success, 2 if help was invoked, -1 on format errors.
 */
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

/**
 * Runs a single-precision floating-point GEMM (SGEMM) benchmark on the CPU.
 *
 * Executes warm-up runs to stabilize caches and CPU frequency scaling,
 * measures execution times, and reports average run-time.
 *
 * Args:
 *     input (const GemmInput *): Loaded host matrix containers.
 *     op_a (char): Transpose setting for A.
 *     op_b (char): Transpose setting for B.
 *     warmup_runs (int): Number of stabilization iterations.
 *     iters (int): Number of measured iterations.
 *     out_time_sec (double *): Output pointer to store average run-time.
 *
 * Returns:
 *     int: 0 on success.
 */
static int run_sgemm_case(
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	float *A = (float *)input->a;
	float *B = (float *)input->b;
	float *C = (float *)input->c;
	
	enum CBLAS_TRANSPOSE transA = to_cblas_op(op_a, input->precision);
	enum CBLAS_TRANSPOSE transB = to_cblas_op(op_b, input->precision);
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	int lda = input->k;
	int ldb = input->n;
	int ldc = input->n;

	// HPC Rigor: Warm-up phase to stabilize caches and CPU frequency
	for (int i = 0; i < warmup_runs; ++i) {
		cblas_sgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					alpha, A, lda, B, ldb, beta, C, ldc);
	}

	// Timed execution phase
	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		cblas_sgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					alpha, A, lda, B, ldb, beta, C, ldc);
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;
	return 0;
}

/**
 * Runs a double-precision floating-point GEMM (DGEMM) benchmark on the CPU.
 *
 * Executes warm-up runs to stabilize caches and CPU frequency scaling,
 * measures execution times, and reports average run-time.
 *
 * Args:
 *     input (const GemmInput *): Loaded host matrix containers.
 *     op_a (char): Transpose setting for A.
 *     op_b (char): Transpose setting for B.
 *     warmup_runs (int): Number of stabilization iterations.
 *     iters (int): Number of measured iterations.
 *     out_time_sec (double *): Output pointer to store average run-time.
 *
 * Returns:
 *     int: 0 on success.
 */
static int run_dgemm_case(
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	double *A = (double *)input->a;
	double *B = (double *)input->b;
	double *C = (double *)input->c;
	
	enum CBLAS_TRANSPOSE transA = to_cblas_op(op_a, input->precision);
	enum CBLAS_TRANSPOSE transB = to_cblas_op(op_b, input->precision);
	
	const double alpha = 1.0;
	const double beta = 0.0;
	int lda = input->k;
	int ldb = input->n;
	int ldc = input->n;

	// HPC Rigor: Warm-up phase to stabilize caches and CPU frequency
	for (int i = 0; i < warmup_runs; ++i) {
		cblas_dgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					alpha, A, lda, B, ldb, beta, C, ldc);
	}

	// Timed execution phase
	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		cblas_dgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					alpha, A, lda, B, ldb, beta, C, ldc);
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;
	return 0;
}

/**
 * Runs a single-precision complex GEMM (CGEMM) benchmark on the CPU.
 *
 * Executes warm-up runs to stabilize caches and CPU frequency scaling,
 * measures execution times, and reports average run-time.
 *
 * Args:
 *     input (const GemmInput *): Loaded host matrix containers.
 *     op_a (char): Transpose setting for A.
 *     op_b (char): Transpose setting for B.
 *     warmup_runs (int): Number of stabilization iterations.
 *     iters (int): Number of measured iterations.
 *     out_time_sec (double *): Output pointer to store average run-time.
 *
 * Returns:
 *     int: 0 on success.
 */
static int run_cgemm_case(
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	void *A = input->a;
	void *B = input->b;
	void *C = input->c;
	
	enum CBLAS_TRANSPOSE transA = to_cblas_op(op_a, input->precision);
	enum CBLAS_TRANSPOSE transB = to_cblas_op(op_b, input->precision);
	
	float complex alpha = 1.0f + 0.0f * I;
	float complex beta = 0.0f + 0.0f * I;
	int lda = input->k;
	int ldb = input->n;
	int ldc = input->n;

	// HPC Rigor: Warm-up phase to stabilize caches and CPU frequency
	for (int i = 0; i < warmup_runs; ++i) {
		cblas_cgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					&alpha, A, lda, B, ldb, &beta, C, ldc);
	}

	// Timed execution phase
	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		cblas_cgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					&alpha, A, lda, B, ldb, &beta, C, ldc);
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;
	return 0;
}

/**
 * Runs a double-precision complex GEMM (ZGEMM) benchmark on the CPU.
 *
 * Executes warm-up runs to stabilize caches and CPU frequency scaling,
 * measures execution times, and reports average run-time.
 *
 * Args:
 *     input (const GemmInput *): Loaded host matrix containers.
 *     op_a (char): Transpose setting for A.
 *     op_b (char): Transpose setting for B.
 *     warmup_runs (int): Number of stabilization iterations.
 *     iters (int): Number of measured iterations.
 *     out_time_sec (double *): Output pointer to store average run-time.
 *
 * Returns:
 *     int: 0 on success.
 */
static int run_zgemm_case(
	const GemmInput *input,
	char op_a,
	char op_b,
	int warmup_runs,
	int iters,
	double *out_time_sec) {
	void *A = input->a;
	void *B = input->b;
	void *C = input->c;
	
	enum CBLAS_TRANSPOSE transA = to_cblas_op(op_a, input->precision);
	enum CBLAS_TRANSPOSE transB = to_cblas_op(op_b, input->precision);
	
	double complex alpha = 1.0 + 0.0 * I;
	double complex beta = 0.0 + 0.0 * I;
	int lda = input->k;
	int ldb = input->n;
	int ldc = input->n;

	// HPC Rigor: Warm-up phase to stabilize caches and CPU frequency
	for (int i = 0; i < warmup_runs; ++i) {
		cblas_zgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					&alpha, A, lda, B, ldb, &beta, C, ldc);
	}

	// Timed execution phase
	double start = monotonic_time_sec();
	for (int i = 0; i < iters; ++i) {
		cblas_zgemm(CblasRowMajor, transA, transB, input->m, input->n, input->k, 
					&alpha, A, lda, B, ldb, &beta, C, ldc);
	}
	double end = monotonic_time_sec();

	*out_time_sec = (end - start) / (double)iters;
	return 0;
}

/**
 * Prints the program's usage guidelines to standard error.
 *
 * Explains positional parameter inputs (legacy mode), option flags,
 * and helper function aliases.
 *
 * Args:
 *     program (const char *): Executable name (typically argv[0]).
 *
 * Returns:
 *     int: Exit code (returns 2).
 */
static int print_usage(const char *program) {
	fprintf(stderr, "Uso legacy: %s M N K <S|D|C|Z> [OpA] [OpB] [matrix_file]\n", program);
	fprintf(stderr, "Modo flags: %s --m M --n N --k K --precision S --op-a N --op-b N --source matrices.bin\n", program);
	fprintf(stderr, "Alias de funcion: --function sgemm|dgemm|cgemm|zgemm\n");
	return 2;
}

/**
 * Main benchmark execution orchestrator.
 *
 * Parses arguments, loads matrices, invokes the corresponding GEMM runner,
 * and prints output statistics.
 *
 * Args:
 *     argc (int): Number of command line arguments.
 *     argv (char **): Array of command line argument strings.
 *
 * Returns:
 *     int: 0 on success; 1 on validation or loading failure; 2 on usage errors.
 */
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

	double time_sec = 0.0;
	int rc = 0;

	// Execute corresponding GEMM kernel based on the loaded matrix precision
	switch (input.precision) {
		case 'S':
			rc = run_sgemm_case(&input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'D':
			rc = run_dgemm_case(&input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'C':
			rc = run_cgemm_case(&input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
			break;
		case 'Z':
			rc = run_zgemm_case(&input, cli.op_a, cli.op_b, cli.warmup_runs, cli.iters, &time_sec);
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

	// HPC Rigor: Free host-allocated matrix arrays
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
