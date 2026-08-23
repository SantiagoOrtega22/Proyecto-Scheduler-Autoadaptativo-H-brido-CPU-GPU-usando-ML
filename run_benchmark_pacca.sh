#!/bin/bash
#SBATCH --job-name=gemm_fft_rl_bench
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:1
#SBATCH --output=benchmark_hpc_%j.log
#SBATCH --error=benchmark_hpc_%j.err

echo "================================================================="
echo "Iniciando Job de Benchmarking en Clúster HPC paccaA100"
echo "Fecha de inicio: $(date)"
echo "Nodo asignado: $(hostname)"
echo "================================================================="

# 1. Limpiar y Cargar Módulos Requeridos
echo "[1/4] Cargando módulos HPC..."
module purge
module load gnu12/12.4.0
module load devtools/nvidia/hpc_sdk/nvhpc/23.1
module load devtools/intel/oneapi/2023
module load Analytics/anaconda3/python3

# 2. Activar Entorno Virtual
echo "[2/4] Activando entorno Conda hpc-gemm..."
source /opt/ohpc/pub/Analytics/anaconda3/etc/profile.d/conda.sh || true
conda activate hpc-gemm

# 3. Compilación de Binarios (con resolución de conflicto libgomp NVIDIA)
echo "[3/4] Compilando algoritmos GEMM y FFT..."

# GEMM GPU
echo "      -> Compilando GEMM GPU..."
nvcc -O3 -o algoritmos/gemm_gpu algoritmos/gemm_gpu.cu -lcublas

# FFT GPU
echo "      -> Compilando FFT GPU..."
nvcc -O3 -o algoritmos/fft_gpu algoritmos/fft_gpu.cu -lcufft

# === TRUCO PARA COMPILAR CPU SIN CONFLICTO LIBGOMP ===
module unload devtools/nvidia/hpc_sdk/nvhpc/23.1

# GEMM CPU
echo "      -> Compilando GEMM CPU (MKL)..."
gcc -O3 -march=native -I$MKLROOT/include -o algoritmos/gemm_cpu algoritmos/gemm_cpu.c -L$MKLROOT/lib/intel64 -Wl,-rpath=$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm

# FFT CPU
echo "      -> Compilando FFT CPU (FFTW)..."
gcc -O3 -fopenmp \
    -I/opt/ohpc/pub/libs/gnu12/mpich/fftw/3.3.10/include \
    -o algoritmos/fft_cpu algoritmos/fft_cpu.c \
    -L/opt/ohpc/pub/libs/gnu12/mpich/fftw/3.3.10/lib \
    -Wl,-rpath,/opt/ohpc/pub/libs/gnu12/mpich/fftw/3.3.10/lib \
    -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f -lm

# Volver a cargar NVIDIA SDK para la ejecución en GPU
module load devtools/nvidia/hpc_sdk/nvhpc/23.1

# 4. Ejecución de Benchmarks
echo "[4/4] Ejecutando Benchmarks RL..."

echo "=================== EJECUCIÓN GEMM ==================="
python3 benchmark_runner.py --benchmark gemm --mode continuous-rl --device both

echo "=================== EJECUCIÓN FFT ===================="
python3 benchmark_runner.py --benchmark fft \
    --mode continuous-rl \
    --device both \
    --fft-sizes-1d auto \
    --fft-sizes-2d auto \
    --fft-sizes-3d auto

echo "================================================================="
echo "Finalizado con éxito a las: $(date)"
echo "================================================================="
