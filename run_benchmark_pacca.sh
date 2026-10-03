#!/bin/bash
#SBATCH --job-name=gemm_fft_rl_bench
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --exclusive
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
# Barrido completo de GEMM con la energia de GPU medida por el contador acumulado de NVML
# (validado en validacion_gpu_counter.csv: CV de Pavg de GPU < 5 % entre repeticiones).
# --repetitions 3: se usa la mediana por configuracion al construir las etiquetas, porque
#   ~9 % de las ejecuciones de CPU salen con tiempos atipicos (+50-80 %) por interrupciones.
# --power-window-sec 0.5: no bajarlo; el contador NVML se actualiza cada ~100 ms y con
#   0.5 s caen solo 4-6 saltos dentro de la ventana.
python3 -u benchmark_runner.py --benchmark gemm --mode continuous-rl --device both \
    --repetitions 1\
    --power-window-sec 0.5 --idle-measure-sec 3 \
    --gpu-energy-source counter \
    --output gemm_counter_full.csv

echo "=================== EJECUCIÓN FFT ===================="
# --power-window-sec: duración del lazo bajo monitoreo energético. Con 0.5 s cada
#   medición reúne ~25 muestras NVML / ~100 RAPL dentro de la ventana; bajarlo a 0.15
#   recorta ~2.5 h del barrido completo a costa de muy pocas muestras por punto.
# La potencia en reposo de CPU y GPU se mide automáticamente al inicio del barrido y se
#   descuenta de ambas, para que la comparación CPU/GPU en EDP sea simétrica.

#python3 -u benchmark_runner.py --benchmark fft \
 #   --mode continuous-rl \
  # --power-window-sec 0.5 \
    # --idle-measure-sec 3 \
    #--fft-sizes-1d auto \
    #--fft-sizes-2d auto \
    #--fft-sizes-3d auto
#python3 -u benchmark_runner.py --benchmark fft --device both \
#    --fft-sizes-1d auto --fft-min-n 64 --fft-max-n 4095 \
#    --fft-sizes-2d " " --fft-sizes-3d " " \
#    --power-window-sec 0.5 --idle-measure-sec 3 \
#    --output fft_1d_pequenos.csv

#echo "============ VALIDACION CONTADOR DE ENERGIA NVML ============"
# Prueba corta para validar --gpu-energy-source counter frente al muestreo de potencia
# legado. Los tamanos son los que mostraron mas dispersion de Pavg en GPU en el job 7833:
#   N=288 (gana CPU), N=1216 (frontera: el ganador en EDP cambiaba segun OpA/OpB) y
#   N=2240 (gana GPU). Con 5 repeticiones por caso se puede estimar la varianza.
# Criterio: el CV de Avg_Power_W de GPU entre repeticiones debe bajar de 15-27 % a unos
#   pocos puntos, y en N=1216 S el ganador en EDP no debe depender de OpA/OpB.

# A) Metodo nuevo (contador acumulado NVML), CPU y GPU.
#python3 -u benchmark_runner.py --benchmark gemm --device both \
#    --sizes 288,1216,2240 --precisions S,D,Z \
#    --sweep-transpose --op-a-list N,T --op-b-list N,T \
#    --repetitions 5 \
#    --power-window-sec 0.5 --idle-measure-sec 3 \
#    --gpu-energy-source counter \
#    --output validacion_gpu_counter.csv

# B) Metodo legado (muestreo de potencia), solo GPU, mismos casos: sirve de comparacion
#    directa en el mismo nodo y en el mismo job.
#python3 -u benchmark_runner.py --benchmark gemm --device gpu \
#    --sizes 288,1216,2240 --precisions S,D,Z \
#    --sweep-transpose --op-a-list N,T --op-b-list N,T \
#    --repetitions 5 \
#    --power-window-sec 0.5 --idle-measure-sec 3 \
#    --gpu-energy-source power \
#    --output validacion_gpu_power.csv

echo "================================================================="
echo "Finalizado con éxito a las: $(date)"
echo "================================================================="
