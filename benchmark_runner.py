#!/usr/bin/env python3
"""
benchmark_runner.py

Orquestador de benchmarking GEMM/FFT para CPU o GPU.
Instala `h5py` en el mismo entorno donde ejecutas este script: `python3 -m pip install h5py`.
GUIA DE USO
-----------
Modo GPU (usa los binarios CUDA):
    python3 benchmark_runner.py --device gpu --gemm-binary-gpu ./algoritmos/gemm_gpu

Modo CPU (usa los binarios C/BLAS/FFTW):
    python3 benchmark_runner.py --device cpu --gemm-binary-cpu ./algoritmos/gemm_cpu

Barrido Híbrido Alternado (Recomendado):
    python3 benchmark_runner.py --benchmark gemm --device both --sizes 128,256,512,1024

Barrido GEMM con variaciones de Transpuestas (OpA / OpB):
    python3 benchmark_runner.py --benchmark gemm --device both --sweep-transpose --op-a-list N,T,C --op-b-list N,T,C

Modo FFT con barrido estricto 1D (aislado):
    python3 benchmark_runner.py --benchmark fft --device both \
        --fft-sizes-1d 16384,65536,262144,1048576 --fft-sizes-2d " " --fft-sizes-3d " "

Modo FFT con barrido estricto 3D (aislado):
    python3 benchmark_runner.py --benchmark fft --device both \
        --fft-sizes-1d " " --fft-sizes-2d " " --fft-sizes-3d 16,16,16,16,16,16,64,64,64,64,64,64,256,256,256,256,256,256,1024,1024,1024,1024,1024,1024

OPCIONES PRINCIPALES
--------------------
    --benchmark         Benchmark a ejecutar: gemm o fft
    --device            Dispositivo donde correr el benchmark: gpu, cpu o both
    --gemm-binary-cpu   Ruta al binario GEMM CPU
    --gemm-binary-gpu   Ruta al binario GEMM GPU
    --fft-binary-cpu    Ruta al binario FFT CPU
    --fft-binary-gpu    Ruta al binario FFT GPU
    --sizes             Lista separada por coma para los tamanos base (GEMM)
    --sweep-transpose   Activa el barrido sistemático de OpA / OpB para GEMM
    --op-a-list         Operaciones posibles para la matriz A (N, T, C)
    --op-b-list         Operaciones posibles para la matriz B (N, T, C)
    --fft-sizes-1d      Lista de tamaños 1D para FFT
    --output            Archivo CSV de salida

SALIDA CSV (GEMM / FFT)
-----------------------
Columnas generadas incluyen detalles específicos (M,N,K para GEMM; Nx,Ny,Nz,Batch,Domain para FFT) más las métricas universales:
    Time_sec, GFLOPS, Avg_Power_W, Energy_J, EDP

Interpretacion:
    Time_sec     -> Tiempo puro de ejecución del kernel, medido en fase de aislamiento de métricas.
    GFLOPS       -> Rendimiento calculado a partir de las dimensiones, la precisión y Time_sec.
    Avg_Power_W  -> Potencia media consumida durante la fase secundaria de monitoreo activo.
    Energy_J     -> Energía total gastada (calculada integrando muestras NVML/RAPL).
    EDP          -> Producto Energía-Retraso (Energy-Delay Product = Energy_J * Time_sec).

NOTAS HPC Y RIGOR
-----------------
    - Aislamiento de Métricas: Cada prueba ejecuta el binario dos veces. La primera (sin hilos de monitoreo) obtiene el tiempo exacto; la segunda (con hilos de lectura NVML/RAPL activos) extrae el perfil energético.
    - Warm-ups: Se corren iteraciones previas (por defecto 4) para inicializar bibliotecas (cuBLAS/FFTW) y estabilizar relojes/Turbo Boost.
    - Consumo en GPU: Se calcula a partir del API de NVML.
    - Consumo en CPU: Usa un hilo demonio inactivo que intercepta lecturas precisas de Intel RAPL en /sys/class/powercap.
    - Tolerancia Zero-Time: Tiempos reportados de ejecución por debajo del microsegundo (0.0s) se reajustan internamente al límite teórico de 1 nanosegundo (1e-9) para evitar crasheos (ZeroDivisionError) en barridos masivos de arrays mínimos.
"""

import argparse
import csv
import itertools
import queue
import os
import sys
import re
import subprocess
import threading
import time
import math
import statistics
import struct
import random
import tempfile

import pynvml

# DataBankManager: banco de datos binario con política Lazy Cache.
# Se importa de forma diferida para no bloquear si numpy no está disponible.
_DATA_BANK_MANAGER_MODULE = None

def _get_data_bank_manager_cls():
    """Importa DataBankManager de forma diferida."""
    global _DATA_BANK_MANAGER_MODULE
    if _DATA_BANK_MANAGER_MODULE is None:
        import importlib.util, pathlib
        _here = pathlib.Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "data_bank_manager",
            _here / "bench_files" / "data_bank_manager.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DATA_BANK_MANAGER_MODULE = mod
    return _DATA_BANK_MANAGER_MODULE.DataBankManager

def _get_data_bank_manager_module():
    """Importa el módulo data_bank_manager de forma diferida."""
    global _DATA_BANK_MANAGER_MODULE
    if _DATA_BANK_MANAGER_MODULE is None:
        _get_data_bank_manager_cls()
    return _DATA_BANK_MANAGER_MODULE

from typing import List, Iterator


class RLWorkloadGenerator:
    """Generador de tamaños de carga de trabajo para entrenamiento de Reinforcement Learning.

    Soporta los algoritmos de GEMM (híbrido por rangos) y FFT (lineal denso para romper potencias de 2).
    """

    def __init__(
        self,
        algorithm: str,
        gemm_min_n: int = 64,
        gemm_max_n: int = 16384,
        gemm_low_step: int = 32,
        gemm_trans_step: int = 512,
        gemm_high_step: int = 1024,
        fft_min_n: int = 4096,
        fft_max_n: int = 67108864,
        fft_low_step: int = 256,
        fft_mid_step: int = 4096,
        fft_high_step: int = 262144,
    ) -> None:
        """Inicializa el generador con los parámetros de crecimiento específicos.

        Args:
            algorithm (str): Algoritmo objetivo ('gemm' o 'fft').
            gemm_min_n (int, optional): Límite inferior para GEMM. Defaults to 64.
            gemm_max_n (int, optional): Límite superior para GEMM. Defaults to 16384.
            gemm_low_step (int, optional): Incremento en rango de baja latencia. Defaults to 32.
            gemm_trans_step (int, optional): Paso de transición (ignorado, por compatibilidad).
            gemm_high_step (int, optional): Paso intensivo (ignorado, por compatibilidad).
            fft_min_n (int, optional): Límite inferior para FFT. Defaults to 4096.
            fft_max_n (int, optional): Límite superior para FFT. Defaults to 67108864.
            fft_low_step (int, optional): Incremento base para FFT. Defaults to 256.
            fft_mid_step (int, optional): Paso medio (ignorado, por compatibilidad).
            fft_high_step (int, optional): Paso alto (ignorado, por compatibilidad).

        Raises:
            ValueError: Si el algoritmo especificado no está soportado o si algún paso es inválido.
        """
        algo_lower = algorithm.lower()
        if algo_lower not in {"gemm", "fft", "fft_1d", "fft_2d", "fft_3d"}:
            raise ValueError(f"Algoritmo '{algorithm}' no soportado. Debe ser 'gemm', 'fft', 'fft_1d', 'fft_2d' o 'fft_3d'.")

        if gemm_low_step not in {32, 64, 128}:
            raise ValueError("gemm_low_step debe ser 32, 64 o 128.")
        if fft_low_step not in {128, 256, 512}:
            raise ValueError("fft_low_step debe ser 128, 256 o 512.")

        self.algorithm = algo_lower
        self.gemm_min_n = gemm_min_n
        self.gemm_max_n = gemm_max_n
        self.gemm_low_step = gemm_low_step
        self.gemm_trans_step = gemm_trans_step
        self.gemm_high_step = gemm_high_step
        self.fft_min_n = fft_min_n
        self.fft_max_n = fft_max_n
        self.fft_low_step = fft_low_step
        self.fft_mid_step = fft_mid_step
        self.fft_high_step = fft_high_step

    def generate(self) -> List[int]:
        """Genera la lista ordenada de tamaños N de acuerdo con el algoritmo seleccionado.

        Returns:
            List[int]: Lista con los tamaños exactos de N.
        """
        if self.algorithm == "gemm":
            return self._generate_gemm()
        elif self.algorithm in ("fft", "fft_1d"):
            return self._generate_fft()
        elif self.algorithm == "fft_2d":
            return self._generate_fft_2d()
        elif self.algorithm == "fft_3d":
            return self._generate_fft_3d()
        else:
            raise ValueError(f"Algoritmo desconocido: {self.algorithm}")

    def __iter__(self) -> Iterator[int]:
        """Permite iterar directamente sobre el generador.

        Returns:
            Iterator[int]: Iterador sobre la lista de tamaños generados.
        """
        return iter(self.generate())

    def _generate_gemm(self) -> List[int]:
        """Genera la lista de tamaños N para GEMM usando la estrategia por octavas con 32 puntos por octava.

        Returns:
            List[int]: Lista de tamaños para GEMM.
        """
        sizes: List[int] = []
        puntos_por_octava = 32
        k_start = (self.gemm_min_n).bit_length() - 1
        k_end = (self.gemm_max_n - 1).bit_length()

        for k in range(k_start, k_end):
            interval_start = max(2**k, self.gemm_min_n)
            interval_end = 2**(k + 1)
            ancho_octava = 2**k
            step_exact = ancho_octava / puntos_por_octava
            step = max(1, int(round(step_exact)))

            n = interval_start
            while n < interval_end and n <= self.gemm_max_n:
                sizes.append(n)
                n += step

        if 2**k_end <= self.gemm_max_n and 2**k_end >= self.gemm_min_n and (not sizes or sizes[-1] < self.gemm_max_n):
            sizes.append(self.gemm_max_n)

        return sizes

    def _generate_fft(self) -> List[int]:
        """Genera la lista de tamaños N para FFT 1D usando el esquema de octavas con 32 puntos por octava.

        Returns:
            List[int]: Lista de tamaños para FFT.
        """
        sizes: List[int] = []
        puntos_por_octava = 32
        k_start = (self.fft_min_n).bit_length() - 1
        k_end = (self.fft_max_n - 1).bit_length()

        for k in range(k_start, k_end):
            interval_start = max(2**k, self.fft_min_n)
            interval_end = 2**(k + 1)
            ancho_octava = 2**k
            step_exact = ancho_octava / puntos_por_octava
            step = max(1, int(round(step_exact)))

            n = interval_start
            while n < interval_end and n <= self.fft_max_n:
                sizes.append(n)
                n += step

        if 2**k_end <= self.fft_max_n and 2**k_end >= self.fft_min_n and (not sizes or sizes[-1] < self.fft_max_n):
            sizes.append(self.fft_max_n)

        return sizes

    def _generate_fft_2d(self) -> List[int]:
        """Genera tamaños N para FFT 2D en esquema de octavas (2^6 a 2^13, 64 a 8192) con 32 puntos por octava.

        Returns:
            List[int]: Lista de tamaños N para matrices N x N.
        """
        sizes: List[int] = []
        puntos_por_octava = 32
        for k in range(6, 13):
            interval_start = 2**k
            interval_end = 2**(k + 1)
            ancho_octava = interval_end - interval_start
            step_exact = ancho_octava / puntos_por_octava
            step = max(1, int(round(step_exact)))
            n = interval_start
            while n < interval_end:
                sizes.append(n)
                n += step
        sizes.append(8192)
        return sizes

    def _generate_fft_3d(self) -> List[int]:
        """Genera tamaños N para FFT 3D en esquema de octavas (2^4 a 2^8, 16 a 256) con 32 puntos por octava.

        Returns:
            List[int]: Lista de tamaños N para volúmenes N x N x N.
        """
        sizes: List[int] = []
        puntos_por_octava = 32
        for k in range(4, 8):
            interval_start = 2**k
            interval_end = 2**(k + 1)
            ancho_octava = interval_end - interval_start
            step_exact = ancho_octava / puntos_por_octava
            step = max(1, int(round(step_exact)))
            n = interval_start
            while n < interval_end:
                sizes.append(n)
                n += step
        sizes.append(256)
        return sizes


# Expresion regular para extraer el tiempo reportado por el binario CUDA.
TIME_PATTERN = re.compile(r"Time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
FFT_TIME_PATTERN = re.compile(
    r"Time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|tiempo=([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)

_RAPL_WARNING_SHOWN = False
POWER_SAMPLE_INTERVAL_SEC = 0.02
IDLE_POWER_CPU = 0.0

DEFAULT_DATABANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_files", "databank")
DEFAULT_DATABANK_MAX_N = 67108864

DEFAULT_BENCHMARK_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_files", "benchmark_bank.h5")




def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py es necesario para usar el banco HDF5 de benchmarks. "
            "Instala la dependencia con: python3 -m pip install h5py"
        ) from exc
    return h5py


def warn_rapl_missing_once():
    global _RAPL_WARNING_SHOWN
    if _RAPL_WARNING_SHOWN:
        return
    print(
        "Aviso: no se encontro energy_uj RAPL en /sys/class/powercap; "
        "se continuara sin metrica de energia para CPU.",
        file=sys.stderr,
    )
    _RAPL_WARNING_SHOWN = True


def parse_sizes(raw):
    # Convierte una lista separada por comas en enteros validos para M/N/K.
    values = [x.strip() for x in raw.split(",") if x.strip()]
    sizes = [int(x) for x in values]
    if not sizes:
        raise ValueError("La lista de tamanos no puede estar vacia")
    for s in sizes:
        if s <= 0:
            raise ValueError("Todos los tamanos deben ser positivos")
    return sizes


def parse_precisions(raw):
    # Normaliza y valida las precisiones soportadas por el binario CUDA.
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"S", "D", "C", "Z"}
    for p in values:
        if p not in valid:
            raise ValueError(f"Precision invalida: {p}")
    if not values:
        raise ValueError("La lista de precisiones no puede estar vacia")
    return values


def parse_ops(raw):
    # Normaliza y valida operaciones de transposicion para cuBLAS GEMM.
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"N", "T", "C"}
    for op in values:
        if op not in valid:
            raise ValueError(f"Operacion invalida: {op}")
    if not values:
        raise ValueError("La lista de operaciones no puede estar vacia")
    return values


def parse_int_list(raw, name):
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"La lista de {name} no puede estar vacia")
    parsed = []
    for v in values:
        n = int(v)
        if n <= 0:
            raise ValueError(f"Valor invalido en {name}: {v}")
        parsed.append(n)
    return parsed


def parse_fft_precisions(raw):
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"S", "D"}
    for p in values:
        if p not in valid:
            raise ValueError(f"Precision FFT invalida: {p}")
    if not values:
        raise ValueError("La lista de precisiones FFT no puede estar vacia")
    return values


def parse_fft_domains(raw):
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"C2C", "R2C", "C2R"}
    for d in values:
        if d not in valid:
            raise ValueError(f"Dominio FFT invalido: {d}")
    if not values:
        raise ValueError("La lista de dominios FFT no puede estar vacia")
    return values


def parse_fft_directions(raw):
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"F", "I"}
    for d in values:
        if d not in valid:
            raise ValueError(f"Direccion FFT invalida: {d}")
    if not values:
        raise ValueError("La lista de direcciones FFT no puede estar vacia")
    return values


def parse_fft_layouts(raw):
    values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    valid = {"I", "O"}
    for d in values:
        if d not in valid:
            raise ValueError(f"Layout FFT invalido: {d}")
    if not values:
        raise ValueError("La lista de layouts FFT no puede estar vacia")
    return values


def parse_fft_shapes(raw, dims):
    if not raw.strip():
        return []
    raw_lower = raw.strip().lower()
    if raw_lower in ("auto", "octave", "default"):
        db_mgr_mod = _get_data_bank_manager_module()
        algo_name = f"fft_{dims}d" if dims in (2, 3) else "fft_1d"
        sizes = db_mgr_mod.generate_size_sweep(algorithm=algo_name)
        if dims == 1:
            return [(n, 0, 0) for n in sizes]
        elif dims == 2:
            return [(n, n, 0) for n in sizes]
        else:
            return [(n, n, n) for n in sizes]

    shapes = []
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    for token in tokens:
        parts = token.lower().split("x")
        if len(parts) != dims:
            raise ValueError(f"Forma FFT invalida: {token}")
        values = [int(p) for p in parts]
        if any(v <= 0 for v in values):
            raise ValueError(f"Forma FFT invalida: {token}")
        if dims == 1:
            shapes.append((values[0], 0, 0))
        elif dims == 2:
            shapes.append((values[0], values[1], 0))
        else:
            shapes.append((values[0], values[1], values[2]))
    return shapes


def bank_dataset_exists(bank_path, dataset_path):
    h5py = _require_h5py()
    if not bank_path or not os.path.isfile(bank_path):
        return False
    try:
        with h5py.File(bank_path, "r") as hf:
            return dataset_path in hf
    except OSError:
        return False


def write_gemm_matrix_file_from_arrays(m, n, k, precision, a_values, b_values, c_values):
    fd, matrix_file = tempfile.mkstemp(prefix="gemm_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("i", m))
            f.write(struct.pack("i", n))
            f.write(struct.pack("i", k))
            f.write(precision.encode("ascii"))

            if precision in ("S", "D"):
                fmt = "d" if precision == "D" else "f"
                for val in a_values:
                    f.write(struct.pack(fmt, float(val)))
                for val in b_values:
                    f.write(struct.pack(fmt, float(val)))
                for val in c_values:
                    f.write(struct.pack(fmt, float(val)))
            else:
                fmt = "d" if precision == "Z" else "f"
                for re, im in a_values:
                    f.write(struct.pack(fmt, float(re)))
                    f.write(struct.pack(fmt, float(im)))
                for re, im in b_values:
                    f.write(struct.pack(fmt, float(re)))
                    f.write(struct.pack(fmt, float(im)))
                for re, im in c_values:
                    f.write(struct.pack(fmt, float(re)))
                    f.write(struct.pack(fmt, float(im)))
    except Exception:
        os.unlink(matrix_file)
        raise
    return matrix_file


def write_fft_matrix_file_from_arrays(nx, ny, nz, batch, precision, domain, input_values, output_values):
    fd, matrix_file = tempfile.mkstemp(prefix="fft_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("i", nx))
            f.write(struct.pack("i", ny))
            f.write(struct.pack("i", nz))
            f.write(struct.pack("i", batch))
            f.write(precision.encode("ascii"))
            f.write(domain.encode("ascii"))
            fmt = "d" if precision == "D" else "f"
            for value in input_values:
                if isinstance(value, complex):
                    f.write(struct.pack(fmt, float(value.real)))
                    f.write(struct.pack(fmt, float(value.imag)))
                else:
                    f.write(struct.pack(fmt, float(value)))
            for value in output_values:
                if isinstance(value, complex):
                    f.write(struct.pack(fmt, float(value.real)))
                    f.write(struct.pack(fmt, float(value.imag)))
                else:
                    f.write(struct.pack(fmt, float(value)))
    except Exception:
        os.unlink(matrix_file)
        raise
    return matrix_file


def fft_dims(nx, ny, nz):
    if nz > 0:
        return [nx, ny, nz]
    if ny > 0:
        return [nx, ny]
    return [nx]


def fft_total_points(dims):
    total = 1
    for d in dims:
        total *= d
    return total


def fft_complex_elements(dims):
    last = dims[-1]
    outer = 1
    for d in dims[:-1]:
        outer *= d
    return outer * (last // 2 + 1)


def fft_sum_log2(dims):
    return sum(math.log2(d) for d in dims)


def fft_radix_class(dims):
    def is_pow2(n):
        return n > 0 and (n & (n - 1)) == 0

    def is_smooth_235(n):
        if n <= 0:
            return False
        for p in (2, 3, 5):
            while n % p == 0:
                n //= p
        return n == 1

    if all(is_pow2(d) for d in dims):
        return "pow2"
    if all(is_smooth_235(d) for d in dims):
        return "smooth235"
    return "other"


def fft_payload_bytes(dims, batch, precision, domain, layout):
    real_bytes = 4 if precision == "S" else 8
    complex_bytes = real_bytes * 2
    nreal = fft_total_points(dims)
    ncomplex = fft_complex_elements(dims)

    if domain == "C2C":
        in_bytes = nreal * complex_bytes * batch
        out_bytes = nreal * complex_bytes * batch
    elif domain == "R2C":
        in_bytes = nreal * real_bytes * batch
        out_bytes = ncomplex * complex_bytes * batch
    else:  # C2R
        in_bytes = ncomplex * complex_bytes * batch
        out_bytes = nreal * real_bytes * batch

    if layout == "I":
        return max(in_bytes, out_bytes)
    return in_bytes + out_bytes


def fft_flops(dims, domain):
    ntotal = fft_total_points(dims)
    sum_log2 = fft_sum_log2(dims)
    factor = 5.0 if domain == "C2C" else 2.5
    return factor * ntotal * sum_log2


def monitor_power_gpu(handle, stop_event, power_queue):
    # Hilo de monitoreo NVML: muestrea potencia con un intervalo fijo para evitar picos espurios.
    # NOTA: NVML documenta nvmlDeviceGetPowerUsage() en mW, pero en algunos entornos se observa
    # un escalado distinto. Para corregirlo sin "filtrar" datos, inferimos el divisor usando
    # los límites de potencia del propio dispositivo (constraints/power limit).
    max_limit_mw = None
    try:
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        max_limit_mw = int(max_mw)
    except Exception:
        max_limit_mw = None

    if max_limit_mw is None:
        try:
            max_limit_mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        except Exception:
            max_limit_mw = None

    if max_limit_mw is None:
        try:
            max_limit_mw = int(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
        except Exception:
            max_limit_mw = None

    raw_samples = []
    while True:
        timestamp = time.perf_counter()
        try:
            power_raw = pynvml.nvmlDeviceGetPowerUsage(handle)
            # Guardamos el entero crudo (segun NVML, milivatios) y convertimos después.
            raw_samples.append((timestamp, int(power_raw)))
        except Exception:
            # En caso de fallo NVML, seguir intentando hasta stop_event
            pass
        if stop_event.wait(POWER_SAMPLE_INTERVAL_SEC):
            break

    # Si tenemos menos de 2 muestras, hacemos un muestreo en ráfaga rápido
    if len(raw_samples) < 2:
        extra = []
        burst_reads = 8
        burst_delay = 0.002  # 2 ms entre lecturas
        for i in range(burst_reads):
            try:
                t = time.perf_counter()
                p_raw = pynvml.nvmlDeviceGetPowerUsage(handle)
                extra.append((t, int(p_raw)))
            except Exception:
                continue
            time.sleep(burst_delay)

        if extra:
            raw_samples.extend(extra)

    if not raw_samples:
        power_queue.put([])
        return

    # Inferir la unidad/divisor correcto examinando la mediana de los valores crudos.
    vals = [v for (_t, v) in raw_samples]
    median_raw = statistics.median(vals)

    # Candidatos de divisor: 1000 (mW->W) y 1e6 (uW->W)
    cand_mw = median_raw / 1000.0
    cand_uw = median_raw / 1e6

    divisor = 1000.0
    if max_limit_mw is not None and max_limit_mw > 0:
        max_limit_w = max_limit_mw / 1000.0
        # Elegimos el candidato que cae dentro de un margen razonable del límite del dispositivo.
        mw_ok = 0.0 <= cand_mw <= (max_limit_w * 1.20)
        uw_ok = 0.0 <= cand_uw <= (max_limit_w * 1.20)
        if uw_ok and not mw_ok:
            divisor = 1e6
            print(
                f"Aviso: NVML power usage parece estar escalado (mediana_raw={median_raw}, "
                f"limite~{max_limit_w:.1f}W). Usando divisor 1e6 (uW->W).",
                file=sys.stderr,
            )
        elif mw_ok:
            divisor = 1000.0
        else:
            # Ninguno encaja: dejamos mW->W y reportamos para diagnóstico.
            divisor = 1000.0
            print(
                f"Aviso: lectura NVML fuera de rango (mediana_mW={cand_mw:.1f}W, "
                f"mediana_uW={cand_uw:.3f}W, limite~{max_limit_w:.1f}W).",
                file=sys.stderr,
            )
    else:
        # Sin límites disponibles, inferimos escala con un umbral simple.
        if median_raw >= 1e6:
            divisor = 1e6
            print(
                f"Aviso: no se pudo leer limite de potencia NVML; "
                f"mediana_raw={median_raw} sugiere uW. Usando divisor 1e6.",
                file=sys.stderr,
            )
        else:
            divisor = 1000.0

    # Convertir todas las muestras a Watts
    samples = [(t, v / divisor) for (t, v) in raw_samples]

    power_queue.put(samples)


def average_power_from_samples(samples):
    # Calcula potencia media a partir de muestras temporizadas.
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0][1]

    samples = sorted(samples, key=lambda item: item[0])
    area = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            area += (p0 + p1) * 0.5 * dt

    duration = samples[-1][0] - samples[0][0]
    if duration <= 0.0:
        return samples[-1][1]
    return area / duration


def average_and_energy_from_samples(samples):
    # Devuelve (avg_power_w, energy_j) integrando las muestras temporizadas.
    # samples: list of (timestamp, power_w)
    if not samples:
        return 0.0, 0.0
    if len(samples) == 1:
        # No duration info: treat as instantaneous power, energy undefined (0)
        return samples[0][1], 0.0

    samples = sorted(samples, key=lambda item: item[0])
    area = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            area += (p0 + p1) * 0.5 * dt

    duration = samples[-1][0] - samples[0][0]
    if duration <= 0.0:
        return samples[-1][1], 0.0
    avg = area / duration
    energy = area  # area is in W*s = Joules over the sampling window
    return avg, energy


def monitor_power_cpu(energy_path, stop_event, power_queue):
    # Monitor RAPL via sysfs: lee energy_uj al inicio y al final.
    def read_energy_uj(path):
        with open(path, "r") as f:
            return int(f.read().strip())

    try:
        t0 = time.perf_counter()
        e0 = read_energy_uj(energy_path)
    except Exception:
        power_queue.put([])
        return

    # Espera a la senal de parada
    stop_event.wait()

    try:
        t1 = time.perf_counter()
        e1 = read_energy_uj(energy_path)
    except Exception:
        power_queue.put([])
        return

    # RAPL energy_uj esta en microjoules, convertimos a joules.
    # Dividimos entre 1e6 (1 microjoule = 1e-6 joules)
    samples = [(t0, e0 / 1e6), (t1, e1 / 1e6)]
    power_queue.put(samples)


def find_rapl_energy_paths():
    # Busca energy_uj sin recorrer recursivamente todo powercap; así evitamos bloqueos.
    base_dir = "/sys/class/powercap"
    paths = []
    if not os.path.isdir(base_dir):
        return paths

    def is_readable(p):
        return os.path.isfile(p) and os.access(p, os.R_OK)

    # Escaneo dinámico buscando Package energy domains (sockets físicos)
    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("intel-rapl"):
                    continue

                name_path = os.path.join(base_dir, entry.name, "name")
                energy_path = os.path.join(base_dir, entry.name, "energy_uj")

                if is_readable(name_path) and is_readable(energy_path):
                    try:
                        with open(name_path, "r") as f:
                            name_val = f.read().strip().lower()
                        if "package" in name_val:
                            paths.append(energy_path)
                    except Exception:
                        continue
    except OSError:
        pass

    # Fallback si no se encontró nada por nombre, pero intel-rapl:0 es legible
    if not paths:
        fallback = os.path.join(base_dir, "intel-rapl:0", "energy_uj")
        if is_readable(fallback):
            paths.append(fallback)

    return sorted(paths)


def generate_gemm_matrix_file(
    m, n, k, precision,
    seed=None, bank_path=None, bank_profile="dense_normal",
    databank_dir=None, databank_max_n=None,
):
    """Retorna la ruta a un archivo GEMM kernel-ready usando DataBankManager.

    Política:
        1. DataBankManager (banco binario persistente, lazy) — ruta primaria.
        2. Banco HDF5 legacy (solo S/D, solo tamaños pre-generados).
        3. Generación aleatoria en memoria → archivo temporal.

    Args:
        m, n, k: Dimensiones de las matrices.
        precision: S, D, C o Z.
        seed: Semilla aleatoria (solo para fallback temporal).
        bank_path: Ruta al HDF5 legacy (opcional).
        bank_profile: Perfil en el banco HDF5 / DataBankManager.
        databank_dir: Raíz del banco binario. None usa DEFAULT_DATABANK_DIR.
        databank_max_n: Techo N para el banco. None usa DEFAULT_DATABANK_MAX_N.

    Returns:
        tuple[str, bool]: (ruta_archivo, es_persistente).
            es_persistente=True  → NO borrar al terminar.
            es_persistente=False → archivo temporal, borrar en finally.
    """
    # ── Ruta primaria: DataBankManager (binario plano, lazy) ──────────────────
    try:
        DataBankManager = _get_data_bank_manager_cls()
        db_dir = databank_dir or DEFAULT_DATABANK_DIR
        db_max = databank_max_n or DEFAULT_DATABANK_MAX_N
        mgr = DataBankManager(base_dir=db_dir, max_n=db_max)
        path = mgr.get_gemm_path(m, n, k, precision, profile=bank_profile)
        return path, True
    except Exception as e:
        import sys
        print(f"\n[DEBUG] Error en DataBankManager: {type(e).__name__} - {e}\n", file=sys.stderr)
        pass  # Fallback al banco HDF5 o generación aleatoria

    # ── Fallback 1: banco HDF5 legacy (solo S/D cuadradas) ───────────────────
    if bank_path and precision in {"S", "D"} and m == n == k:
        try:
            h5py = _require_h5py()
            dataset_root = f"/gemm/N{m}/{bank_profile}"
            a_key = f"{dataset_root}/A_{'f64' if precision == 'D' else 'f32'}"
            b_key = f"{dataset_root}/B_{'f64' if precision == 'D' else 'f32'}"
            if bank_dataset_exists(bank_path, a_key) and bank_dataset_exists(bank_path, b_key):
                with h5py.File(bank_path, "r") as hf:
                    a_values = hf[a_key][()]
                    b_values = hf[b_key][()]
                c_values = [0.0] * (m * n)
                return write_gemm_matrix_file_from_arrays(
                    m, n, k, precision,
                    a_values.ravel(), b_values.ravel(), c_values,
                ), False
        except Exception:
            pass

    # ── Fallback 2: generación aleatoria (archivo temporal) ──────────────────
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(random.randint(0, 2**31 - 1))

    if precision in ("S", "D"):
        A = [random.random() for _ in range(m * k)]
        B = [random.random() for _ in range(k * n)]
        C = [0.0] * (m * n)
    else:
        A = [(random.random(), random.random()) for _ in range(m * k)]
        B = [(random.random(), random.random()) for _ in range(k * n)]
        C = [(0.0, 0.0)] * (m * n)

    return write_gemm_matrix_file_from_arrays(m, n, k, precision, A, B, C), False


def generate_fft_matrix_file(
    nx, ny, nz, batch, precision, domain, layout,
    seed=None, bank_path=None, bank_profile="broadband",
    databank_dir=None, databank_max_n=None,
):
    """Retorna la ruta a un archivo FFT kernel-ready usando DataBankManager.

    Política:
        1. DataBankManager (banco binario persistente, lazy) — ruta primaria.
        2. Banco HDF5 legacy (solo 1D, solo tamaños pre-generados).
        3. Generación aleatoria en memoria → archivo temporal.

    Args:
        nx, ny, nz: Dimensiones del transform (ny=nz=0 para 1D).
        batch: Número de transforms.
        precision: S o D.
        domain: C2C, R2C o C2R.
        layout: I (in-place) u O (out-of-place); solo para referencia del caller.
        seed, bank_path, bank_profile, databank_dir, databank_max_n: Igual que GEMM.

    Returns:
        tuple[str, bool]: (ruta_archivo, es_persistente).
    """
    # ── Ruta primaria: DataBankManager ────────────────────────────────────────
    try:
        DataBankManager = _get_data_bank_manager_cls()
        db_dir = databank_dir or DEFAULT_DATABANK_DIR
        db_max = databank_max_n or DEFAULT_DATABANK_MAX_N
        mgr = DataBankManager(base_dir=db_dir, max_n=db_max)
        path = mgr.get_fft_path(nx, ny, nz, batch, precision, domain, profile=bank_profile)
        return path, True
    except Exception:
        pass

    # ── Fallback 1: banco HDF5 legacy (solo 1D) ───────────────────────────────
    if bank_path and ny == 0 and nz == 0:
        try:
            h5py = _require_h5py()
            dataset_root = f"/fft/N{nx}/{bank_profile}"
            if domain == "C2C":
                label = "c128" if precision == "D" else "c64"
            elif domain == "R2C":
                label = "f64" if precision == "D" else "f32"
            else:
                label = "c128" if precision == "D" else "c64"
            dataset_key = f"{dataset_root}/{label}"
            if bank_dataset_exists(bank_path, dataset_key):
                with h5py.File(bank_path, "r") as hf:
                    payload = list(hf[dataset_key][()])
                if batch > 1:
                    payload = payload * batch
                if domain == "C2C":
                    input_values, output_values = payload, [0j] * len(payload)
                elif domain == "R2C":
                    input_values = payload
                    output_values = [0j] * (nx * batch)
                else:
                    input_values = payload
                    output_values = [0.0] * (nx * batch)
                return write_fft_matrix_file_from_arrays(
                    nx, ny, nz, batch, precision, domain,
                    input_values, output_values,
                ), False
        except Exception:
            pass

    # ── Fallback 2: generación aleatoria en memoria ───────────────────────────
    if seed is not None:
        random.seed(seed)
    nreal = 1
    for d in [nx, ny, nz]:
        if d > 0:
            nreal *= d
    nreal *= batch
    input_values = [complex(random.random(), random.random()) for _ in range(nreal)]
    output_values = [0j] * nreal
    return write_fft_matrix_file_from_arrays(
        nx, ny, nz, batch, precision, domain, input_values, output_values
    ), False


def run_gemm_warmup(cmd, timeout, warmup_runs, matrix_file=None):
    # Ejecuta warmup(s) sin recolectar potencia ni parsear tiempos.
    for _ in range(warmup_runs):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Fallo en warmup GEMM.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )


def run_single_case(
    binary,
    device,
    gpu_index,
    m,
    n,
    k,
    precision,
    op_a,
    op_b,
    timeout,
    is_warmup,
    seed,
    bank_path,
    bank_profile,
    databank_dir=None,
    databank_max_n=None,
):
    matrix_file, _gemm_file_is_persistent = generate_gemm_matrix_file(
        m,
        n,
        k,
        precision,
        seed=seed,
        bank_path=bank_path,
        bank_profile=bank_profile,
        databank_dir=databank_dir,
        databank_max_n=databank_max_n,
    )

    try:
        # Build binary execution command using CLI flags (allows specifying --warmup 0 --iters 1)
        cmd = [
            binary,
            "--m", str(m),
            "--n", str(n),
            "--k", str(k),
            "--precision", precision,
            "--op-a", op_a,
            "--op-b", op_b,
            "--source", matrix_file,
            "--warmup", "0",
            "--iters", "0" if not is_warmup else "1"
        ]

        if is_warmup:
            # 1. Warmup Run: Execute the binary once with 0 warmups, 1 iter, and no telemetry
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Fallo en warmup GEMM para "
                    f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            return {}

        # 2. Metric Isolation Execution (Solo Tiempo)
        proc_iso = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc_iso.returncode != 0:
            raise RuntimeError(
                "Fallo en binario para "
                f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\n"
                f"STDOUT:\n{proc_iso.stdout}\nSTDERR:\n{proc_iso.stderr}"
            )

        match = TIME_PATTERN.search(proc_iso.stdout)
        if not match:
            raise RuntimeError(
                "No se pudo parsear Time_sec de la salida del binario en aislamiento.\n"
                f"Salida:\n{proc_iso.stdout}"
            )

        time_sec = float(match.group(1))

        # 3. Power Monitoring Execution (Segunda ejecucion identica con monitor activo)
        K = min(20000, max(1, round(0.15 / time_sec)))
        cmd_pwr = list(cmd)
        try:
            iters_idx = cmd_pwr.index("--iters")
            cmd_pwr[iters_idx + 1] = str(K)
        except ValueError:
            cmd_pwr.extend(["--iters", str(K)])

        power_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        monitor_thread = None
        
        rapl_paths = []
        e0_list = []
        t0 = 0.0

        if device == "gpu":
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            monitor_thread = threading.Thread(
                target=monitor_power_gpu,
                args=(handle, stop_event, power_queue),
                daemon=True,
            )
            if monitor_thread is not None:
                monitor_thread.start()
        else:
            rapl_paths = find_rapl_energy_paths()
            if rapl_paths:
                try:
                    t0 = time.perf_counter()
                    for p in rapl_paths:
                        with open(p, "r") as f:
                            e0_list.append(int(f.read().strip()))
                except Exception as ex:
                    print(f"[!] Error al leer RAPL inicial: {ex}", file=sys.stderr)
                    rapl_paths = []
            else:
                warn_rapl_missing_once()

        start_wall = time.perf_counter()
        try:
            proc_pwr = subprocess.run(
                cmd_pwr,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        finally:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join()

        end_wall = time.perf_counter()
        samples = []

        if proc_pwr.returncode != 0:
            raise RuntimeError(
                "Fallo en binario para ejecucion de monitoreo "
                f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\n"
                f"STDOUT:\n{proc_pwr.stdout}\nSTDERR:\n{proc_pwr.stderr}"
            )
        if time_sec <= 0.0:
            time_sec = 1e-9

        avg_power_w = 0.0
        energy_j = 0.0
        wall_time = end_wall - start_wall
        power_window_sec = wall_time

        if device == "gpu":
            samples = power_queue.get() if not power_queue.empty() else []
            avg_power_w, energy_total_j = average_and_energy_from_samples(samples)
            energy_j = energy_total_j / K
            if samples:
                power_window_sec = samples[-1][0] - samples[0][0]
        else:
            if rapl_paths:
                try:
                    t1 = time.perf_counter()
                    energy_total_j = 0.0
                    for i, p in enumerate(rapl_paths):
                        with open(p, "r") as f:
                            val = int(f.read().strip())
                        diff = max(0.0, (val - e0_list[i]) / 1e6)
                        energy_total_j += diff

                    power_window_sec = t1 - t0
                    if power_window_sec <= 0.0:
                        power_window_sec = wall_time
                    
                    # Deduce total idle energy consumption during the process window
                    energy_idle = IDLE_POWER_CPU * power_window_sec
                    energy_active = max(0.0, energy_total_j - energy_idle)
                    
                    # Calculate active power over the active computation loop (excluding startup/IO)
                    t_active = K * time_sec
                    avg_power_w = energy_active / t_active if t_active > 0.0 else 0.0
                    energy_j = energy_active / K if K > 0 else 0.0
                except Exception as ex:
                    print(f"[!] Error al leer RAPL final: {ex}", file=sys.stderr)
                    avg_power_w = 0.0
                    energy_j = 0.0
            else:
                avg_power_w = 0.0
                energy_j = 0.0

        if precision in {"C", "Z"}:
            ops = 8.0 * m * n * k
        else:
            ops = 2.0 * m * n * k

        gflops = (ops / time_sec) / 1e9
        edp = energy_j * time_sec

        return {
            "M": m,
            "N": n,
            "K": k,
            "Precision": precision,
            "OpA": op_a,
            "OpB": op_b,
            "Time_sec": time_sec,
            "GFLOPS": gflops,
            "Avg_Power_W": avg_power_w,
            "Energy_J": energy_j,
            "EDP": edp,
            "Power_Samples": len(samples) if device == "gpu" else (2 * len(rapl_paths) if rapl_paths else 0),
            "Wall_Elapsed_sec": end_wall - start_wall,
        }
    finally:
        # Solo eliminar el archivo si es temporal (no proviene del DataBankManager).
        if not _gemm_file_is_persistent and os.path.exists(matrix_file):
            os.unlink(matrix_file)


def run_single_case_fft(
    binary,
    device,
    gpu_index,
    nx,
    ny,
    nz,
    batch,
    precision,
    domain,
    direction,
    layout,
    plan,
    is_warmup,
    timeout,
    matrix_file,
):
    # Construct binary execution command using positional arguments:
    # Nx Ny Nz Batch Precision Domain Direction Layout Warmup Iters Plan [matrix_file]
    cmd = [
        binary,
        str(nx),
        str(ny),
        str(nz),
        str(batch),
        precision,
        domain,
        direction,
        layout,
        "0",  # warmup_runs = 0
        "0" if not is_warmup else "1",  # iters
    ]
    if plan is not None:
        cmd.append(plan)
    else:
        cmd.append("E")
    if matrix_file:
        cmd.append(matrix_file)

    if is_warmup:
        # 1. Warmup Run: Execute the binary once with 0 warmups, 1 iter, and no telemetry
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Fallo en warmup FFT para "
                f"Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, Dir={direction}, L={layout}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return {}

    # 2. Metric Isolation Execution (Solo Tiempo)
    proc_iso = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if proc_iso.returncode != 0:
        raise RuntimeError(
            "Fallo en binario FFT para "
            f"Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, Dir={direction}, L={layout}.\n"
            f"STDOUT:\n{proc_iso.stdout}\nSTDERR:\n{proc_iso.stderr}"
        )

    match = FFT_TIME_PATTERN.search(proc_iso.stdout)
    if not match:
        raise RuntimeError(
            "No se pudo parsear tiempo de la salida FFT.\n"
            f"Salida:\n{proc_iso.stdout}"
        )

    if match.group(1) is not None:
        time_sec = float(match.group(1))
    else:
        time_sec = float(match.group(2)) / 1e3

    if time_sec <= 0.0:
        time_sec = 1e-9

    # 3. Power Monitoring Execution (Segunda ejecucion con monitor activo)
    K = min(20000, max(1, round(0.15 / time_sec)))
    cmd_pwr = list(cmd)
    if len(cmd_pwr) > 10:
        cmd_pwr[10] = str(K)
    else:
        raise ValueError(f"Comando FFT mal formado para agregar iteraciones: {cmd_pwr}")

    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    monitor_thread = None
    
    rapl_paths = []
    e0_list = []
    t0 = 0.0

    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu,
            args=(handle, stop_event, power_queue),
            daemon=True,
        )
        if monitor_thread is not None:
            monitor_thread.start()
    else:
        rapl_paths = find_rapl_energy_paths()
        if rapl_paths:
            try:
                t0 = time.perf_counter()
                for p in rapl_paths:
                    with open(p, "r") as f:
                        e0_list.append(int(f.read().strip()))
            except Exception as ex:
                print(f"[!] Error al leer RAPL inicial en FFT: {ex}", file=sys.stderr)
                rapl_paths = []
        else:
            warn_rapl_missing_once()

    start_wall = time.perf_counter()
    try:
        proc_pwr = subprocess.run(
            cmd_pwr,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join()

    end_wall = time.perf_counter()
    samples = []
    
    if proc_pwr.returncode != 0:
        raise RuntimeError("Fallo en ejecucion de monitoreo de FFT.")
    if time_sec <= 0.0:
        time_sec = 1e-9

    avg_power_w = 0.0
    energy_j = 0.0
    wall_time = end_wall - start_wall
    power_window_sec = wall_time

    if device == "gpu":
        samples = power_queue.get() if not power_queue.empty() else []
        avg_power_w, energy_total_j = average_and_energy_from_samples(samples)
        energy_j = energy_total_j / K
        if samples:
            power_window_sec = samples[-1][0] - samples[0][0]
    else:
        if rapl_paths:
            try:
                t1 = time.perf_counter()
                energy_total_j = 0.0
                for i, p in enumerate(rapl_paths):
                    with open(p, "r") as f:
                        val = int(f.read().strip())
                    diff = max(0.0, (val - e0_list[i]) / 1e6)
                    energy_total_j += diff

                power_window_sec = t1 - t0
                if power_window_sec <= 0.0:
                    power_window_sec = wall_time
                
                # Deduce total idle energy consumption during the process window
                energy_idle = IDLE_POWER_CPU * power_window_sec
                energy_active = max(0.0, energy_total_j - energy_idle)
                
                # Calculate active power over the active computation loop (excluding startup/IO)
                t_active = K * time_sec
                avg_power_w = energy_active / t_active if t_active > 0.0 else 0.0
                energy_j = energy_active / K if K > 0 else 0.0
            except Exception as ex:
                print(f"[!] Error al leer RAPL final en FFT: {ex}", file=sys.stderr)
                avg_power_w = 0.0
                energy_j = 0.0
        else:
            avg_power_w = 0.0
            energy_j = 0.0

    dims = fft_dims(nx, ny, nz)
    ops = fft_flops(dims, domain) * batch
    gflops = (ops / time_sec) / 1e9
    edp = energy_j * time_sec
    payload_bytes = fft_payload_bytes(dims, batch, precision, domain, layout)
    radix_class = fft_radix_class(dims)

    return {
        "Device": device,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "Batch": batch,
        "Precision": precision,
        "Domain": domain,
        "Direction": direction,
        "Layout": layout,
        "Time_sec": time_sec,
        "GFLOPS": gflops,
        "Avg_Power_W": avg_power_w,
        "Energy_J": energy_j,
        "EDP": edp,
        "Payload_Bytes": payload_bytes,
        "Radix_Class": radix_class,
        "Samples_Power": len(samples) if device == "gpu" else (2 * len(rapl_paths) if rapl_paths else 0),
        "Wall_Elapsed_sec": end_wall - start_wall,
    }


def run_fft_warmup(
    binary,
    nx,
    ny,
    nz,
    batch,
    precision,
    domain,
    direction,
    layout,
    plan,
    warmup,
    timeout,
    matrix_file,
):
    cmd = [
        binary,
        str(nx),
        str(ny),
        str(nz),
        str(batch),
        precision,
        domain,
        direction,
        layout,
        str(warmup),
        "0",
    ]
    if plan is not None:
        cmd.append(plan)
    if matrix_file:
        cmd.append(matrix_file)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Fallo en warmup FFT.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def init_nvml_if_needed(device_list, gpu_index):
    if "gpu" not in device_list:
        return
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if gpu_index < 0 or gpu_index >= device_count:
            raise RuntimeError(
                f"gpu-index invalido: {gpu_index}. GPUs disponibles: {device_count}"
            )
    except Exception:
        pynvml.nvmlShutdown()
        raise


def run_gemm(args):
    if args.device not in {"gpu", "cpu", "both"}:
        raise ValueError("Para GEMM, --device debe ser cpu, gpu o both")
    if args.gemm_warmup < 0:
        raise ValueError("--gemm-warmup no puede ser negativo")

    if args.mode == "continuous-rl":
        generator = RLWorkloadGenerator(
            algorithm="gemm",
            gemm_min_n=args.gemm_min_n,
            gemm_max_n=args.gemm_max_n,
            gemm_low_step=args.gemm_low_step,
            gemm_trans_step=args.gemm_trans_step,
            gemm_high_step=args.gemm_high_step,
        )
        sizes = generator.generate()
    else:
        sizes = parse_sizes(args.sizes)
    precisions = parse_precisions(args.precisions)
    default_op = "N,T,C" if (args.mode == "continuous-rl" or args.sweep_transpose) else "N"
    raw_op_a = args.op_a_list if args.op_a_list is not None else default_op
    raw_op_b = args.op_b_list if args.op_b_list is not None else default_op
    op_a_list = parse_ops(raw_op_a)
    op_b_list = parse_ops(raw_op_b)

    output_path = args.output or "benchmark_results.csv"

    if args.device == "both":
        devices = ["cpu", "gpu"]
    else:
        devices = [args.device]

    if "gpu" in devices:
        init_nvml_if_needed(devices, args.gpu_index)

    try:
        fieldnames = [
            "Device",
            "M",
            "N",
            "K",
            "Precision",
            "OpA",
            "OpB",
            "Iteration",
            "Time_sec",
            "GFLOPS",
            "Avg_Power_W",
            "Energy_J",
            "EDP",
        ]

        if args.full_dim_sweep:
            dim_cases = list(itertools.product(sizes, sizes, sizes))
        else:
            dim_cases = [(s, s, s) for s in sizes]

        total = len(dim_cases) * len(precisions) * len(op_a_list) * len(op_b_list) * len(devices) * args.repetitions
        done = 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m, n, k in dim_cases:
                for precision, op_a, op_b in itertools.product(precisions, op_a_list, op_b_list):
                    for device in devices:
                        binary = args.gemm_binary_gpu if device == "gpu" else args.gemm_binary_cpu
                        for rep in range(args.repetitions):
                            done += 1
                            
                            # 1. Warm-ups (Python-controlled flat loop)
                            # Only execute warmups on the first repetition if not explicitly requested on CLI
                            if args.is_warmup:
                                # If called via CLI with --is-warmup, only execute 1 warmup call and proceed/skip measurement
                                run_single_case(
                                    binary,
                                    device,
                                    args.gpu_index,
                                    m,
                                    n,
                                    k,
                                    precision,
                                    op_a,
                                    op_b,
                                    args.timeout,
                                    True, # is_warmup
                                    args.seed if args.seed else None,
                                    args.benchmark_bank,
                                    args.gemm_profile,
                                    databank_dir=args.databank_dir,
                                    databank_max_n=args.databank_max_n,
                                )
                                print(f"[{done}/{total}] {device.upper()} M={m} N={n} K={k} P={precision} OpA={op_a} OpB={op_b} Rep={rep} [WARMUP ONLY]")
                                continue
                            else:
                                warmup_count = args.gemm_warmup if rep == 0 else 0
                                for _ in range(warmup_count):
                                    run_single_case(
                                        binary,
                                        device,
                                        args.gpu_index,
                                        m,
                                        n,
                                        k,
                                        precision,
                                        op_a,
                                        op_b,
                                        args.timeout,
                                        True, # is_warmup
                                        args.seed if args.seed else None,
                                        args.benchmark_bank,
                                        args.gemm_profile,
                                        databank_dir=args.databank_dir,
                                        databank_max_n=args.databank_max_n,
                                    )

                                # 2. Measurement (is_warmup = False)
                                result = run_single_case(
                                    binary,
                                    device,
                                    args.gpu_index,
                                    m,
                                    n,
                                    k,
                                    precision,
                                    op_a,
                                    op_b,
                                    args.timeout,
                                    False, # is_warmup
                                    args.seed if args.seed else None,
                                    args.benchmark_bank,
                                    args.gemm_profile,
                                    databank_dir=args.databank_dir,
                                    databank_max_n=args.databank_max_n,
                                )

                            # Include Device and Iteration in the written row
                            row = {key: result.get(key, 0.0) for key in fieldnames if key not in ["Device", "Iteration"]}
                            row["Device"] = device
                            row["Iteration"] = rep
                            writer.writerow(row)
                            f.flush()

                            print(
                                f"[{done}/{total}] {device.upper()} M={m} N={n} K={k} P={precision} OpA={op_a} OpB={op_b} "
                                f"Rep={rep} Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                                f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                                f"EDP={result['EDP']:.9f}"
                            )

        print(f"\nResultados guardados en: {output_path}")
    finally:
        if "gpu" in devices:
            pynvml.nvmlShutdown()


def run_fft(args):
    if args.mode == "continuous-rl":
        shapes = []
        if args.fft_sizes_1d and args.fft_sizes_1d.strip():
            shapes.extend(parse_fft_shapes(args.fft_sizes_1d, 1))
        if args.fft_sizes_2d and args.fft_sizes_2d.strip():
            shapes.extend(parse_fft_shapes(args.fft_sizes_2d, 2))
        if args.fft_sizes_3d and args.fft_sizes_3d.strip():
            shapes.extend(parse_fft_shapes(args.fft_sizes_3d, 3))

        if not shapes:
            generator = RLWorkloadGenerator(
                algorithm="fft",
                fft_min_n=args.fft_min_n,
                fft_max_n=args.fft_max_n,
                fft_low_step=args.fft_low_step,
                fft_mid_step=args.fft_mid_step,
                fft_high_step=args.fft_high_step,
            )
            sizes = generator.generate()
            shapes = [(n, 0, 0) for n in sizes]
    else:
        sizes_1d = parse_fft_shapes(args.fft_sizes_1d, 1)
        sizes_2d = parse_fft_shapes(args.fft_sizes_2d, 2)
        sizes_3d = parse_fft_shapes(args.fft_sizes_3d, 3)
        shapes = sizes_1d + sizes_2d + sizes_3d
        if not shapes:
            raise ValueError("No se definieron tamanos FFT (1D/2D/3D)")

    batches = parse_int_list(args.fft_batches, "batches")
    precisions = parse_fft_precisions(args.fft_precisions)
    domains = parse_fft_domains(args.fft_domains)
    directions = parse_fft_directions(args.fft_directions)
    layouts = parse_fft_layouts(args.fft_layouts)

    if args.device == "both":
        devices = ["cpu", "gpu"]
    else:
        devices = [args.device]

    output_path = args.output or "fft_benchmark_results.csv"
    init_nvml_if_needed(devices, args.gpu_index)

    try:
        fieldnames = [
            "Device",
            "Nx",
            "Ny",
            "Nz",
            "Batch",
            "Precision",
            "Domain",
            "Direction",
            "Layout",
            "Iteration",
            "Time_sec",
            "GFLOPS",
            "Avg_Power_W",
            "Energy_J",
            "EDP",
            "Payload_Bytes",
            "Radix_Class",
            "Samples_Power",
        ]

        cases = []
        for nx, ny, nz in shapes:
            for batch in batches:
                for precision in precisions:
                    for domain in domains:
                        if domain == "C2C":
                            dir_list = directions
                        elif domain == "R2C":
                            dir_list = ["F"]
                        else:
                            dir_list = ["I"]
                        for direction in dir_list:
                            for layout in layouts:
                                cases.append((nx, ny, nz, batch, precision, domain, direction, layout))

        total = len(cases) * len(devices) * args.repetitions
        done = 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for nx, ny, nz, batch, precision, domain, direction, layout in cases:
                for device in devices:
                    binary = args.fft_binary_gpu if device == "gpu" else args.fft_binary_cpu
                    matrix_file, _fft_file_is_persistent = generate_fft_matrix_file(
                        nx,
                        ny,
                        nz,
                        batch,
                        precision,
                        domain,
                        layout,
                        seed=args.seed,
                        bank_path=args.benchmark_bank,
                        bank_profile=args.fft_profile,
                        databank_dir=args.databank_dir,
                        databank_max_n=args.databank_max_n,
                    )
                    
                    try:
                        for rep in range(args.repetitions):
                            done += 1
                            
                            # 1. Warm-ups (Python-controlled flat loop)
                            # Only execute warmups on the first repetition if not explicitly requested on CLI
                            if args.is_warmup:
                                # If called via CLI with --is-warmup, only execute 1 warmup call and proceed/skip measurement
                                run_single_case_fft(
                                    binary,
                                    device,
                                    args.gpu_index,
                                    nx,
                                    ny,
                                    nz,
                                    batch,
                                    precision,
                                    domain,
                                    direction,
                                    layout,
                                    args.fft_plan,
                                    True, # is_warmup
                                    args.timeout,
                                    matrix_file,
                                )
                                print(f"[{done}/{total}] {device.upper()} Nx={nx} Ny={ny} Nz={nz} Batch={batch} P={precision} D={domain} Dir={direction} L={layout} Rep={rep} [WARMUP ONLY]")
                                continue
                            else:
                                warmup_count = args.fft_warmup if rep == 0 else 0
                                for _ in range(warmup_count):
                                    run_single_case_fft(
                                        binary,
                                        device,
                                        args.gpu_index,
                                        nx,
                                        ny,
                                        nz,
                                        batch,
                                        precision,
                                        domain,
                                        direction,
                                        layout,
                                        args.fft_plan,
                                        True, # is_warmup
                                        args.timeout,
                                        matrix_file,
                                    )

                                # 2. Measurement (is_warmup = False)
                                result = run_single_case_fft(
                                    binary,
                                    device,
                                    args.gpu_index,
                                    nx,
                                    ny,
                                    nz,
                                    batch,
                                    precision,
                                    domain,
                                    direction,
                                    layout,
                                    args.fft_plan,
                                    False, # is_warmup
                                    args.timeout,
                                    matrix_file,
                                )

                            row = {key: result.get(key, 0.0) for key in fieldnames if key not in ["Device", "Iteration"]}
                            row["Device"] = device
                            row["Iteration"] = rep
                            writer.writerow(row)
                            f.flush()

                            print(
                                f"[{done}/{total}] {device.upper()} Nx={nx} Ny={ny} Nz={nz} Batch={batch} "
                                f"P={precision} D={domain} Dir={direction} L={layout} Rep={rep} "
                                f"Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                                f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                                f"EDP={result['EDP']:.9f}"
                            )

                    finally:
                        # Solo eliminar si es temporal (no del DataBankManager).
                        if matrix_file and not _fft_file_is_persistent and os.path.exists(matrix_file):
                            os.unlink(matrix_file)

        print(f"\nResultados guardados en: {output_path}")
    finally:
        if "gpu" in devices:
            pynvml.nvmlShutdown()


def main():
    global IDLE_POWER_CPU
    parser = argparse.ArgumentParser(
        description="Orquestador de benchmarking GEMM/FFT con monitoreo de potencia"
    )
    parser.add_argument(
        "--benchmark",
        choices=["gemm", "fft"],
        default="gemm",
        help="Selecciona el benchmark a ejecutar (gemm|fft)",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "continuous-rl"],
        default="standard",
        help="Modo de ejecucion del benchmark (standard|continuous-rl)",
    )
    parser.add_argument(
        "--gemm-min-n",
        type=int,
        default=64,
        help="Limite inferior para GEMM en modo continuous-rl (por defecto: 64)",
    )
    parser.add_argument(
        "--gemm-max-n",
        type=int,
        default=16384,
        help="Limite superior para GEMM en modo continuous-rl (por defecto: 16384)",
    )
    parser.add_argument(
        "--gemm-low-step",
        type=int,
        choices=[32, 64, 128],
        default=32,
        help="Paso base en rango de baja latencia para GEMM en modo continuous-rl (32|64|128, por defecto: 32)",
    )
    parser.add_argument(
        "--gemm-trans-step",
        type=int,
        default=512,
        help="Paso en rango de transicion para GEMM en modo continuous-rl (por defecto: 512, ignorado con dynamic steps)",
    )
    parser.add_argument(
        "--gemm-high-step",
        type=int,
        default=1024,
        help="Paso en rango intensivo para GEMM en modo continuous-rl (por defecto: 1024, ignorado con dynamic steps)",
    )
    parser.add_argument(
        "--fft-min-n",
        type=int,
        default=4096,
        help="Limite inferior para FFT en modo continuous-rl (por defecto: 4096)",
    )
    parser.add_argument(
        "--fft-max-n",
        type=int,
        default=67108864,
        help="Limite superior para FFT en modo continuous-rl (por defecto: 67108864)",
    )
    parser.add_argument(
        "--fft-low-step",
        type=int,
        choices=[128, 256, 512],
        default=256,
        help="Paso base en rango de baja latencia para FFT en modo continuous-rl (128|256|512, por defecto: 256)",
    )
    parser.add_argument(
        "--fft-mid-step",
        type=int,
        default=4096,
        help="Paso en rango de transicion para FFT en modo continuous-rl (por defecto: 4096)",
    )
    parser.add_argument(
        "--fft-high-step",
        type=int,
        default=262144,
        help="Paso en rango intensivo para FFT en modo continuous-rl (por defecto: 262144)",
    )
    parser.add_argument("--gemm-binary-cpu", default="./algoritmos/gemm_cpu", help="Ruta al binario GEMM CPU (BLAS)")
    parser.add_argument("--gemm-binary-gpu", default="./algoritmos/gemm_gpu", help="Ruta al binario GEMM GPU (cuBLAS)")
    parser.add_argument("--binary", default=None, help="(Deprecado) Alias de --gemm-binary-gpu")
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu", "both"],
        default="gpu",
        help="Dispositivo donde ejecutar el benchmark (gpu|cpu|both)",
    )
    parser.add_argument(
        "--sizes",
        default="128,256,512,1024,2048,4096",
        help="Lista separada por comas para M,N,K (GEMM)",
    )
    parser.add_argument(
        "--precisions",
        default="S,D,C,Z",
        help="Lista separada por comas de precisiones (GEMM): S,D,C,Z",
    )
    parser.add_argument(
        "--full-dim-sweep",
        action="store_true",
        help="Activa el barrido completo de M, N y K (GEMM)",
    )
    parser.add_argument(
        "--sweep-transpose",
        action="store_true",
        help="Activa el barrido de transposicion para opA/opB (GEMM)",
    )
    parser.add_argument(
        "--op-a-list",
        default=None,
        help="Lista separada por comas para opA: N,T,C (GEMM)",
    )
    parser.add_argument(
        "--op-b-list",
        default=None,
        help="Lista separada por comas para opB: N,T,C (GEMM)",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="Indice de GPU para NVML")
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo CSV de salida (por defecto: benchmark_results.csv o fft_benchmark_results.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout por ejecucion en segundos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla fija para matrices (se exporta como BENCH_SEED)",
    )
    parser.add_argument(
        "--benchmark-bank",
        default=DEFAULT_BENCHMARK_BANK,
        help="Ruta al banco HDF5 legacy (fallback si DataBankManager falla)",
    )
    parser.add_argument(
        "--gemm-profile",
        default="dense_normal",
        help="Perfil de generacion de datos (dense_normal, dense_uniform, ill_conditioned)",
    )
    parser.add_argument(
        "--fft-profile",
        default="broadband",
        help="Perfil de generacion de datos FFT (broadband, single_tone, multi_tone)",
    )
    parser.add_argument(
        "--databank-dir",
        default=DEFAULT_DATABANK_DIR,
        help="Raiz del banco de datos binario (DataBankManager)",
    )
    parser.add_argument(
        "--databank-max-n",
        type=int,
        default=DEFAULT_DATABANK_MAX_N,
        help="Techo de N para el rango compute-intensive del banco binario",
    )
    parser.add_argument(
        "--idle-power-cpu",
        type=float,
        default=IDLE_POWER_CPU,
        help="Potencia de CPU en reposo (idle) en Watts.",
    )
    parser.add_argument(
        "--gemm-warmup",
        type=int,
        default=4,
        help="Ejecuciones de warmup previas a GEMM",
    )
    parser.add_argument(
        "--is-warmup",
        action="store_true",
        help="Si se activa, el benchmark solo ejecutara la fase de calentamiento (warmup)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Numero de repeticiones continuas de cada caso de prueba (para analisis estadistico)",
    )

    parser.add_argument(
        "--fft-binary-cpu",
        default="./algoritmos/fft_cpu",
        help="Ruta al binario FFT CPU",
    )
    parser.add_argument(
        "--fft-binary-gpu",
        default="./algoritmos/fft_gpu",
        help="Ruta al binario FFT GPU",
    )
    parser.add_argument(
        "--fft-sizes-1d",
        default="512,1024,2048,4096,8192,16384,3072,5120,6144,10240",
        help="Lista de tamanos 1D FFT (ej: 512,1024)",
    )
    parser.add_argument(
        "--fft-sizes-2d",
        default="32x32,64x64,128x128,48x48,96x96",
        help="Lista de tamanos 2D FFT (ej: 64x64,128x128)",
    )
    parser.add_argument(
        "--fft-sizes-3d",
        default="16x16x16,32x32x32,24x24x24",
        help="Lista de tamanos 3D FFT (ej: 16x16x16)",
    )
    parser.add_argument(
        "--fft-batches",
        default="1",
        help="Lista de batches FFT (ej: 1,2,4)",
    )
    parser.add_argument(
        "--fft-precisions",
        default="S,D",
        help="Lista de precisiones FFT: S,D",
    )
    parser.add_argument(
        "--fft-domains",
        default="C2C,R2C,C2R",
        help="Lista de dominios FFT: C2C,R2C,C2R",
    )
    parser.add_argument(
        "--fft-directions",
        default="F,I",
        help="Lista de direcciones FFT: F,I",
    )
    parser.add_argument(
        "--fft-layouts",
        default="I,O",
        help="Lista de layouts FFT: I,O",
    )
    parser.add_argument(
        "--fft-plan",
        choices=["E", "M"],
        default="E",
        help="Plan FFT: E=ESTIMATE, M=MEASURE",
    )
    parser.add_argument(
        "--fft-warmup",
        type=int,
        default=4,
        help="Iteraciones de warmup FFT",
    )
    parser.add_argument(
        "--fft-iters",
        type=int,
        default=1,
        help="Iteraciones medidas FFT",
    )
    args = parser.parse_args()

    IDLE_POWER_CPU = args.idle_power_cpu

    # Compatibilidad: --binary sobreescribe --gemm-binary-gpu
    if args.binary is not None:
        args.gemm_binary_gpu = args.binary

    if args.benchmark == "gemm":
        run_gemm(args)
    else:
        run_fft(args)


if __name__ == "__main__":
    main()
