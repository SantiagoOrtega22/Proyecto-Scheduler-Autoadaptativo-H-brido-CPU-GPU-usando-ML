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
    Time_sec        -> Tiempo puro de ejecución del kernel, medido en fase de aislamiento de métricas.
    GFLOPS          -> Rendimiento calculado a partir de las dimensiones, la precisión y Time_sec.
    Avg_Power_W     -> Potencia media ACTIVA (ya descontada la potencia en reposo) durante el lazo medido.
    Energy_J        -> Energía de UNA ejecución: Avg_Power_W * Time_sec.
    EDP             -> Producto Energía-Retardo (Energy-Delay Product = Energy_J * Time_sec).
    Loop_Window_sec -> Duración real de la ventana sobre la que se integró la telemetría.
    Iters_K         -> Iteraciones ejecutadas dentro de esa ventana.
    Power_Samples   -> Muestras crudas de RAPL/NVML que cayeron dentro de la ventana.
    Idle_Power_W    -> Línea base en reposo descontada para obtener la potencia activa.
    Wall_Elapsed_sec-> Duración total del subproceso (incluye setup y cierre; solo auditoría).

Las cinco últimas columnas existen para poder auditar a posteriori que la energía se
integró sobre el lazo y no sobre el proceso completo.

NOTAS HPC Y RIGOR
-----------------
    - Aislamiento de Métricas: Cada prueba ejecuta el binario dos veces. La primera (sin hilos de monitoreo) obtiene el tiempo exacto; la segunda (con hilos de lectura NVML/RAPL activos) extrae el perfil energético.
    - Warm-ups: Se corren iteraciones previas (por defecto 4) para inicializar bibliotecas (cuBLAS/FFTW) y estabilizar relojes/Turbo Boost.
    - Ventana de medición: el muestreo cubre todo el subproceso, pero la energía se integra
      SOLO entre las marcas LOOP_WINDOW que publican los binarios, que delimitan el lazo
      cronometrado. Sin ese recorte, la energía del arranque del proceso, la creación del
      plan, las reservas de memoria y el cierre se atribuía al kernel: en GPU diluía la
      potencia hacia el reposo y en CPU la inflaba por encima del límite físico del socket.
      Requiere binarios recompilados; si falta la marca, la medición se omite con aviso.
    - Línea base en reposo: se mide la potencia idle de CPU y GPU al inicio del barrido
      (o se fija con --idle-power-cpu / --idle-power-gpu) y se descuenta de la ventana,
      de modo que Avg_Power_W refleja solo el consumo atribuible al cómputo. El descuento
      es simétrico entre dispositivos para que la comparación CPU/GPU sea justa.
    - Coherencia interna: por construcción Energy_J = Avg_Power_W * Time_sec y
      EDP = Energy_J * Time_sec, con Time_sec proveniente de la fase sin telemetría.
    - Consumo en GPU: muestreo continuo de nvmlDeviceGetPowerUsage() e integración
      trapezoidal de la curva de potencia recortada a la ventana del lazo.
    - Consumo en CPU: muestreo continuo de los contadores acumulados Intel RAPL
      (/sys/class/powercap, todos los sockets package), interpolando el contador en
      ambos bordes de la ventana y tomando la diferencia.
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
from typing import Dict, List, Optional, Sequence, Tuple

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
_RAPL_RANGE_WARNING_SHOWN = False

# Intervalo de muestreo NVML. El sensor de potencia de la GPU tiene su propia latencia
# de refresco (~10-20 ms), asi que muestrear mas rapido no aporta resolucion real.
POWER_SAMPLE_INTERVAL_SEC = 0.02
# RAPL actualiza energy_uj cada ~1 ms. Como es un CONTADOR ACUMULADO y no una lectura
# instantanea, muestrear mas lento que esa tasa no pierde energia: el contador integra
# todo lo ocurrido entre lecturas. El intervalo solo fija la precision con la que se
# interpolan los BORDES de la ventana, con un error acotado por (intervalo x salto de
# potencia en el borde) -> ~0.4% en una ventana de 0.5 s. Bajarlo a 1 ms no compensa:
# multiplicaria por cinco las lecturas de sysfs, perturbando la propia medicion, y no
# puede superar la granularidad de 1 ms del contador.
RAPL_SAMPLE_INTERVAL_SEC = 0.005
# Duracion objetivo del lazo bajo monitoreo. Cuanto mas larga, mejor relacion
# senal-ruido de la telemetria, a costa de tiempo total de barrido.
POWER_WINDOW_TARGET_SEC = 0.5
# Minimo de muestras crudas dentro de la ventana para aceptar la medicion.
MIN_SAMPLES_IN_WINDOW = 2
# Divergencia tolerada entre el tiempo por iteracion de la fase de aislamiento y el
# de la fase de potencia. Superarla no invalida el dato, pero avisa de que el estado
# del dispositivo (turbo, cache, relojes) no fue estable entre ambas fases.
LOOP_TIME_DIVERGENCE_WARN = 0.25
# Margen bajo la linea base de reposo que se acepta como ruido antes de invalidar la
# medicion. Por debajo de esto la linea base es fisicamente imposible.
IDLE_BASELINE_TOLERANCE = 0.99
# Potencias en reposo. Se restan para reportar solo la potencia atribuible al computo;
# se miden al inicio del barrido salvo que se fijen explicitamente por CLI.
IDLE_POWER_CPU = 0.0
IDLE_POWER_GPU = 0.0

# Marca que publican los binarios para delimitar el lazo cronometrado. Sin ella la
# telemetria abarcaria tambien el arranque del proceso, la creacion del plan, las
# reservas de memoria y el cierre, cuya energia no pertenece al kernel medido.
LOOP_WINDOW_PATTERN = re.compile(
    r"LOOP_WINDOW\s+start=([0-9.eE+-]+)\s+end=([0-9.eE+-]+)\s+iters=([0-9]+)"
)

DEFAULT_DATABANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_files", "databank")
DEFAULT_DATABANK_MAX_N = 67108864


def warn_rapl_missing_once():
    global _RAPL_WARNING_SHOWN
    if _RAPL_WARNING_SHOWN:
        return
    print(
        "Aviso: no se encontro energy_uj RAPL legible en /sys/class/powercap; "
        "las mediciones de CPU se omitiran por falta de telemetria.",
        file=sys.stderr,
    )
    _RAPL_WARNING_SHOWN = True


def warn_rapl_range_missing_once(path: str) -> None:
    """Avisa una sola vez si no se puede leer el rango de vuelta del contador RAPL.

    Sin max_energy_range_uj no se puede reconstruir la energia de un reinicio del
    contador, asi que esa fraccion se perderia en silencio.

    Args:
        path: Ruta del contador energy_uj afectado.
    """
    global _RAPL_RANGE_WARNING_SHOWN
    if _RAPL_RANGE_WARNING_SHOWN:
        return
    print(
        f"Aviso: no se pudo leer max_energy_range_uj para {path}; si el contador da la "
        "vuelta durante una medicion, esa energia no podra reconstruirse.",
        file=sys.stderr,
    )
    _RAPL_RANGE_WARNING_SHOWN = True


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

    # El lazo medido termina antes de que el proceso salga, asi que tomamos una muestra
    # posterior a la senal de parada para poder interpolar el borde derecho de la ventana.
    try:
        raw_samples.append((time.perf_counter(), int(pynvml.nvmlDeviceGetPowerUsage(handle))))
    except Exception:
        pass

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


class MeasurementError(RuntimeError):
    """Fallo recuperable de telemetria: la iteracion se salta sin abortar el barrido."""


def parse_loop_window(stdout: str, context: str) -> Tuple[float, float, int]:
    """Extrae la ventana del lazo cronometrado que publica el binario.

    Args:
        stdout: Salida estandar de la ejecucion monitorizada.
        context: Descripcion del caso, usada en los mensajes de error.

    Returns:
        Tupla (inicio, fin, iteraciones) con los instantes CLOCK_MONOTONIC que
        delimitan exactamente el lazo medido.

    Raises:
        MeasurementError: Si el binario no publico la marca LOOP_WINDOW (binario sin
            recompilar) o si la ventana resulta degenerada.
    """
    match = LOOP_WINDOW_PATTERN.search(stdout)
    if not match:
        raise MeasurementError(
            f"El binario no publico LOOP_WINDOW para {context}. Recompila los binarios: "
            "sin esa marca la telemetria no puede acotarse al lazo medido."
        )

    loop_start = float(match.group(1))
    loop_end = float(match.group(2))
    loop_iters = int(match.group(3))
    if loop_end <= loop_start:
        raise MeasurementError(
            f"Ventana de lazo degenerada para {context}: start={loop_start}, end={loop_end}"
        )
    return loop_start, loop_end, loop_iters


def interpolate_series(samples: Sequence[Tuple[float, float]], t: float) -> float:
    """Interpola linealmente el valor de una serie temporizada en el instante t.

    Args:
        samples: Muestras (timestamp, valor) ordenadas por timestamp.
        t: Instante en el que se quiere evaluar la serie.

    Returns:
        Valor interpolado; se satura al primer/ultimo valor fuera del rango muestreado.
    """
    if t <= samples[0][0]:
        return samples[0][1]
    if t >= samples[-1][0]:
        return samples[-1][1]

    for (t_a, v_a), (t_b, v_b) in zip(samples, samples[1:]):
        if t_a <= t <= t_b:
            if t_b == t_a:
                return v_b
            return v_a + (v_b - v_a) * ((t - t_a) / (t_b - t_a))
    return samples[-1][1]


def energy_from_power_samples(
    samples: Sequence[Tuple[float, float]], t_start: float, t_end: float
) -> Tuple[float, int]:
    """Integra muestras de potencia (NVML) restringidas a una ventana temporal.

    Args:
        samples: Muestras (timestamp, potencia_w) ordenadas por timestamp.
        t_start: Inicio de la ventana de integracion.
        t_end: Fin de la ventana de integracion.

    Returns:
        Tupla (energia_j, muestras_crudas_dentro_de_la_ventana). Los bordes se
        interpolan para no truncar ni extender la ventana.
    """
    samples = sorted(samples, key=lambda item: item[0])
    interior = [s for s in samples if t_start < s[0] < t_end]
    bounded = (
        [(t_start, interpolate_series(samples, t_start))]
        + interior
        + [(t_end, interpolate_series(samples, t_end))]
    )

    energy_j = 0.0
    for (t_a, p_a), (t_b, p_b) in zip(bounded, bounded[1:]):
        dt = t_b - t_a
        if dt > 0.0:
            energy_j += (p_a + p_b) * 0.5 * dt
    return energy_j, len(interior)


def energy_from_counter_samples(
    samples: Sequence[Tuple[float, float]], t_start: float, t_end: float
) -> Tuple[float, int]:
    """Calcula la energia consumida en una ventana a partir de un contador acumulado (RAPL).

    Args:
        samples: Muestras (timestamp, energia_acumulada_j) ordenadas por timestamp.
        t_start: Inicio de la ventana.
        t_end: Fin de la ventana.

    Returns:
        Tupla (energia_j, muestras_crudas_dentro_de_la_ventana). El contador se
        interpola en ambos bordes y se toma la diferencia.
    """
    samples = sorted(samples, key=lambda item: item[0])
    interior = [s for s in samples if t_start < s[0] < t_end]
    energy_j = interpolate_series(samples, t_end) - interpolate_series(samples, t_start)
    return max(0.0, energy_j), len(interior)


def monitor_energy_cpu(
    rapl_paths: Sequence[str], stop_event: threading.Event, sample_queue: queue.Queue
) -> None:
    """Muestrea de forma continua los contadores RAPL acumulados de todos los sockets.

    A diferencia de una lectura inicial/final, el muestreo continuo permite recortar
    despues la energia a la ventana exacta del lazo medido.

    Args:
        rapl_paths: Rutas a los archivos energy_uj de cada dominio package.
        stop_event: Evento que detiene el muestreo.
        sample_queue: Cola donde se publica la tupla (muestras, vueltas_irrecuperables);
            cada muestra es (timestamp, energia_acumulada_j).
    """
    max_ranges: List[Optional[int]] = []
    for path in rapl_paths:
        try:
            with open(path.replace("energy_uj", "max_energy_range_uj"), "r") as f_max:
                max_ranges.append(int(f_max.read().strip()))
        except Exception:
            max_ranges.append(None)
            warn_rapl_range_missing_once(path)

    last_raw: List[Optional[int]] = [None] * len(rapl_paths)
    accumulated_uj: List[int] = [0] * len(rapl_paths)
    samples: List[Tuple[float, float]] = []
    # Instantes en los que el contador dio la vuelta sin que se pudiera reconstruir la
    # energia perdida. El consumidor decide si invalidan la medicion segun caigan dentro
    # o fuera de la ventana del lazo.
    lost_wraps: List[float] = []

    def take_sample() -> None:
        timestamp = time.perf_counter()
        total_uj = 0
        for index, path in enumerate(rapl_paths):
            try:
                with open(path, "r") as f_energy:
                    raw = int(f_energy.read().strip())
            except Exception:
                # Lectura puntual fallida: descartamos la muestra completa para no
                # introducir un salto artificial en el contador acumulado.
                return
            previous = last_raw[index]
            if previous is not None:
                delta = raw - previous
                if delta < 0:
                    # Reinicio del contador. El hardware no lo senaliza de ninguna forma
                    # (en un Xeon Silver 4314 el ciclo completo toma ~52 min bajo carga
                    # mixta), asi que solo el muestreo continuo permite detectarlo.
                    # Al muestrear cada RAPL_SAMPLE_INTERVAL_SEC
                    # solo puede haber ocurrido una vuelta entre dos lecturas (darlas dos
                    # veces en milisegundos exigiria una potencia irreal), asi que una sola
                    # correccion basta. La acumulacion se hace ANTES de interpolar, de modo
                    # que la serie entregada es monotona y el reinicio nunca aparece como
                    # un salto dentro de la ventana.
                    max_range = max_ranges[index]
                    if max_range:
                        delta = raw + max_range - previous
                    else:
                        # Sin el rango no hay forma de saber cuanta energia hubo entre
                        # `previous` y la vuelta: se anota para invalidar la medicion en
                        # lugar de reportar en silencio un valor subestimado.
                        delta = 0
                        lost_wraps.append(timestamp)
                accumulated_uj[index] += delta
            last_raw[index] = raw
            total_uj += accumulated_uj[index]
        samples.append((timestamp, total_uj / 1e6))

    while True:
        take_sample()
        if stop_event.wait(RAPL_SAMPLE_INTERVAL_SEC):
            break

    # Muestra final posterior a la parada: cubre el borde derecho de la ventana.
    take_sample()
    sample_queue.put((samples, lost_wraps))


def measure_idle_power_gpu(gpu_index: int, duration_sec: float) -> float:
    """Mide la potencia en reposo de la GPU con NVML.

    Args:
        gpu_index: Indice del dispositivo NVML.
        duration_sec: Duracion del muestreo en reposo.

    Returns:
        Potencia media en Watts; 0.0 si la medicion no fue posible.
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        stop_event = threading.Event()
        sample_queue: queue.Queue = queue.Queue(maxsize=1)
        thread = threading.Thread(
            target=monitor_power_gpu, args=(handle, stop_event, sample_queue), daemon=True
        )
        thread.start()
        time.sleep(duration_sec)
        stop_event.set()
        thread.join()
        samples = sample_queue.get() if not sample_queue.empty() else []
    except Exception as ex:
        print(f"[!] No se pudo medir la potencia idle de GPU: {ex}", file=sys.stderr)
        return 0.0

    if len(samples) < 2:
        return 0.0
    energy_j, _ = energy_from_power_samples(samples, samples[0][0], samples[-1][0])
    span = samples[-1][0] - samples[0][0]
    return energy_j / span if span > 0.0 else 0.0


def measure_idle_power_cpu(rapl_paths: Sequence[str], duration_sec: float) -> float:
    """Mide la potencia en reposo de los sockets de CPU con RAPL.

    Args:
        rapl_paths: Rutas a los contadores energy_uj de cada package.
        duration_sec: Duracion del muestreo en reposo.

    Returns:
        Potencia media en Watts; 0.0 si la medicion no fue posible.
    """
    if not rapl_paths:
        return 0.0
    try:
        stop_event = threading.Event()
        sample_queue: queue.Queue = queue.Queue(maxsize=1)
        thread = threading.Thread(
            target=monitor_energy_cpu, args=(rapl_paths, stop_event, sample_queue), daemon=True
        )
        thread.start()
        time.sleep(duration_sec)
        stop_event.set()
        thread.join()
        payload = sample_queue.get() if not sample_queue.empty() else None
        samples, lost_wraps = payload if payload else ([], [])
        if lost_wraps:
            print(
                "[!] El contador RAPL dio la vuelta al medir el idle de CPU sin poder "
                "reconstruir la energia; se omite la linea base.",
                file=sys.stderr,
            )
            return 0.0
    except Exception as ex:
        print(f"[!] No se pudo medir la potencia idle de CPU: {ex}", file=sys.stderr)
        return 0.0

    if len(samples) < 2:
        return 0.0
    span = samples[-1][0] - samples[0][0]
    if span <= 0.0:
        return 0.0
    energy_j, _ = energy_from_counter_samples(samples, samples[0][0], samples[-1][0])
    return energy_j / span


def configure_idle_baselines(
    devices: Sequence[str],
    measure_sec: float,
    cpu_override: Optional[float],
    gpu_override: Optional[float],
    gpu_index: int,
) -> None:
    """Fija las lineas base de potencia en reposo usadas para aislar la potencia activa.

    Args:
        devices: Dispositivos incluidos en el barrido.
        measure_sec: Segundos de muestreo en reposo (0 desactiva la medicion).
        cpu_override: Potencia idle de CPU fijada por CLI, o None para medirla.
        gpu_override: Potencia idle de GPU fijada por CLI, o None para medirla.
        gpu_index: Indice del dispositivo NVML.
    """
    global IDLE_POWER_CPU, IDLE_POWER_GPU

    if cpu_override is not None:
        IDLE_POWER_CPU = cpu_override
    elif "cpu" in devices and measure_sec > 0.0:
        print(f"Midiendo potencia idle de CPU durante {measure_sec:.1f}s...")
        IDLE_POWER_CPU = measure_idle_power_cpu(find_rapl_energy_paths(), measure_sec)

    if gpu_override is not None:
        IDLE_POWER_GPU = gpu_override
    elif "gpu" in devices and measure_sec > 0.0:
        print(f"Midiendo potencia idle de GPU durante {measure_sec:.1f}s...")
        IDLE_POWER_GPU = measure_idle_power_gpu(gpu_index, measure_sec)

    print(
        f"Linea base idle -> CPU: {IDLE_POWER_CPU:.3f} W | GPU: {IDLE_POWER_GPU:.3f} W"
    )


def run_monitored_execution(
    cmd_pwr: Sequence[str],
    device: str,
    gpu_index: int,
    timeout: float,
    sub_env: Dict[str, str],
    time_sec: float,
    context: str,
) -> Dict[str, float]:
    """Ejecuta el binario con telemetria activa y acota las metricas al lazo medido.

    Corresponde a la tercera fase del protocolo de aislamiento (warm-up -> medicion sin
    energia -> medicion con energia). El muestreo cubre todo el proceso, pero la energia
    se integra unicamente entre las marcas LOOP_WINDOW que publica el binario, de modo
    que el setup (plan, reservas, generacion de datos) y el cierre no contaminan la
    potencia ni la energia atribuidas al kernel.

    Args:
        cmd_pwr: Comando completo del binario con el numero de iteraciones de la fase
            de potencia ya inyectado.
        device: "cpu" o "gpu".
        gpu_index: Indice del dispositivo NVML.
        timeout: Timeout en segundos para el subproceso.
        sub_env: Entorno del subproceso (incluye BENCH_SEED).
        time_sec: Tiempo por iteracion medido en la fase de aislamiento.
        context: Descripcion del caso para los mensajes de error.

    Returns:
        Diccionario con Avg_Power_W, Energy_J, EDP y los campos de auditoria de la
        ventana de medicion.

    Raises:
        RuntimeError: Si el binario termina con codigo distinto de cero.
        MeasurementError: Si no hay telemetria, si esta no cubre la ventana, si es
            demasiado escasa, si el contador dio una vuelta irreconstruible dentro de la
            ventana, o si la potencia medida cae por debajo de la linea base de reposo.
    """
    sample_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    monitor_thread: Optional[threading.Thread] = None
    rapl_paths: List[str] = []

    if device == "gpu":
        idle_power_w = IDLE_POWER_GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu, args=(handle, stop_event, sample_queue), daemon=True
        )
        monitor_thread.start()
    else:
        idle_power_w = IDLE_POWER_CPU
        rapl_paths = find_rapl_energy_paths()
        if rapl_paths:
            monitor_thread = threading.Thread(
                target=monitor_energy_cpu,
                args=(rapl_paths, stop_event, sample_queue),
                daemon=True,
            )
            monitor_thread.start()
        else:
            warn_rapl_missing_once()

    start_wall = time.perf_counter()
    try:
        proc_pwr = subprocess.run(
            list(cmd_pwr),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=sub_env,
        )
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join()
    end_wall = time.perf_counter()

    if proc_pwr.returncode != 0:
        raise RuntimeError(
            f"Fallo en la ejecucion de monitoreo para {context}.\n"
            f"STDOUT:\n{proc_pwr.stdout}\nSTDERR:\n{proc_pwr.stderr}"
        )

    loop_start, loop_end, loop_iters = parse_loop_window(proc_pwr.stdout, context)
    window_sec = loop_end - loop_start

    payload = sample_queue.get() if not sample_queue.empty() else None
    if device == "gpu":
        samples = payload or []
        lost_wraps: List[float] = []
    else:
        samples, lost_wraps = payload if payload else ([], [])

    # Una vuelta del contador que no se pudo reconstruir solo corrompe la medicion si
    # ocurrio DENTRO de la ventana: la energia del lazo es una diferencia de la serie
    # acumulada, asi que un salto anterior o posterior se cancela.
    wraps_in_window = [t for t in lost_wraps if loop_start <= t <= loop_end]
    if wraps_in_window:
        raise MeasurementError(
            f"El contador RAPL dio la vuelta durante la ventana de {context} y no se pudo "
            "reconstruir la energia perdida (falta max_energy_range_uj)."
        )

    metrics: Dict[str, float] = {
        "Loop_Window_sec": window_sec,
        "Iters_K": loop_iters,
        "Idle_Power_W": idle_power_w,
        "Wall_Elapsed_sec": end_wall - start_wall,
    }

    if not samples:
        # Sin telemetria no hay medicion energetica. Escribir ceros seria peor que no
        # escribir nada: aguas abajo un EDP=0 es el valor optimo, asi que el agente
        # aprenderia a preferir precisamente las mediciones que fallaron.
        raise MeasurementError(
            f"No se obtuvo telemetria para {context} "
            f"({'NVML no devolvio muestras' if device == 'gpu' else 'RAPL no legible'})."
        )

    if samples[0][0] > loop_start or samples[-1][0] < loop_end:
        raise MeasurementError(
            f"Las muestras no cubren la ventana del lazo para {context}: "
            f"muestreo [{samples[0][0]:.6f}, {samples[-1][0]:.6f}] vs "
            f"lazo [{loop_start:.6f}, {loop_end:.6f}]."
        )

    if device == "gpu":
        energy_window_j, samples_inside = energy_from_power_samples(samples, loop_start, loop_end)
    else:
        energy_window_j, samples_inside = energy_from_counter_samples(samples, loop_start, loop_end)

    # Confiabilidad: el protocolo exige medir el tiempo sin telemetria y la potencia en una
    # segunda corrida, asi que ambos tiempos por iteracion deben coincidir. Una divergencia
    # grande indica que el dispositivo no estaba en el mismo estado en las dos fases.
    loop_time_per_iter = window_sec / loop_iters if loop_iters > 0 else 0.0
    if loop_time_per_iter > 0.0:
        divergence = abs(loop_time_per_iter - time_sec) / time_sec
        if divergence > LOOP_TIME_DIVERGENCE_WARN:
            print(
                f"[!] Aviso {context}: el tiempo por iteracion difiere {divergence * 100:.1f}% "
                f"entre la fase de aislamiento ({time_sec:.9f}s) y la de potencia "
                f"({loop_time_per_iter:.9f}s).",
                file=sys.stderr,
            )

    if samples_inside < MIN_SAMPLES_IN_WINDOW:
        raise MeasurementError(
            f"Telemetria insuficiente para {context}: {samples_inside} muestras dentro de "
            f"una ventana de {window_sec * 1e3:.1f} ms. Aumenta --power-window-sec."
        )

    # Aislamiento de la potencia activa: descontamos el consumo en reposo del mismo
    # intervalo para que Avg_Power_W refleje solo el costo atribuible al computo.
    window_power_w = energy_window_j / window_sec
    if idle_power_w > 0.0 and window_power_w < idle_power_w * IDLE_BASELINE_TOLERANCE:
        # Un lazo activo no puede consumir menos que el reposo: o la linea base se midio
        # con la maquina ocupada, o la telemetria no corresponde a esta ejecucion. Sin
        # este corte la resta se saturaria en cero y la fila entraria al CSV como si el
        # kernel fuese gratis.
        raise MeasurementError(
            f"Potencia bajo el reposo en {context}: {window_power_w:.2f} W medidos frente "
            f"a una linea base de {idle_power_w:.2f} W. Revisa la medicion de idle "
            "(--idle-power-cpu / --idle-power-gpu) o el estado del nodo."
        )

    energy_active_j = max(0.0, energy_window_j - idle_power_w * window_sec)
    avg_power_w = energy_active_j / window_sec

    # Energia de UNA ejecucion, coherente por construccion con el Time_sec reportado:
    # Energy_J = Avg_Power_W * Time_sec y EDP = Energy_J * Time_sec.
    energy_j = avg_power_w * time_sec

    metrics.update(
        {
            "Avg_Power_W": avg_power_w,
            "Energy_J": energy_j,
            "EDP": energy_j * time_sec,
            "Power_Samples": samples_inside,
        }
    )
    return metrics


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
    seed=None, use_databank=False, bank_profile="dense_normal",
    databank_dir=None, databank_max_n=None,
):
    """Retorna la ruta a un archivo GEMM kernel-ready (DataBankManager) o None para generación in-memory determinista."""
    if use_databank:
        try:
            DataBankManager = _get_data_bank_manager_cls()
            db_dir = databank_dir or DEFAULT_DATABANK_DIR
            db = DataBankManager(base_dir=db_dir, seed=seed or 42, max_n=databank_max_n)
            matrix_file = db.get_gemm_path(m, n, k, precision, profile=bank_profile)
            return matrix_file, True  # Es persistente del DataBankManager
        except Exception as ex:
            print(f"[!] Error en DataBankManager para GEMM ({m}x{n}x{k}, {precision}): {ex}", file=sys.stderr)

    return None, False


def generate_fft_matrix_file(
    nx, ny, nz, batch, precision, domain, layout,
    seed=None, use_databank=False, bank_profile="broadband",
    databank_dir=None, databank_max_n=None,
):
    """Retorna la ruta a un archivo FFT kernel-ready (DataBankManager) o None para generación in-memory determinista."""
    if use_databank:
        try:
            DataBankManager = _get_data_bank_manager_cls()
            db_dir = databank_dir or DEFAULT_DATABANK_DIR
            db = DataBankManager(base_dir=db_dir, seed=seed or 42, max_n=databank_max_n)
            matrix_file = db.get_fft_path(nx, ny, nz, batch, precision, domain, profile=bank_profile)
            return matrix_file, True  # Es persistente del DataBankManager
        except Exception as ex:
            print(f"[!] Error en DataBankManager para FFT ({nx}x{ny}x{nz}, {precision}, {domain}): {ex}", file=sys.stderr)

    return None, False


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
    use_databank=False,
    bank_profile="dense_normal",
    databank_dir=None,
    databank_max_n=None,
):
    matrix_file, _gemm_file_is_persistent = generate_gemm_matrix_file(
        m,
        n,
        k,
        precision,
        seed=seed,
        use_databank=use_databank,
        bank_profile=bank_profile,
        databank_dir=databank_dir,
        databank_max_n=databank_max_n,
    )

    try:
        sub_env = os.environ.copy()
        if seed is not None:
            sub_env["BENCH_SEED"] = str(seed)

        # Build binary execution command using CLI flags (allows specifying --warmup 0 --iters 1)
        cmd = [
            binary,
            "--m", str(m),
            "--n", str(n),
            "--k", str(k),
            "--precision", precision,
            "--op-a", op_a,
            "--op-b", op_b,
            "--warmup", "0",
            "--iters", "0" if not is_warmup else "1"
        ]
        if matrix_file:
            cmd.extend(["--source", matrix_file])

        if is_warmup:
            # 1. Warmup Run: Execute the binary once with 0 warmups, 1 iter, and no telemetry
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=sub_env,
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
            env=sub_env,
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
        if time_sec <= 0.0:
            time_sec = 1e-9

        # El lazo se dimensiona para durar POWER_WINDOW_TARGET_SEC: una ventana corta
        # deja demasiado pocas muestras de NVML/RAPL para integrar la energia con rigor.
        power_iters = min(20000, max(1, round(POWER_WINDOW_TARGET_SEC / time_sec)))
        cmd_pwr = list(cmd)
        try:
            iters_idx = cmd_pwr.index("--iters")
            cmd_pwr[iters_idx + 1] = str(power_iters)
        except ValueError:
            cmd_pwr.extend(["--iters", str(power_iters)])

        context = (
            f"GEMM M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b} [{device}]"
        )
        telemetry = run_monitored_execution(
            cmd_pwr, device, gpu_index, timeout, sub_env, time_sec, context
        )

        if precision in {"C", "Z"}:
            ops = 8.0 * m * n * k
        else:
            ops = 2.0 * m * n * k

        result = {
            "M": m,
            "N": n,
            "K": k,
            "Precision": precision,
            "OpA": op_a,
            "OpB": op_b,
            "Time_sec": time_sec,
            "GFLOPS": (ops / time_sec) / 1e9,
        }
        result.update(telemetry)
        return result
    finally:
        # Solo eliminar el archivo si es temporal (no proviene del DataBankManager).
        if matrix_file and not _gemm_file_is_persistent and os.path.exists(matrix_file):
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
    seed=None,
):
    sub_env = os.environ.copy()
    if seed is not None:
        sub_env["BENCH_SEED"] = str(seed)

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
            env=sub_env,
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
        env=sub_env,
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
    # El lazo se dimensiona para durar POWER_WINDOW_TARGET_SEC: una ventana corta deja
    # demasiado pocas muestras de NVML/RAPL para integrar la energia con rigor.
    power_iters = min(20000, max(1, round(POWER_WINDOW_TARGET_SEC / time_sec)))
    cmd_pwr = list(cmd)
    if len(cmd_pwr) > 10:
        cmd_pwr[10] = str(power_iters)
    else:
        raise ValueError(f"Comando FFT mal formado para agregar iteraciones: {cmd_pwr}")

    context = (
        f"FFT Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, "
        f"Dir={direction}, L={layout} [{device}]"
    )
    telemetry = run_monitored_execution(
        cmd_pwr, device, gpu_index, timeout, sub_env, time_sec, context
    )

    dims = fft_dims(nx, ny, nz)
    ops = fft_flops(dims, domain) * batch

    result = {
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
        "GFLOPS": (ops / time_sec) / 1e9,
    }
    result.update(telemetry)
    return result


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

    configure_idle_baselines(
        devices,
        args.idle_measure_sec,
        args.idle_power_cpu,
        args.idle_power_gpu,
        args.gpu_index,
    )

    try:
        fieldnames = [
            "Device",
            "M",
            "N",
            "K",
            "Precision",
            "OpA",
            "OpB",
        ]
        if args.repetitions > 1:
            fieldnames.append("Iteration")
        fieldnames.extend([
            "Time_sec",
            "GFLOPS",
            "Avg_Power_W",
            "Energy_J",
            "EDP",
            # Metadata de la ventana de medicion: permite auditar a posteriori que la
            # telemetria se integro sobre el lazo y no sobre el proceso completo.
            "Loop_Window_sec",
            "Iters_K",
            "Power_Samples",
            "Idle_Power_W",
            "Wall_Elapsed_sec",
        ])

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
                            
                            # Confiabilidad del barrido: una medicion fallida (telemetria
                            # incompleta, timeout o binario sin marcas) se omite sin abortar
                            # el resto del sweep.
                            try:
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
                                        use_databank=args.use_databank,
                                        bank_profile=args.gemm_profile,
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
                                            use_databank=args.use_databank,
                                            bank_profile=args.gemm_profile,
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
                                        use_databank=args.use_databank,
                                        bank_profile=args.gemm_profile,
                                        databank_dir=args.databank_dir,
                                        databank_max_n=args.databank_max_n,
                                    )

                                # Include Device and Iteration in the written row
                                row = {key: result.get(key, 0.0) for key in fieldnames if key not in ["Device", "Iteration"]}
                                row["Device"] = device
                                if args.repetitions > 1:
                                    row["Iteration"] = rep
                                writer.writerow(row)
                                f.flush()

                                print(
                                    f"[{done}/{total}] {device.upper()} M={m} N={n} K={k} P={precision} OpA={op_a} OpB={op_b} "
                                    f"Rep={rep} Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                                    f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                                    f"EDP={result['EDP']:.9f}"
                                )
                            except (MeasurementError, RuntimeError, ValueError, OSError,
                                    subprocess.SubprocessError) as ex:
                                print(
                                    f"[!] Medicion omitida [{done}/{total}] {device.upper()} rep={rep}: {ex}",
                                    file=sys.stderr,
                                )
                                continue

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
            raise ValueError("No se definieron tamaños FFT (1D/2D/3D)")

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

    configure_idle_baselines(
        devices,
        args.idle_measure_sec,
        args.idle_power_cpu,
        args.idle_power_gpu,
        args.gpu_index,
    )

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
        ]
        if args.repetitions > 1:
            fieldnames.append("Iteration")
        fieldnames.extend([
            "Time_sec",
            "GFLOPS",
            "Avg_Power_W",
            "Energy_J",
            "EDP",
            # Metadata de la ventana de medicion: permite auditar a posteriori que la
            # telemetria se integro sobre el lazo y no sobre el proceso completo.
            "Loop_Window_sec",
            "Iters_K",
            "Power_Samples",
            "Idle_Power_W",
            "Wall_Elapsed_sec",
        ])

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
                        use_databank=args.use_databank,
                        bank_profile=args.fft_profile,
                        databank_dir=args.databank_dir,
                        databank_max_n=args.databank_max_n,
                    )
                    
                    try:
                        for rep in range(args.repetitions):
                            done += 1
                            
                            # Confiabilidad del barrido: una medicion fallida (telemetria
                            # incompleta, timeout o binario sin marcas) se omite sin abortar
                            # el resto del sweep.
                            try:
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
                                        seed=args.seed,
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
                                            seed=args.seed,
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
                                        seed=args.seed,
                                    )

                                row = {key: result.get(key, 0.0) for key in fieldnames if key not in ["Device", "Iteration"]}
                                row["Device"] = device
                                if args.repetitions > 1:
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
                            except (MeasurementError, RuntimeError, ValueError, OSError,
                                    subprocess.SubprocessError) as ex:
                                print(
                                    f"[!] Medicion omitida [{done}/{total}] {device.upper()} rep={rep}: {ex}",
                                    file=sys.stderr,
                                )
                                continue

                    finally:
                        # Solo eliminar si es temporal (no del DataBankManager).
                        if matrix_file and not _fft_file_is_persistent and os.path.exists(matrix_file):
                            os.unlink(matrix_file)

        print(f"\nResultados guardados en: {output_path}")
    finally:
        if "gpu" in devices:
            pynvml.nvmlShutdown()


def main():
    global POWER_WINDOW_TARGET_SEC
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
        "--use-databank",
        action="store_true",
        help="Usa archivos binarios del DataBankManager en lugar de generación determinista in-memory",
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
        default=None,
        help="Potencia de CPU en reposo (W). Si se omite, se mide al inicio del barrido.",
    )
    parser.add_argument(
        "--idle-power-gpu",
        type=float,
        default=None,
        help="Potencia de GPU en reposo (W). Si se omite, se mide al inicio del barrido.",
    )
    parser.add_argument(
        "--idle-measure-sec",
        type=float,
        default=3.0,
        help="Segundos de muestreo para medir la potencia en reposo (0 desactiva la medicion)",
    )
    parser.add_argument(
        "--power-window-sec",
        type=float,
        default=POWER_WINDOW_TARGET_SEC,
        help=(
            "Duracion objetivo del lazo bajo monitoreo energetico. Ventanas mas largas "
            "dan mas muestras de RAPL/NVML por medicion, a costa de tiempo de barrido."
        ),
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

    if args.power_window_sec <= 0.0:
        raise ValueError("--power-window-sec debe ser positivo")
    POWER_WINDOW_TARGET_SEC = args.power_window_sec
    # Las lineas base de reposo se fijan en configure_idle_baselines(), ya con NVML activo.

    # Compatibilidad: --binary sobreescribe --gemm-binary-gpu
    if args.binary is not None:
        args.gemm_binary_gpu = args.binary

    if args.benchmark == "gemm":
        run_gemm(args)
    else:
        run_fft(args)


if __name__ == "__main__":
    main()
