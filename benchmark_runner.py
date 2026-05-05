#!/usr/bin/env python3
"""
benchmark_runner.py

Orquestador de benchmarking GEMM/FFT para CPU o GPU.

GUIA DE USO
-----------
Modo GPU (usa el binario CUDA `GEMMparametros`):
    python3 benchmark_runner.py --device gpu --binary ./GEMMparametros

Modo CPU (usa `gemm_cpu_mkl` u otro binario BLAS compatible):
    python3 benchmark_runner.py --device cpu --binary ./gemm_cpu_mkl

Barrido rapido de prueba:
    python3 benchmark_runner.py --device cpu --binary ./gemm_cpu_mkl --sizes 128 --precisions S --output cpu_results.csv

Barrido cuadrado por defecto (M=N=K):
    python3 benchmark_runner.py --device gpu

Barrido completo de dimensiones (M, N y K independientes):
    python3 benchmark_runner.py --full-dim-sweep

Barrido con transposiciones en A y B:
    python3 benchmark_runner.py --sweep-transpose --op-a-list N,T,C --op-b-list N,T,C

Modo FFT (CPU+GPU en una sola ejecucion):
    python3 benchmark_runner.py --benchmark fft --device both

FFT con barrido personalizado:
    python3 benchmark_runner.py --benchmark fft --device gpu \
        --fft-sizes-1d 1024,2048 --fft-sizes-2d 64x64 --fft-batches 1,4

OPCIONES PRINCIPALES
--------------------
    --benchmark         Benchmark a ejecutar: gemm o fft
    --device            Dispositivo donde correr el benchmark: gpu, cpu o both (solo FFT)
    --binary            Ruta al binario GEMM a ejecutar
    --sizes             Lista separada por coma para los tamanos base
    --precisions        Precisiones a probar: S, D, C, Z
    --full-dim-sweep    Activa combinacion completa de M x N x K
    --sweep-transpose   Activa barrido de OpA / OpB
    --op-a-list         Operaciones posibles para la matriz A: N, T, C
    --op-b-list         Operaciones posibles para la matriz B: N, T, C
    --gpu-index         Indice de GPU para NVML cuando device=gpu
    --output            Archivo CSV de salida
    --timeout           Timeout por caso en segundos

SALIDA CSV
----------
Columnas generadas:
    M, N, K, Precision, OpA, OpB, Time_sec, GFLOPS, Avg_Power_W, Energy_J, EDP

Interpretacion:
    Time_sec     -> tiempo medido por el binario
    GFLOPS       -> rendimiento calculado a partir de M, N, K y la precision
    Avg_Power_W  -> potencia media estimada durante el caso
    Energy_J     -> energia consumida durante el caso
    EDP          -> Energy-Delay Product

NOTAS
-----
    - En GPU, la potencia se lee con NVML.
    - En CPU, se intenta leer RAPL desde /sys/class/powercap.
    - Si RAPL no esta disponible, el runner sigue y deja potencia/energia en 0.0.
    - La salida del binario debe incluir una linea con `Time_sec=...`.
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

import pynvml

# Expresion regular para extraer el tiempo reportado por el binario CUDA.
TIME_PATTERN = re.compile(r"Time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
FFT_TIME_PATTERN = re.compile(
    r"Time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|tiempo=([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)

_RAPL_WARNING_SHOWN = False
POWER_SAMPLE_INTERVAL_SEC = 0.02


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
        # Sin límites disponibles, seguimos con el comportamiento estándar de NVML: mW->W.
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


def find_rapl_energy_path():
    # Busca energy_uj sin recorrer recursivamente todo powercap; así evitamos bloqueos.
    base_dir = "/sys/class/powercap"
    if not os.path.isdir(base_dir):
        return None

    def is_readable_energy(path):
        return os.path.isfile(path) and os.access(path, os.R_OK)

    # Rutas comunes conocidas (intel-rapl:0 es el primer socket en la mayoría de sistemas)
    common_paths = [
        os.path.join(base_dir, "intel-rapl:0", "energy_uj"),
        os.path.join(base_dir, "intel-rapl:0:0", "energy_uj"),
        os.path.join(base_dir, "intel-rapl", "energy_uj"),
    ]
    
    for path in common_paths:
        if is_readable_energy(path):
            return path

    # Si no encontró en rutas conocidas, busca recursivamente limitado a nivel 2
    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("intel-rapl"):
                    continue

                direct_energy = os.path.join(base_dir, entry.name, "energy_uj")
                if is_readable_energy(direct_energy):
                    return direct_energy

                try:
                    with os.scandir(os.path.join(base_dir, entry.name)) as child_entries:
                        for child in child_entries:
                            if not child.is_dir(follow_symlinks=False):
                                continue
                            nested_energy = os.path.join(base_dir, entry.name, child.name, "energy_uj")
                            if is_readable_energy(nested_energy):
                                return nested_energy
                except OSError:
                    continue
    except OSError:
        return None

    return None


def run_single_case(binary, device, gpu_index, m, n, k, precision, op_a, op_b, timeout):
    # Ejecuta un unico experimento (M,N,K,precision) y toma potencia en paralelo.
    cmd = [binary, str(m), str(n), str(k), precision, op_a, op_b]

    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    monitor_thread = None
    start_wall = time.perf_counter()

    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu,
            args=(handle, stop_event, power_queue),
            daemon=True,
        )
    else:
        rapl = find_rapl_energy_path()
        if rapl:
            monitor_thread = threading.Thread(
                target=monitor_power_cpu,
                args=(rapl, stop_event, power_queue),
                daemon=True,
            )
        else:
            warn_rapl_missing_once()

    if monitor_thread is not None:
        monitor_thread.start()

    try:
        proc = subprocess.run(
            cmd,
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
    samples = power_queue.get() if not power_queue.empty() else []

    if proc.returncode != 0:
        raise RuntimeError(
            "Fallo en binario para "
            f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    match = TIME_PATTERN.search(proc.stdout)
    if not match:
        raise RuntimeError(
            "No se pudo parsear Time_sec de la salida del binario.\n"
            f"Salida:\n{proc.stdout}"
        )

    time_sec = float(match.group(1))
    if time_sec <= 0.0:
        raise RuntimeError(
            f"Time_sec invalido ({time_sec}) para M={m},N={n},K={k},P={precision},OpA={op_a},OpB={op_b}"
        )

    # Calcula potencia y energia dependiendo del dispositivo
    avg_power_w = 0.0
    energy_j = 0.0
    wall_time = end_wall - start_wall

    if device == "gpu":
        # Promedio temporal de la potencia NVML (W).
        # Para energía, usamos el tiempo reportado por el binario (kernel) para ser consistente con FFT.
        avg_power_w = average_power_from_samples(samples)
        energy_j = avg_power_w * time_sec
    else:
        # samples = [(t0, e0_J), (t1, e1_J)]
        # IMPORTANTE: Para CPU, usamos wall_time (tiempo total) en lugar de time_sec (tiempo del benchmark),
        # porque RAPL mide energía durante toda la ejecución del proceso, no solo el kernel.
        if len(samples) >= 2:
            e0 = samples[0][1]
            e1 = samples[1][1]
            energy_j = max(0.0, e1 - e0)
            avg_power_w = energy_j / wall_time if wall_time > 0 else 0.0
        else:
            avg_power_w = 0.0
            energy_j = 0.0

    # FLOPs teoricos por tipo: complejos ~8MNK, reales ~2MNK.
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
        "Power_Samples": len(samples),
        "Wall_Elapsed_sec": end_wall - start_wall,
    }


def run_single_case_fft(binary, device, gpu_index, nx, ny, nz, batch, precision, domain, direction, layout, timeout):
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
    ]

    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    monitor_thread = None
    start_wall = time.perf_counter()

    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu,
            args=(handle, stop_event, power_queue),
            daemon=True,
        )
    else:
        rapl = find_rapl_energy_path()
        if rapl:
            monitor_thread = threading.Thread(
                target=monitor_power_cpu,
                args=(rapl, stop_event, power_queue),
                daemon=True,
            )
        else:
            warn_rapl_missing_once()

    if monitor_thread is not None:
        monitor_thread.start()

    try:
        proc = subprocess.run(
            cmd,
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
    samples = power_queue.get() if not power_queue.empty() else []

    if proc.returncode != 0:
        raise RuntimeError(
            "Fallo en binario FFT para "
            f"Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, Dir={direction}, L={layout}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    match = FFT_TIME_PATTERN.search(proc.stdout)
    if not match:
        raise RuntimeError(
            "No se pudo parsear tiempo de la salida FFT.\n"
            f"Salida:\n{proc.stdout}"
        )

    if match.group(1) is not None:
        time_sec = float(match.group(1))
    else:
        time_sec = float(match.group(2)) / 1e3
    if time_sec <= 0.0:
        raise RuntimeError(
            f"Time_sec invalido ({time_sec}) para Nx={nx},Ny={ny},Nz={nz},Batch={batch}"
        )

    avg_power_w = 0.0
    energy_j = 0.0
    wall_time = end_wall - start_wall

    if device == "gpu":
        avg_power_w = average_power_from_samples(samples)
        energy_j = avg_power_w * time_sec
    else:
        # IMPORTANTE: Para CPU, usamos wall_time (tiempo total) en lugar de time_sec (tiempo del FFT),
        # porque RAPL mide energía durante toda la ejecución del proceso, no solo el kernel.
        if len(samples) >= 2:
            e0 = samples[0][1]
            e1 = samples[1][1]
            energy_j = max(0.0, e1 - e0)
            avg_power_w = energy_j / wall_time if wall_time > 0 else 0.0

    dims = fft_dims(nx, ny, nz)
    ops = fft_flops(dims, domain)
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
        "Samples_Power": len(samples),
        "Wall_Elapsed_sec": end_wall - start_wall,
    }


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
    if args.device not in {"gpu", "cpu"}:
        raise ValueError("Para GEMM, --device debe ser cpu o gpu")

    sizes = parse_sizes(args.sizes)
    precisions = parse_precisions(args.precisions)
    if args.sweep_transpose:
        op_a_list = parse_ops(args.op_a_list)
        op_b_list = parse_ops(args.op_b_list)
    else:
        op_a_list = ["N"]
        op_b_list = ["N"]

    output_path = args.output or "benchmark_results.csv"
    device_list = [args.device]

    if args.device == "gpu":
        init_nvml_if_needed(device_list, args.gpu_index)

    try:
        fieldnames = [
            "M",
            "N",
            "K",
            "Precision",
            "OpA",
            "OpB",
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

        total = len(dim_cases) * len(precisions) * len(op_a_list) * len(op_b_list)
        done = 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m, n, k in dim_cases:
                for precision, op_a, op_b in itertools.product(precisions, op_a_list, op_b_list):
                    done += 1
                    result = run_single_case(
                        args.binary,
                        args.device,
                        args.gpu_index,
                        m,
                        n,
                        k,
                        precision,
                        op_a,
                        op_b,
                        args.timeout,
                    )

                    writer.writerow({key: result[key] for key in fieldnames})
                    f.flush()

                    print(
                        f"[{done}/{total}] M={m} N={n} K={k} P={precision} OpA={op_a} OpB={op_b} "
                        f"Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                        f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                        f"EDP={result['EDP']:.9f}"
                    )

        print(f"\nResultados guardados en: {output_path}")
    finally:
        if args.device == "gpu":
            pynvml.nvmlShutdown()


def run_fft(args):
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

        total = len(cases) * len(devices)
        done = 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for nx, ny, nz, batch, precision, domain, direction, layout in cases:
                for device in devices:
                    done += 1
                    binary = args.fft_binary_gpu if device == "gpu" else args.fft_binary_cpu
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
                        args.timeout,
                    )

                    writer.writerow({key: result[key] for key in fieldnames})
                    f.flush()

                    print(
                        f"[{done}/{total}] {device.upper()} Nx={nx} Ny={ny} Nz={nz} Batch={batch} "
                        f"P={precision} D={domain} Dir={direction} L={layout} "
                        f"Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                        f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                        f"EDP={result['EDP']:.9f}"
                    )

                    if args.cooldown > 0 and done < total:
                        time.sleep(args.cooldown)

        print(f"\nResultados guardados en: {output_path}")
    finally:
        if "gpu" in devices:
            pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador de benchmarking GEMM/FFT con monitoreo de potencia"
    )
    parser.add_argument(
        "--benchmark",
        choices=["gemm", "fft"],
        default="gemm",
        help="Selecciona el benchmark a ejecutar (gemm|fft)",
    )
    parser.add_argument("--binary", default="./GEMMparametros", help="Ruta al binario GEMM CUDA")
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
        default="N",
        help="Lista separada por comas para opA: N,T,C (GEMM)",
    )
    parser.add_argument(
        "--op-b-list",
        default="N",
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
        "--fft-binary-cpu",
        default="./Asignador/fft_cpu",
        help="Ruta al binario FFT CPU",
    )
    parser.add_argument(
        "--fft-binary-gpu",
        default="./Asignador/fft_gpu",
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
        "--cooldown",
        type=float,
        default=1.0,
        help="Pausa en segundos entre ejecuciones secuenciales (FFT)",
    )

    args = parser.parse_args()

    if args.benchmark == "gemm":
        run_gemm(args)
    else:
        run_fft(args)


if __name__ == "__main__":
    main()
