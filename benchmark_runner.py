#!/usr/bin/env python3
"""
benchmark_runner.py

Orquestador de benchmarking GEMM para CPU o GPU.

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

OPCIONES PRINCIPALES
--------------------
    --device            Dispositivo donde correr el benchmark: gpu o cpu
    --binary            Ruta al binario a ejecutar
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

import pynvml

# Expresion regular para extraer el tiempo reportado por el binario CUDA.
TIME_PATTERN = re.compile(r"Time_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


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


def monitor_power_gpu(handle, stop_event, power_queue):
    # Hilo de monitoreo NVML: muestrea potencia en mW y la guarda como W.
    samples = []
    while not stop_event.is_set():
        timestamp = time.perf_counter()
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append((timestamp, power_mw / 1000.0))
        except Exception:
            # En caso de fallo NVML, seguir intentando hasta stop_event
            pass
    power_queue.put(samples)


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

    # Guardamos dos muestras (ts, energy_J) en lugar de potencia instantanea.
    samples = [(t0, e0 / 1e6), (t1, e1 / 1e6)]
    power_queue.put(samples)


def find_rapl_energy_path():
    # Busca energy_uj sin recorrer recursivamente todo powercap; así evitamos bloqueos.
    base_dir = "/sys/class/powercap"
    if not os.path.isdir(base_dir):
        return None

    def is_readable_energy(path):
        return os.path.isfile(path) and os.access(path, os.R_OK)

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
            print(
                "Aviso: no se encontro energy_uj RAPL en /sys/class/powercap; "
                "se registrara Avg_Power_W/Energy_J/EDP como 0.0 para CPU.",
                file=sys.stderr,
            )

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

    if device == "gpu":
        # Promedio de muestras NVML (W)
        avg_power_w = sum(p for _, p in samples) / len(samples) if samples else 0.0
        energy_j = avg_power_w * time_sec
    else:
        # samples = [(t0, e0_J), (t1, e1_J)]
        if len(samples) >= 2:
            e0 = samples[0][1]
            e1 = samples[1][1]
            energy_j = max(0.0, e1 - e0)
            avg_power_w = energy_j / time_sec if time_sec > 0 else 0.0
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


def main():
    # Configuracion de entrada para barrer tamanos y precisiones.
    parser = argparse.ArgumentParser(
        description="Orquestador GEMM cuBLAS con monitoreo de potencia NVML"
    )
    parser.add_argument("--binary", default="./GEMMparametros", help="Ruta al binario CUDA")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Dispositivo donde ejecutar el benchmark (gpu|cpu)")
    parser.add_argument(
        "--sizes",
        default="128,256,512,1024,2048,4096",
        help="Lista separada por comas para M,N,K",
    )
    parser.add_argument(
        "--precisions",
        default="S,D,C,Z",
        help="Lista separada por comas de precisiones: S,D,C,Z",
    )
    parser.add_argument(
        "--full-dim-sweep",
        action="store_true",
        help="Activa el barrido completo de M, N y K; por defecto solo se prueban matrices cuadradas",
    )
    parser.add_argument(
        "--sweep-transpose",
        action="store_true",
        help="Activa el barrido de transposicion para opA/opB",
    )
    parser.add_argument(
        "--op-a-list",
        default="N",
        help="Lista separada por comas para opA: N,T,C (solo con --sweep-transpose)",
    )
    parser.add_argument(
        "--op-b-list",
        default="N",
        help="Lista separada por comas para opB: N,T,C (solo con --sweep-transpose)",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="Indice de GPU para NVML")
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Archivo CSV de salida",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout por ejecucion en segundos",
    )

    args = parser.parse_args()

    # Validacion temprana de parametros antes de iniciar NVML y ejecuciones.
    sizes = parse_sizes(args.sizes)
    precisions = parse_precisions(args.precisions)
    if args.sweep_transpose:
        op_a_list = parse_ops(args.op_a_list)
        op_b_list = parse_ops(args.op_b_list)
    else:
        # Caso base: GEMM sin transposicion en A y B.
        op_a_list = ["N"]
        op_b_list = ["N"]

    # Inicializaciones por dispositivo
    if args.device == "gpu":
        pynvml.nvmlInit()
        try:
            # Verifica que la GPU solicitada exista en el sistema.
            device_count = pynvml.nvmlDeviceGetCount()
            if args.gpu_index < 0 or args.gpu_index >= device_count:
                raise RuntimeError(
                    f"gpu-index invalido: {args.gpu_index}. GPUs disponibles: {device_count}"
                )
        except Exception:
            pynvml.nvmlShutdown()
            raise

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

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Recorre los tamanos solicitados; por defecto se fijan como matrices cuadradas.
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

                    # Guarda una fila por experimento en el CSV final.
                    writer.writerow({key: result[key] for key in fieldnames})
                    f.flush()

                    print(
                        f"[{done}/{total}] M={m} N={n} K={k} P={precision} OpA={op_a} OpB={op_b} "
                        f"Time={result['Time_sec']:.6f}s GFLOPS={result['GFLOPS']:.3f} "
                        f"Pavg={result['Avg_Power_W']:.3f}W Energy={result['Energy_J']:.6f}J "
                        f"EDP={result['EDP']:.9f}"
                    )

        print(f"\nResultados guardados en: {args.output}")
    finally:
        if args.device == "gpu":
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
