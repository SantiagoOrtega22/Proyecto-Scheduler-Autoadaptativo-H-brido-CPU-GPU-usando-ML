#!/usr/bin/env python3
"""
BENCHMARK RUNNER - Orquestador para ejecutar barridos de parametros GEMM.

USO:
    # Barrido baseline (matrices cuadradas M=N=K, tamanos 128-4096, precisiones S/D/C/Z, operaciones N,N)
  python3 benchmark_runner.py
  
  # Barrido completo con transposiciones (N/T/C en ambas matrices)
  python3 benchmark_runner.py --sweep-transpose --op-a-list N,T --op-b-list N,T

    # Barrido completo de dimensiones (M, N y K independientes)
    python3 benchmark_runner.py --full-dim-sweep
  
  # Barrido parcial personalizado
  python3 benchmark_runner.py --sizes 256,512,1024 --precisions D,C --op-a-list N,T --op-b-list N
  
  # Especificar ruta del binario y GPU
  python3 benchmark_runner.py --binary ./GEMMparametros --gpu-index 0 --output mi_resultado.csv

OPCIONES:
  --binary            Ruta al binario compilado GEMMparametros (default: ./GEMMparametros)
  --sizes             Lista de tamanos separados por coma, ej: 128,256,512 (default: 128-4096 escala 2x)
  --precisions        Precisiones a probar: S/D/C/Z (default: S,D,C,Z)
    --full-dim-sweep    Activa el barrido completo MxNxK (default: solo M=N=K)
  --op-a-list         Operaciones en matriz A: N/T/C (default: N, ignorado si --sweep-transpose omitido)
  --op-b-list         Operaciones en matriz B: N/T/C (default: N, ignorado si --sweep-transpose omitido)
  --sweep-transpose   Activar barrido de transposiciones (default: desactivado)
  --gpu-index         Indice de GPU a usar (default: 0)
  --output            Archivo CSV de salida (default: benchmark_results.csv)
  --timeout           Segundos maximos por caso (default: 300)

SALIDA CSV:
  Columnas: M, N, K, Precision, OpA, OpB, Time_sec, GFLOPS, Avg_Power_W, Energy_J, EDP
  - Time_sec:  Tiempo de ejecucion (segundos)
  - GFLOPS:    Operaciones en punto flotante / segundo / 1e9
  - Avg_Power_W: Potencia GPU promedio durante GEMM (Watts)
  - Energy_J:  Energia consumida = Potencia × Tiempo (Joules)
  - EDP:       Energy-Delay Product = Energia × Tiempo

EJEMPLOS:
    # Barrido rapido: solo matrices 256x256x256, precision D
  python3 benchmark_runner.py --sizes 256 --precisions D
  
  # Barrido con transposiciones solo en OpA
  python3 benchmark_runner.py --sweep-transpose --op-a-list N,T --op-b-list N
  
    # Todo (completo): 6 tamanos × 4 precisiones × 9 ops = 216 casos cuadrados+ops
  python3 benchmark_runner.py --sweep-transpose --op-a-list N,T,C --op-b-list N,T,C
"""

import argparse
import csv
import itertools
import queue
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


def monitor_power(handle, stop_event, power_queue):
    # Hilo de monitoreo: muestrea potencia NVML en bucle hasta recibir la senal de parada.
    samples = []
    while not stop_event.is_set():
        timestamp = time.perf_counter()
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        samples.append((timestamp, power_mw / 1000.0))
    power_queue.put(samples)


def run_single_case(binary, gpu_index, m, n, k, precision, op_a, op_b, timeout):
    # Ejecuta un unico experimento (M,N,K,precision) y toma potencia en paralelo.
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    cmd = [binary, str(m), str(n), str(k), precision, op_a, op_b]

    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_power,
        args=(handle, stop_event, power_queue),
        daemon=True,
    )

    monitor_thread.start()
    start = time.perf_counter()

    try:
        # Lanza el binario CUDA y captura stdout para extraer Time_sec.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        stop_event.set()
        monitor_thread.join()

    end = time.perf_counter()
    samples = power_queue.get() if not power_queue.empty() else []

    if proc.returncode != 0:
        raise RuntimeError(
            "Fallo en binario CUDA para "
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

    # Potencia promedio durante la ejecucion medida.
    avg_power_w = sum(p for _, p in samples) / len(samples) if samples else 0.0

    # FLOPs teoricos por tipo: complejos ~8MNK, reales ~2MNK.
    if precision in {"C", "Z"}:
        ops = 8.0 * m * n * k
    else:
        ops = 2.0 * m * n * k

    # Metricas derivadas para rendimiento y eficiencia energetica.
    gflops = (ops / time_sec) / 1e9
    energy_j = avg_power_w * time_sec
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
        "Wall_Elapsed_sec": end - start,
    }


def main():
    # Configuracion de entrada para barrer tamanos y precisiones.
    parser = argparse.ArgumentParser(
        description="Orquestador GEMM cuBLAS con monitoreo de potencia NVML"
    )
    parser.add_argument("--binary", default="./GEMMparametros", help="Ruta al binario CUDA")
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

    pynvml.nvmlInit()
    try:
        # Verifica que la GPU solicitada exista en el sistema.
        device_count = pynvml.nvmlDeviceGetCount()
        if args.gpu_index < 0 or args.gpu_index >= device_count:
            raise RuntimeError(
                f"gpu-index invalido: {args.gpu_index}. GPUs disponibles: {device_count}"
            )

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
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
