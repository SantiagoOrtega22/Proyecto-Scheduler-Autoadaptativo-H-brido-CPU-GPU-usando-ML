#!/usr/bin/env python3
"""
inspect_databank.py
===================
Herramienta CLI e interactiva para leer, inspeccionar y visualizar archivos
binarios del databank del benchmark (GEMM y FFT).

Cumple con las especificaciones del SDD (Section 7 - Coding Standards and Documentation):
- Type hints explicito en todas las funciones.
- Docstrings con formato Google-style.
- Context managers para la manipulación segura de archivos.
- Visualización de estadísticas y matrices/vectores.
"""

import sys
import struct
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np


# ── Mapeos de Tipos de Datos y Precisión ──────────────────────────────────────
_PREC_DTYPE: Dict[str, np.dtype] = {
    "S": np.dtype("<f4"),
    "D": np.dtype("<f8"),
    "C": np.dtype("<c8"),  # complex64 (2 x float32 interleaved)
    "Z": np.dtype("<c16"), # complex128 (2 x float64 interleaved)
}

_PREC_BYTES: Dict[str, int] = {
    "S": 4,
    "D": 8,
    "C": 8,
    "Z": 16,
}


def read_databank_file(file_path: str) -> Dict[str, Any]:
    """Lee y parsea la cabecera y el contenido de un archivo binario del databank.

    Detecta automáticamente si el archivo corresponde a un benchmark de GEMM o FFT
    basándose en la estructura de su header.

    Args:
        file_path (str): Ruta al archivo binario (.bin).

    Returns:
        Dict[str, Any]: Diccionario con el tipo de algoritmo, metadatos y arrays NumPy
            correspondientes a las entradas y salidas del archivo binario.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta provista.
        ValueError: Si la estructura o el encabezado del archivo no es válido.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

    with open(path, "rb") as f:
        header_bytes = f.read(13)
        if len(header_bytes) < 13:
            raise ValueError(f"El archivo '{file_path}' es demasiado pequeño para contener una cabecera válida.")

        # Inspeccionamos para determinar si es GEMM o FFT
        # GEMM header: M(int32), N(int32), K(int32), Prec(char) -> 4+4+4+1 = 13 bytes
        # FFT header:  Nx(int32), Ny(int32), Nz(int32), Batch(int32), Prec(char), Domain(3 chars) -> 4*4 + 1 + 3 = 20 bytes

        # Intentamos verificar si cumple la estructura FFT leyendo 20 bytes
        f.seek(0)
        full_header_20 = f.read(20)

        # Tratar de decodificar como FFT
        is_fft = False
        if len(full_header_20) == 20:
            try:
                nx, ny, nz, batch = struct.unpack("<iiii", full_header_20[:16])
                prec_fft = full_header_20[16:17].decode("ascii").upper()
                dom_fft = full_header_20[17:20].decode("ascii").upper()
                if prec_fft in _PREC_DTYPE and dom_fft in ("C2C", "R2C", "C2R"):
                    is_fft = True
            except Exception:
                is_fft = False

        if is_fft:
            f.seek(20)  # Tras la cabecera de FFT
            dtype = _PREC_DTYPE[prec_fft]

            # Calcular tamaño de entrada/salida
            n_total = nx * (ny if ny > 0 else 1) * (nz if nz > 0 else 1)
            n_total_batched = n_total * batch

            if dom_fft in ("C2C", "C2R"):
                in_dtype = dtype
            else:  # R2C -> Entrada Real (float32 para S, float64 para D)
                in_dtype = np.dtype("<f4") if prec_fft == "S" else np.dtype("<f8")

            if dom_fft == "C2C":
                out_dtype = dtype
                out_size = n_total_batched
            elif dom_fft == "R2C":
                out_dtype = np.dtype("<c8") if prec_fft == "S" else np.dtype("<c16")
                last = nz if nz > 0 else (ny if ny > 0 else nx)
                outer = n_total // last
                out_size = outer * (last // 2 + 1) * batch
            else:  # C2R -> Salida Real
                out_dtype = np.dtype("<f4") if prec_fft == "S" else np.dtype("<f8")
                out_size = n_total_batched

            in_data = np.fromfile(f, dtype=in_dtype, count=n_total_batched)
            out_data = np.fromfile(f, dtype=out_dtype, count=out_size)

            return {
                "algo": "FFT",
                "file_path": str(path),
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "batch": batch,
                "precision": prec_fft,
                "domain": dom_fft,
                "input": in_data,
                "output": out_data,
            }

        # De lo contrario, intentamos parsear como GEMM (13 bytes header)
        f.seek(0)
        gemm_header = f.read(13)
        m, n, k = struct.unpack("<iii", gemm_header[:12])
        prec_gemm = gemm_header[12:13].decode("ascii").upper()

        if prec_gemm not in _PREC_DTYPE:
            raise ValueError(f"Precisión no reconocida '{prec_gemm}' en el archivo binario.")

        dtype = _PREC_DTYPE[prec_gemm]
        
        # Leer matrices A, B, C
        A = np.fromfile(f, dtype=dtype, count=m * k).reshape((m, k))
        B = np.fromfile(f, dtype=dtype, count=k * n).reshape((k, n))
        C = np.fromfile(f, dtype=dtype, count=m * n).reshape((m, n))

        return {
            "algo": "GEMM",
            "file_path": str(path),
            "m": m,
            "n": n,
            "k": k,
            "precision": prec_gemm,
            "A": A,
            "B": B,
            "C": C,
        }


def print_databank_info(data: Dict[str, Any], show_samples: int = 5) -> None:
    """Imprime información detallada y una vista previa de los datos contenidos.

    Args:
        data (Dict[str, Any]): Diccionario devuelto por `read_databank_file`.
        show_samples (int): Número de filas/elementos a mostrar en la vista previa.
    """
    print("\n" + "=" * 60)
    print(f" INFORMACIÓN DEL ARCHIVO DATABANK ({data['algo']})")
    print("=" * 60)
    print(f"Ruta: {data['file_path']}")
    print(f"Precisión: {data['precision']}")

    if data["algo"] == "GEMM":
        m, n, k = data["m"], data["n"], data["k"]
        print(f"Dimensiones: M={m}, N={n}, K={k}")
        print(f"Matriz A: shape={data['A'].shape}, dtype={data['A'].dtype}")
        print(f"Matriz B: shape={data['B'].shape}, dtype={data['B'].dtype}")
        print(f"Matriz C: shape={data['C'].shape}, dtype={data['C'].dtype}")

        print("\n--- Estadísticas de Matriz A ---")
        print(f"Min: {np.min(data['A'])}, Max: {np.max(data['A'])}, Mean: {np.mean(data['A']):.4f}, Std: {np.std(data['A']):.4f}")

        print("\n--- Muestra de Matriz A (primeras filas/columnas) ---")
        sub_a = data["A"][:show_samples, :min(show_samples, k)]
        print(sub_a)

        print("\n--- Muestra de Matriz B (primeras filas/columnas) ---")
        sub_b = data["B"][:show_samples, :min(show_samples, n)]
        print(sub_b)

    elif data["algo"] == "FFT":
        print(f"Dominio: {data['domain']}")
        print(f"Dimensiones: Nx={data['nx']}, Ny={data['ny']}, Nz={data['nz']}, Batch={data['batch']}")
        print(f"Entrada: size={len(data['input'])}, dtype={data['input'].dtype}")
        print(f"Salida:  size={len(data['output'])}, dtype={data['output'].dtype}")

        print("\n--- Estadísticas de la Entrada ---")
        print(f"Min: {np.min(data['input'])}, Max: {np.max(data['input'])}, Mean: {np.mean(data['input']):.4f}, Std: {np.std(data['input']):.4f}")

        print("\n--- Muestra de la Entrada (primeros elementos) ---")
        print(data["input"][:show_samples])

    print("=" * 60 + "\n")


def main() -> None:
    """Función principal CLI para inspección de archivos binarios."""
    parser = argparse.ArgumentParser(
        description="Herramienta para inspeccionar y visualizar archivos binarios del Databank (GEMM / FFT)."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Ruta absoluta o relativa al archivo binario (.bin) del databank.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Número de elementos o filas a mostrar en la vista previa (default: 5).",
    )

    args = parser.parse_args()

    try:
        data = read_databank_file(args.file_path)
        print_databank_info(data, show_samples=args.samples)
    except Exception as e:
        print(f"Error al leer el archivo: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
