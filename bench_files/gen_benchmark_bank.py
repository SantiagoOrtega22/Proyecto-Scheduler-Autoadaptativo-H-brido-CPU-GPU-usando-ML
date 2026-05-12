"""
gen_benchmark_bank.py
=====================
Genera un banco de entradas reproducible para benchmarks de GEMM y FFT.

Salida: benchmark_bank.h5  (HDF5, compatible con C/C++/Python/Julia/MATLAB)
        benchmark_bank_manifest.txt  (resumen legible)

Criterios de calidad para benchmark:
  - Sin ceros artificiales (densidad > 99 %)
  - Rango de valores balanceado (media ≈ 0, varianza ≈ 1) → evita overflow/underflow
  - Condición numérica moderada (matrices no singulares, no mal condicionadas)
  - Semilla fija → resultados reproducibles entre GPU y CPU
  - Tipos: float32 (GPU-friendly) y float64 (referencia)
  - Para GEMM: tres perfiles por tamaño (aleatorio-denso, uniforme-denso, ill-conditioned)
  - Para FFT: señales con distintas cargas espectrales (single-tone, multi-tone, broadband, impulse-train)
"""

import numpy as np
import h5py
import os
import textwrap
from datetime import datetime

# ─── Configuración ──────────────────────────────────────────────────────────
SEED       = 42
GEMM_SIZES = [128, 256, 512, 1024, 2048, 4096]
FFT_SIZES  = [256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 1048576]
OUT_FILE   = "benchmark_bank.h5"
MAN_FILE   = "benchmark_bank_manifest.txt"

rng = np.random.default_rng(SEED)

manifest_lines = [
    "=" * 70,
    "  BENCHMARK INPUT BANK — Generado: " + datetime.now().isoformat(timespec='seconds'),
    "  Semilla: {}   |  Archivo: benchmark_bank.h5".format(SEED),
    "=" * 70,
    "",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def density(arr):
    """Fracción de elementos != 0."""
    return np.count_nonzero(arr) / arr.size

def make_dense_normal(n, dtype):
    """N(0,1) garantiza densidad ~100 %, media≈0, varianza≈1."""
    A = rng.standard_normal((n, n)).astype(dtype)
    B = rng.standard_normal((n, n)).astype(dtype)
    return A, B

def make_dense_uniform(n, dtype):
    """Uniforme en [-1, 1], centrado, sin ceros estructurales."""
    A = (rng.random((n, n)).astype(dtype) * 2 - 1)
    B = (rng.random((n, n)).astype(dtype) * 2 - 1)
    A[A == 0] = 1e-6
    B[B == 0] = 1e-6
    return A, B

def make_ill_conditioned(n, dtype):
    """
    Matrices con número de condición moderado (~1e4):
    A = U · diag(logspace) · Vt  (SVD controlado).
    Interesante porque estresa la precisión de la FMA.
    """
    U, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64))
    sv = np.logspace(0, 4, n, dtype=np.float64)   # cond ≈ 1e4
    A = (U * sv) @ V.T
    U2, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64))
    V2, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64))
    sv2 = np.logspace(0, 2, n, dtype=np.float64)  # cond ≈ 1e2
    B = (U2 * sv2) @ V2.T
    return A.astype(dtype), B.astype(dtype)

# ─── FFT signal generators ────────────────────────────────────────────────────

def sig_single_tone(n, dtype):
    t = np.arange(n, dtype=np.float64)
    freq = n // 8
    s = np.sin(2 * np.pi * freq * t / n)
    return s.astype(dtype)

def sig_multi_tone(n, dtype):
    t = np.arange(n, dtype=np.float64)
    freqs  = [n//64, n//32, n//16, n//8, n//4]
    amps   = [1.0, 0.7, 0.5, 0.3, 0.15]
    s = sum(a * np.sin(2 * np.pi * f * t / n) for f, a in zip(freqs, amps))
    return s.astype(dtype)

def sig_broadband(n, dtype):
    s = rng.standard_normal(n)
    return s.astype(dtype)

def sig_impulse_train(n, dtype):
    s = np.zeros(n, dtype=np.float64)
    period = max(n // 16, 1)
    s[::period] = 1.0
    s += rng.standard_normal(n) * 1e-3
    return s.astype(dtype)

def sig_chirp(n, dtype):
    t = np.arange(n, dtype=np.float64) / n
    f0, f1 = 1.0, n / 2 - 1
    s = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
    return s.astype(dtype)

def sig_complex_broadband(n, dtype_real):
    cdtype = np.complex64 if dtype_real == np.float32 else np.complex128
    real = rng.standard_normal(n)
    imag = rng.standard_normal(n)
    return (real + 1j * imag).astype(cdtype)

FFT_PROFILES = [
    ("single_tone",   sig_single_tone),
    ("multi_tone",    sig_multi_tone),
    ("broadband",     sig_broadband),
    ("impulse_train", sig_impulse_train),
    ("chirp",         sig_chirp),
]

# ─── Generación ──────────────────────────────────────────────────────────────

with h5py.File(OUT_FILE, "w") as hf:
    hf.attrs["seed"]    = SEED
    hf.attrs["created"] = datetime.now().isoformat()
    hf.attrs["description"] = (
        "Reproducible benchmark input bank for GEMM and FFT. "
        "All matrices/signals designed for high density, balanced range, "
        "and non-trivial numerical content."
    )

    # ── GEMM ────────────────────────────────────────────────────────────────
    gemm_grp = hf.create_group("gemm")
    manifest_lines += ["", "── GEMM " + "─"*62, ""]

    GEMM_PROFILES = [
        ("dense_normal",    make_dense_normal,    "N(0,1) aleatorio — referencia estándar"),
        ("dense_uniform",   make_dense_uniform,   "Uniforme[-1,1] — sin sesgo de distribución"),
        ("ill_conditioned", make_ill_conditioned, "Cond≈1e4 — estressa precisión FMA"),
    ]

    for N in GEMM_SIZES:
        n_grp = gemm_grp.create_group(f"N{N}")
        manifest_lines.append(f"  N = {N:5d}  ({N}×{N}  =  {N*N:,} elementos por matriz)")
        for prof_name, gen_fn, desc in GEMM_PROFILES:
            p_grp = n_grp.create_group(prof_name)
            for dtype, label in [(np.float32, "f32"), (np.float64, "f64")]:
                A, B = gen_fn(N, dtype)
                p_grp.create_dataset(f"A_{label}", data=A,
                                     compression="gzip", compression_opts=1)
                p_grp.create_dataset(f"B_{label}", data=B,
                                     compression="gzip", compression_opts=1)
                dens_A = density(A)
                dens_B = density(B)
                manifest_lines.append(
                    f"    [{prof_name:16s}] {label}  "
                    f"densidad A={dens_A:.4f}  B={dens_B:.4f}  "
                    f"μ_A={A.mean():.4f}  σ_A={A.std():.4f}  — {desc}"
                )
        manifest_lines.append("")

    # ── FFT ─────────────────────────────────────────────────────────────────
    fft_grp = hf.create_group("fft")
    manifest_lines += ["", "── FFT " + "─"*63, ""]

    for N in FFT_SIZES:
        n_grp = fft_grp.create_group(f"N{N}")
        manifest_lines.append(f"  N = {N:8,d}  (señal 1-D)")
        for prof_name, gen_fn in FFT_PROFILES:
            p_grp = n_grp.create_group(prof_name)
            for dtype, label in [(np.float32, "f32"), (np.float64, "f64")]:
                s = gen_fn(N, dtype)
                p_grp.create_dataset(label, data=s,
                                     compression="gzip", compression_opts=1)
            for dtype_r, label in [(np.float32, "c64"), (np.float64, "c128")]:
                sc = sig_complex_broadband(N, dtype_r)
                p_grp.create_dataset(label, data=sc,
                                     compression="gzip", compression_opts=1)
            manifest_lines.append(
                f"    [{prof_name:16s}]  perfiles: f32, f64, c64, c128"
            )
        manifest_lines.append("")

# ─── Manifiesto ─────────────────────────────────────────────────────────────
manifest_lines += [
    "", "=" * 70,
    "ESTRUCTURA HDF5:",
    "  /gemm/N{size}/{profile}/A_f32   float32 (N×N)",
    "  /gemm/N{size}/{profile}/A_f64   float64 (N×N)",
    "  /gemm/N{size}/{profile}/B_f32   float32 (N×N)",
    "  /gemm/N{size}/{profile}/B_f64   float64 (N×N)",
    "  /fft/N{size}/{profile}/f32      float32 (N,)",
    "  /fft/N{size}/{profile}/f64      float64 (N,)",
    "  /fft/N{size}/{profile}/c64      complex64 (N,)",
    "  /fft/N{size}/{profile}/c128     complex128 (N,)",
    "=" * 70,
]

with open(MAN_FILE, "w") as mf:
    mf.write("\n".join(manifest_lines))

print("✓ benchmark_bank.h5 generado")
print("✓ benchmark_bank_manifest.txt generado")

with h5py.File(OUT_FILE, "r") as hf:
    def count_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            count_datasets.n += 1
    count_datasets.n = 0
    hf.visititems(count_datasets)
    print(f"  Datasets totales: {count_datasets.n}")
    print(f"  Tamaño en disco:  {os.path.getsize(OUT_FILE)/1e6:.1f} MB")