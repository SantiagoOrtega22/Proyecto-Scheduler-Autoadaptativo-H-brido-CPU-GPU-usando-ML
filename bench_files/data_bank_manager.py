#!/usr/bin/env python3
"""
data_bank_manager.py
====================
DataBankManager: Banco de datos binario con política Lazy Cache para benchmarks
de GEMM y FFT orientados al entrenamiento de un agente DQN que optimiza EDP.

Los archivos generados están en el formato binario exacto que leen los kernels
C/CUDA del proyecto (gemm_cpu.c, gemm_gpu.cu, fft_cpu.c, fft_gpu.cu), por lo
que el runner puede pasar la ruta directamente sin pasos intermedios.

Estructura de directorios generada:
    <base_dir>/gemm/{prec}/matrix_{M}x{N}x{K}_{profile}.bin
    <base_dir>/fft/{domain}/{prec}/vector_{Nx}x{Ny}x{Nz}_b{batch}_{profile}.bin

Protocolo binario (GEMM):
    Header: M(int32) N(int32) K(int32) Precision(char)
    Body A:  M*K elementos en el dtype de la precisión
    Body B:  K*N elementos
    Body C:  M*N ceros

Protocolo binario (FFT):
    Header: Nx(int32) Ny(int32) Nz(int32) Batch(int32) Precision(char) Domain(3 chars)
    Body:   Elementos de entrada (real o complejo, según domain×precision)
            Elementos de salida  (ceros)

Tipos de datos por precisión (GEMM / FFT):
    S → np.float32     (4 bytes/elem)
    D → np.float64     (8 bytes/elem)
    C → np.complex64   (8 bytes/elem, interleaved float32 re+im)
    Z → np.complex128  (16 bytes/elem, interleaved float64 re+im)
"""

import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────────
SEED: int = 42

_PREC_LABEL: dict = {"S": "s", "D": "d", "C": "c", "Z": "z"}
_DOM_LABEL: dict = {"C2C": "c2c", "R2C": "r2c", "C2R": "c2r"}

_PREC_DTYPE: dict = {
    "S": np.float32,
    "D": np.float64,
    "C": np.complex64,
    "Z": np.complex128,
}
_REAL_PART: dict = {
    "S": np.float32,
    "D": np.float64,
    "C": np.float32,  # parte real de complex64
    "Z": np.float64,  # parte real de complex128
}


# ── Generación de la lista de tamaños ─────────────────────────────────────────

def generate_size_sweep(max_n: int = 8192) -> List[int]:
    """Genera la lista de tamaños N según la estrategia de 3 rangos.

    Rangos:
        Baja latencia   (64..1024):  paso +128   — modela overhead PCIe.
        Transición    (1025..4096):  paso +512.
        Cómputo intensivo  (>4096):  paso +1024  — hasta max_n.

    Args:
        max_n: Techo del rango compute-intensive (limitado por VRAM/RAM).

    Returns:
        List[int]: Lista ordenada de tamaños N.
    """
    sizes: List[int] = []

    # Rango 1: baja latencia
    n = 64
    while n <= 1024:
        sizes.append(n)
        n += 128

    # Rango 2: transición
    n = 1536
    while n <= 4096:
        sizes.append(n)
        n += 512

    # Rango 3: cómputo intensivo
    n = 5120
    while n <= max_n:
        sizes.append(n)
        n += 1024

    return sizes


# ── Clase principal ───────────────────────────────────────────────────────────

class DataBankManager:
    """Gestor de banco de datos binario con política Lazy Cache para kernels HPC.

    Genera archivos .bin compatibles con los kernels C/CUDA del proyecto solo
    cuando no existen en disco, reutilizándolos en ejecuciones sucesivas para
    garantizar que el tiempo de I/O no contamine las mediciones de energía.

    Attributes:
        base_dir: Raíz del banco de datos en disco.
        seed: Semilla fija para reproducibilidad.
        max_n: Techo del rango compute-intensive.
    """

    def __init__(
        self,
        base_dir: str = "bench_files/databank",
        seed: int = SEED,
        max_n: int = 8192,
    ) -> None:
        """Inicializa el DataBankManager.

        Args:
            base_dir: Ruta raíz del banco de datos. Se crea si no existe.
            seed: Semilla numpy para reproducibilidad (default: 42).
            max_n: Tamaño N máximo del sweep (ajusta según VRAM disponible).
        """
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.max_n = max_n

    # ── Rutas ─────────────────────────────────────────────────────────────────

    def _gemm_path(
        self, m: int, n: int, k: int, precision: str, profile: str
    ) -> Path:
        """Construye la ruta canónica del archivo GEMM en el banco."""
        prec = _PREC_LABEL[precision.upper()]
        return (
            self.base_dir
            / "gemm"
            / prec
            / f"matrix_{m}x{n}x{k}_{profile}.bin"
        )

    def _fft_path(
        self,
        nx: int,
        ny: int,
        nz: int,
        batch: int,
        precision: str,
        domain: str,
        profile: str,
    ) -> Path:
        """Construye la ruta canónica del archivo FFT en el banco."""
        prec = _PREC_LABEL[precision.upper()]
        dom = _DOM_LABEL[domain.upper()]
        if nz > 0:
            dims_str = f"{nx}x{ny}x{nz}"
        elif ny > 0:
            dims_str = f"{nx}x{ny}"
        else:
            dims_str = f"{nx}"
        return (
            self.base_dir
            / "fft"
            / dom
            / prec
            / f"vector_{dims_str}_b{batch}_{profile}.bin"
        )

    # ── Generadores de datos ──────────────────────────────────────────────────

    def _rng(self, offset: int = 0) -> np.random.Generator:
        """Devuelve un RNG determinista con la semilla del banco."""
        return np.random.default_rng(self.seed + offset)

    def _make_real_matrix(
        self, rows: int, cols: int, dtype: np.dtype, rng: np.random.Generator
    ) -> np.ndarray:
        """Genera una matriz real N(0,1) densa."""
        return rng.standard_normal((rows, cols)).astype(dtype)

    def _make_complex_matrix(
        self,
        rows: int,
        cols: int,
        cdtype: np.dtype,
        rdtype: np.dtype,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Genera una matriz compleja broadband densa."""
        re = rng.standard_normal((rows, cols)).astype(rdtype)
        im = rng.standard_normal((rows, cols)).astype(rdtype)
        return (re + 1j * im).astype(cdtype)

    # ── Escritura GEMM ────────────────────────────────────────────────────────

    def _write_gemm_file(
        self, path: Path, m: int, n: int, k: int, precision: str
    ) -> None:
        """Genera y escribe un archivo binario GEMM listo para los kernels.

        Formato:
            [M:i32][N:i32][K:i32][Prec:char] A_data B_data C_zeros

        Para precisiones complejas (C, Z), los datos se escriben como pares
        interleaved [re, im] que coinciden con cuComplex / cuDoubleComplex.

        Args:
            path: Ruta destino del archivo.
            m: Filas de A y C.
            n: Columnas de B y C.
            k: Columnas de A / filas de B.
            precision: S, D, C o Z.

        Raises:
            ValueError: Si la precisión no es válida.
        """
        prec = precision.upper()
        path.parent.mkdir(parents=True, exist_ok=True)

        rng_a = self._rng(offset=0)
        rng_b = self._rng(offset=1)  # offset distinto → matrices no idénticas

        if prec in ("S", "D"):
            dtype = _PREC_DTYPE[prec]
            A = self._make_real_matrix(m, k, dtype, rng_a)
            B = self._make_real_matrix(k, n, dtype, rng_b)
            C = np.zeros((m, n), dtype=dtype)
        elif prec in ("C", "Z"):
            cdtype = _PREC_DTYPE[prec]
            rdtype = _REAL_PART[prec]
            A = self._make_complex_matrix(m, k, cdtype, rdtype, rng_a)
            B = self._make_complex_matrix(k, n, cdtype, rdtype, rng_b)
            C = np.zeros((m, n), dtype=cdtype)
        else:
            raise ValueError(f"Precisión GEMM no válida: {precision}")

        with open(path, "wb") as f:
            # Header: M, N, K como int32 little-endian + Precision como ASCII
            f.write(struct.pack("<i", m))
            f.write(struct.pack("<i", n))
            f.write(struct.pack("<i", k))
            f.write(prec.encode("ascii"))
            # Body: numpy.tofile escribe en orden de memoria (little-endian en
            # sistemas x86). Para complejos, numpy almacena [re, im] interleaved,
            # lo que coincide exactamente con cuComplex/cuDoubleComplex.
            A.ravel().tofile(f)
            B.ravel().tofile(f)
            C.ravel().tofile(f)

    def get_gemm_path(
        self,
        m: int,
        n: int,
        k: int,
        precision: str,
        profile: str = "dense_normal",
    ) -> str:
        """Retorna la ruta al archivo GEMM, generándolo lazily si no existe.

        Args:
            m: Filas de A y C.
            n: Columnas de B y C.
            k: Columnas de A / filas de B.
            precision: Precisión GEMM: S, D, C o Z.
            profile: Nombre del perfil de datos (etiqueta en el nombre de archivo).

        Returns:
            str: Ruta absoluta al archivo binario kernel-ready.
        """
        path = self._gemm_path(m, n, k, precision, profile)
        if not path.exists():
            self._write_gemm_file(path, m, n, k, precision)
        return str(path)

    # ── Escritura FFT ─────────────────────────────────────────────────────────

    @staticmethod
    def _fft_n_total(nx: int, ny: int, nz: int) -> int:
        """Producto de las dimensiones activas."""
        total = nx
        if ny > 0:
            total *= ny
        if nz > 0:
            total *= nz
        return total

    @staticmethod
    def _r2c_complex_elems(nx: int, ny: int, nz: int) -> int:
        """Número de elementos complejos en la salida R2C (último eje N//2+1)."""
        last = nz if nz > 0 else (ny if ny > 0 else nx)
        outer = DataBankManager._fft_n_total(nx, ny, nz) // last
        return outer * (last // 2 + 1)

    def _make_fft_signal(
        self,
        n_total: int,
        precision: str,
        domain: str,
    ) -> np.ndarray:
        """Genera la señal de entrada para FFT (broadband por defecto).

        Para C2C y C2R se genera una señal compleja broadband.
        Para R2C se genera ruido gaussiano real.

        Args:
            n_total: Número total de elementos (dims × batch).
            precision: S o D.
            domain: C2C, R2C o C2R.

        Returns:
            np.ndarray: Señal en el dtype correcto.
        """
        prec = precision.upper()
        dom = domain.upper()
        rng = self._rng(offset=10)

        rdtype = _REAL_PART[prec]

        if dom in ("C2C", "C2R"):
            # Entrada compleja broadband.
            # Se construye asignando partes real e imaginaria para evitar ComplexWarning.
            cdtype = _PREC_DTYPE[prec]
            rdtype = _REAL_PART[prec]
            re = rng.standard_normal(n_total).astype(rdtype)
            im = rng.standard_normal(n_total).astype(rdtype)
            # Construir el array complejo interleaved directamente via view para
            # evitar el ValueError de asignación a .real/.imag en complex64.
            elem_bytes = re.itemsize * 2  # 8 para c64, 16 para c128
            out = np.empty(n_total * elem_bytes // re.itemsize, dtype=rdtype)
            out[0::2] = re
            out[1::2] = im
            return out.view(cdtype)
        else:
            # R2C: entrada real
            return rng.standard_normal(n_total).astype(rdtype)

    def _write_fft_file(
        self,
        path: Path,
        nx: int,
        ny: int,
        nz: int,
        batch: int,
        precision: str,
        domain: str,
    ) -> None:
        """Genera y escribe un archivo binario FFT listo para los kernels.

        Formato:
            [Nx:i32][Ny:i32][Nz:i32][Batch:i32][Prec:char][Domain:3chars]
            input_data  output_zeros

        Args:
            path: Ruta destino del archivo.
            nx, ny, nz: Dimensiones del transform (ny=nz=0 para 1D).
            batch: Número de transforms por invocación.
            precision: S o D.
            domain: C2C, R2C o C2R.

        Raises:
            ValueError: Si el dominio no es válido.
        """
        prec = precision.upper()
        dom = domain.upper()
        path.parent.mkdir(parents=True, exist_ok=True)

        n_total = self._fft_n_total(nx, ny, nz)
        n_total_batched = n_total * batch
        n_complex_batched = self._r2c_complex_elems(nx, ny, nz) * batch

        in_data = self._make_fft_signal(n_total_batched, prec, dom)

        # Salida: ceros en el dtype correcto
        if dom == "C2C":
            out_dtype = _PREC_DTYPE[prec]  # complex64 / complex128
            out_data = np.zeros(n_total_batched, dtype=out_dtype)
        elif dom == "R2C":
            out_dtype = np.complex64 if prec == "S" else np.complex128
            out_data = np.zeros(n_complex_batched, dtype=out_dtype)
        elif dom == "C2R":
            out_dtype = np.float32 if prec == "S" else np.float64
            out_data = np.zeros(n_total_batched, dtype=out_dtype)
        else:
            raise ValueError(f"Dominio FFT no válido: {domain}")

        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<i", nx))
            f.write(struct.pack("<i", ny))
            f.write(struct.pack("<i", nz))
            f.write(struct.pack("<i", batch))
            f.write(prec.encode("ascii"))
            f.write(dom.encode("ascii"))  # 3 bytes: "C2C", "R2C", o "C2R"
            # Body
            in_data.ravel().tofile(f)
            out_data.ravel().tofile(f)

    def get_fft_path(
        self,
        nx: int,
        ny: int,
        nz: int,
        batch: int,
        precision: str,
        domain: str,
        profile: str = "broadband",
    ) -> str:
        """Retorna la ruta al archivo FFT, generándolo lazily si no existe.

        Args:
            nx, ny, nz: Dimensiones del transform (ny=nz=0 para 1D).
            batch: Número de transforms por invocación.
            precision: S o D.
            domain: C2C, R2C o C2R.
            profile: Etiqueta de perfil en el nombre del archivo.

        Returns:
            str: Ruta absoluta al archivo binario kernel-ready.
        """
        path = self._fft_path(nx, ny, nz, batch, precision, domain, profile)
        if not path.exists():
            self._write_fft_file(path, nx, ny, nz, batch, precision, domain)
        return str(path)

    # ── Utilidades de precarga ─────────────────────────────────────────────────

    def generate_size_sweep(self) -> List[int]:
        """Devuelve la lista completa de N para el sweep DQN."""
        return generate_size_sweep(self.max_n)

    def preload_gemm(
        self,
        precisions: Optional[List[str]] = None,
        profile: str = "dense_normal",
    ) -> None:
        """Pre-genera todos los archivos GEMM para el sweep completo.

        Args:
            precisions: Lista de precisiones a precargar. Default: [S, D, C, Z].
            profile: Perfil de generación de datos.
        """
        if precisions is None:
            precisions = ["S", "D", "C", "Z"]
        sizes = self.generate_size_sweep()
        total = len(sizes) * len(precisions)
        done = 0
        for n in sizes:
            for prec in precisions:
                self.get_gemm_path(n, n, n, prec, profile)
                done += 1
                print(f"  GEMM preload [{done}/{total}] N={n} P={prec}", end="\r")
        print()

    def preload_fft(
        self,
        precisions: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        profile: str = "broadband",
    ) -> None:
        """Pre-genera todos los archivos FFT 1D para el sweep completo.

        Args:
            precisions: Lista de precisiones. Default: [S, D].
            domains: Lista de dominios. Default: [C2C, R2C, C2R].
            profile: Perfil de generación de datos.
        """
        if precisions is None:
            precisions = ["S", "D"]
        if domains is None:
            domains = ["C2C", "R2C", "C2R"]
        sizes = self.generate_size_sweep()
        total = len(sizes) * len(precisions) * len(domains)
        done = 0
        for n in sizes:
            for prec in precisions:
                for dom in domains:
                    self.get_fft_path(n, 0, 0, 1, prec, dom, profile)
                    done += 1
                    print(f"  FFT preload [{done}/{total}] N={n} P={prec} D={dom}", end="\r")
        print()

    def stats(self) -> dict:
        """Devuelve estadísticas del banco en disco.

        Returns:
            dict con total_files, total_bytes, paths por subdirectorio.
        """
        total_files = 0
        total_bytes = 0
        for p in self.base_dir.rglob("*.bin"):
            total_files += 1
            total_bytes += p.stat().st_size
        return {
            "base_dir": str(self.base_dir),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1e6, 2),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest(base_dir: str = "bench_files/databank") -> None:
    """Verifica la generación y el formato de los archivos producidos.

    Args:
        base_dir: Directorio de prueba para el banco.
    """
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp(prefix="dbm_test_")
    try:
        mgr = DataBankManager(base_dir=tmp, seed=42, max_n=2048)

        print("=== Sweep de tamaños ===")
        sizes = mgr.generate_size_sweep()
        print(f"  {len(sizes)} tamaños: {sizes[:4]} ... {sizes[-2:]}")

        print("\n=== GEMM S (float32) ===")
        path = mgr.get_gemm_path(128, 128, 128, "S")
        with open(path, "rb") as f:
            m, n, k = struct.unpack("<iii", f.read(12))
            prec = f.read(1).decode("ascii")
            raw_a = np.frombuffer(f.read(m * k * 4), dtype=np.float32)
        assert m == 128 and prec == "S" and len(raw_a) == m * k
        print(f"  OK  M={m} N={n} K={k} P={prec}  A[0]={raw_a[0]:.4f}")

        print("=== GEMM C (complex64) ===")
        path = mgr.get_gemm_path(64, 64, 64, "C")
        with open(path, "rb") as f:
            m, n, k = struct.unpack("<iii", f.read(12))
            prec = f.read(1).decode("ascii")
            raw_a = np.frombuffer(f.read(m * k * 8), dtype=np.complex64)
        assert m == 64 and prec == "C"
        print(f"  OK  M={m} N={n} K={k} P={prec}  A[0]={raw_a[0]}")

        print("=== GEMM Z (complex128) ===")
        path = mgr.get_gemm_path(64, 64, 64, "Z")
        with open(path, "rb") as f:
            m, n, k = struct.unpack("<iii", f.read(12))
            prec = f.read(1).decode("ascii")
            raw_a = np.frombuffer(f.read(m * k * 16), dtype=np.complex128)
        assert prec == "Z"
        print(f"  OK  M={m} N={n} K={k} P={prec}  A[0]={raw_a[0]}")

        print("=== FFT C2C S (complex64) ===")
        path = mgr.get_fft_path(512, 0, 0, 1, "S", "C2C")
        with open(path, "rb") as f:
            nx, ny, nz, batch = struct.unpack("<iiii", f.read(16))
            prec = f.read(1).decode("ascii")
            dom = f.read(3).decode("ascii")
            raw = np.frombuffer(f.read(nx * 8), dtype=np.complex64)
        assert nx == 512 and prec == "S" and dom == "C2C"
        print(f"  OK  Nx={nx} Ny={ny} Nz={nz} Batch={batch} P={prec} D={dom}")

        print("=== FFT R2C D (float64 in → complex128 out) ===")
        path = mgr.get_fft_path(256, 0, 0, 1, "D", "R2C")
        with open(path, "rb") as f:
            nx, ny, nz, batch = struct.unpack("<iiii", f.read(16))
            prec = f.read(1).decode("ascii")
            dom = f.read(3).decode("ascii")
        assert prec == "D" and dom == "R2C"
        print(f"  OK  Nx={nx} Ny={ny} Nz={nz} Batch={batch} P={prec} D={dom}")

        print("=== Lazy: segundo acceso no regenera el archivo ===")
        p1 = mgr.get_gemm_path(128, 128, 128, "S")
        mtime1 = os.path.getmtime(p1)
        p2 = mgr.get_gemm_path(128, 128, 128, "S")
        mtime2 = os.path.getmtime(p2)
        assert mtime1 == mtime2, "El archivo fue regenerado (no debería)"
        print(f"  OK  mtime invariante: {mtime1}")

        st = mgr.stats()
        print(f"\n=== Estadísticas ===\n  Archivos: {st['total_files']}  Tamaño: {st['total_mb']} MB")
        print("\n✓ Todos los self-tests pasaron correctamente.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataBankManager CLI")
    sub = parser.add_subparsers(dest="cmd")

    # Subcomando: selftest
    sub.add_parser("selftest", help="Ejecuta el conjunto de pruebas internas")

    # Subcomando: preload
    p_pre = sub.add_parser("preload", help="Pre-genera el banco completo en disco")
    p_pre.add_argument("--base-dir", default="bench_files/databank")
    p_pre.add_argument("--max-n", type=int, default=8192)
    p_pre.add_argument("--algo", choices=["gemm", "fft", "all"], default="all")

    # Subcomando: stats
    p_st = sub.add_parser("stats", help="Muestra estadísticas del banco en disco")
    p_st.add_argument("--base-dir", default="bench_files/databank")

    args = parser.parse_args()

    if args.cmd == "selftest":
        _selftest()
    elif args.cmd == "preload":
        mgr = DataBankManager(base_dir=args.base_dir, max_n=args.max_n)
        if args.algo in ("gemm", "all"):
            print("Precargando GEMM...")
            mgr.preload_gemm()
        if args.algo in ("fft", "all"):
            print("Precargando FFT...")
            mgr.preload_fft()
        print(mgr.stats())
    elif args.cmd == "stats":
        mgr = DataBankManager(base_dir=args.base_dir)
        st = mgr.stats()
        print(f"Base: {st['base_dir']}")
        print(f"Archivos: {st['total_files']}")
        print(f"Tamaño:   {st['total_mb']} MB")
    else:
        parser.print_help()
