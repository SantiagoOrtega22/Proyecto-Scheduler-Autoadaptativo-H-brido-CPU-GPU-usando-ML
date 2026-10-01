"""
graficar_tiempo_ejecucion.py

Grafica tiempo de ejecucion, energia y EDP (CPU vs GPU) en funcion del
tamano N, a partir de `Entorno/dataset_rl.csv` (corpus del libro, sin la
campana FFT 1D de tamanos pequenos). Se genera una figura de
tres paneles (tiempo | energia | EDP) por carga, en `graficas_tiempo_energia/`:

    gemm.png, fft_1d.png, fft_2d.png, fft_3d.png

El dataset guarda las dimensiones como log2(dim) / MAX_LOG2_DIM, por lo que
aqui se decodifican a tamanos enteros. Para mantener las curvas comparables
se fija un solo perfil por carga (ver preparar_*): el promedio entre
configuraciones distintas mezclaria efectos de precision/direccion con el de N.

En FFT, N es el lado del problema: 1D -> N puntos, 2D -> NxN, 3D -> NxNxN.

Uso:
    python graficar_tiempo_ejecucion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from estilo_graficas import (
    COLOR_CPU,
    COLOR_GPU,
    configurar_estilo_pastel,
    limpiar_bordes,
)

RUTA_DATASET = Path(__file__).parent / "Entorno" / "dataset_rl.csv"
DIR_SALIDA = Path(__file__).parent / "graficas_tiempo_energia"

# Debe coincidir con MAX_LOG2_DIM de Entorno/codificador_csv.py.
MAX_LOG2_DIM = 26.0

# metrica -> (prefijo de columna, etiqueta del eje Y, factor de conversion)
# El dataset guarda tiempo en s, energia en J y EDP en J*s; el tiempo se muestra en ms.
METRICAS: dict[str, tuple[str, str, float]] = {
    "tiempo": ("time", "Tiempo de ejecucion (ms)", 1e3),
    "energia": ("energy", "Energia (J)", 1.0),
    "edp": ("edp", "EDP (J*s)", 1.0),
}


def decodificar_dimension(columna_log2: pd.Series) -> pd.Series:
    """Invierte la normalizacion log2(dim)/MAX_LOG2_DIM del dataset.

    Args:
        columna_log2: Columna normalizada (Dim_*_log2).

    Returns:
        Serie de enteros con el tamano original de la dimension.
    """
    return np.round(2 ** (columna_log2 * MAX_LOG2_DIM)).astype(int)


def agrupar_por_n(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega la mediana de tiempo y energia de CPU/GPU por tamano N.

    Args:
        datos: Filas ya filtradas a un perfil, con columna N.

    Returns:
        DataFrame con N y las columnas cpu_/gpu_ de cada metrica de `METRICAS`.
    """
    columnas = [f"{disp}_{pref}" for pref, _, _ in METRICAS.values() for disp in ("cpu", "gpu")]
    return datos.groupby("N")[columnas].median().reset_index()


def preparar_gemm(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra GEMM sgemm con matrices cuadradas y OpA=OpB=N.

    Args:
        df: Dataset completo.

    Returns:
        DataFrame agrupado por N (ver `agrupar_por_n`).
    """
    gemm = df[(df.is_GEMM == 1) & (df.Prec_S == 1) & (df.OpA_N == 1) & (df.OpB_N == 1)].copy()
    gemm["N"] = decodificar_dimension(gemm.Dim_1_log2)
    return agrupar_por_n(gemm)


def preparar_fft(df: pd.DataFrame, dimensiones: int) -> pd.DataFrame:
    """Filtra FFT C2C, directa, precision simple, batch 1 para 1D, 2D o 3D.

    Args:
        df: Dataset completo.
        dimensiones: 1, 2 o 3. Se distingue por Dim_2/Dim_3 (log2 = 0 <=> dim 1).

    Returns:
        DataFrame agrupado por N (lado del problema, ver `agrupar_por_n`).
    """
    filtro = (
        (df.is_FFT == 1) & (df.Prec_S == 1) & (df.Dom_C2C == 1)
        & (df.Dir_F == 1) & (df.Batch_log2 == 0)
        & ((df.Dim_2_log2 > 0) == (dimensiones >= 2))
        & ((df.Dim_3_log2 > 0) == (dimensiones >= 3))
    )
    fft = df[filtro].copy()
    fft["N"] = decodificar_dimension(fft.Dim_1_log2)
    return agrupar_por_n(fft)


def graficar_paneles(datos: pd.DataFrame, perfil: str, ruta: Path) -> None:
    """Dibuja tiempo, energia y EDP vs N (ejes log) en una figura de tres paneles.

    Args:
        datos: DataFrame con N y las columnas cpu_/gpu_ de cada metrica.
        perfil: Descripcion de la carga y el perfil fijado (titulo de la figura).
        ruta: Archivo PNG de salida.
    """
    fig, ejes = plt.subplots(1, len(METRICAS), figsize=(18, 5))
    for ax, (metrica, (prefijo, etiqueta_y, factor)) in zip(ejes, METRICAS.items()):
        ax.plot(datos.N, datos[f"cpu_{prefijo}"] * factor, color=COLOR_CPU, marker="o", markersize=3, label="CPU")
        ax.plot(datos.N, datos[f"gpu_{prefijo}"] * factor, color=COLOR_GPU, marker="o", markersize=3, label="GPU")
        # Escala log-log: N y las metricas abarcan varios ordenes de magnitud.
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Tamano N")
        ax.set_ylabel(etiqueta_y)
        ax.set_title(metrica.upper() if metrica == "edp" else metrica.capitalize())
        ax.legend()
        limpiar_bordes(ax)
    fig.suptitle(perfil, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main() -> None:
    """Genera la figura de paneles de GEMM y FFT (1D/2D/3D)."""
    df = pd.read_csv(RUTA_DATASET)
    configurar_estilo_pastel()
    DIR_SALIDA.mkdir(exist_ok=True)

    # Cada entrada: prefijo de archivo, datos, descripcion del perfil fijado.
    conjuntos = [("gemm", preparar_gemm(df), "GEMM (sgemm, NxNxN, transA=N, transB=N)")]
    conjuntos += [
        (f"fft_{d}d", preparar_fft(df, d), f"FFT {d}D (C2C, directa, precision simple, batch 1)")
        for d in (1, 2, 3)
    ]
    for prefijo, datos, perfil in conjuntos:
        graficar_paneles(datos, perfil, DIR_SALIDA / f"{prefijo}.png")


if __name__ == "__main__":
    main()
