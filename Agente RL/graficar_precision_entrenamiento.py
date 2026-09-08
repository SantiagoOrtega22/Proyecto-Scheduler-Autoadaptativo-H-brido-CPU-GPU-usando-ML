"""
graficar_precision_entrenamiento.py

Grafica la curva de convergencia (precisión vs. steps de entrenamiento) que
`PrecisionCallback` (ver train.py) registra en TensorBoard bajo la etiqueta
`metricas_personalizadas/precision`: el % de decisiones del agente que
coinciden con el óptimo verificable (argmin(edp_cpu, edp_gpu)), calculado
sobre ventanas móviles de 100 steps durante el entrenamiento.

No reemplaza la medición post-entrenamiento de `diagnostico_holdout.py`
(precisión con política determinista, sin exploración, sobre un split fijo):
esta gráfica muestra cómo evolucionó esa métrica mientras el agente todavía
exploraba (epsilon-greedy activo), por eso luce más ruidosa.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from estilo_graficas import (
    COLOR_TEXTO_PRIMARIO,
    PALETA_CATEGORICA,
    configurar_estilo_pastel,
    limpiar_bordes,
)

TAG_PRECISION = "metricas_personalizadas/precision"


def _detectar_run_mas_reciente(logs_dir: str) -> str:
    """Encuentra el directorio de run (`DQN_*`) modificado más recientemente.

    Se usa como valor por defecto porque, en la práctica, el run más nuevo es
    el que corresponde al modelo de tesis vigente (`modelo_dqn_scheduler.zip`).

    Args:
        logs_dir: Ruta al directorio `logs_entrenamiento`.

    Returns:
        str: Ruta al subdirectorio del run más reciente.

    Raises:
        FileNotFoundError: Si no hay ningún directorio `DQN_*` bajo `logs_dir`.
    """
    candidatos = glob.glob(os.path.join(logs_dir, "DQN_*"))
    if not candidatos:
        raise FileNotFoundError(f"No se encontraron runs 'DQN_*' en {logs_dir}")
    return max(candidatos, key=os.path.getmtime)


def cargar_precision(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Lee la serie de precisión registrada en un run de TensorBoard.

    Args:
        run_dir: Directorio del run (contiene el archivo `events.out.tfevents.*`).

    Returns:
        tuple[np.ndarray, np.ndarray]: (steps, precision en porcentaje [0, 100]).

    Raises:
        KeyError: Si el run no tiene la etiqueta `metricas_personalizadas/precision`
            (p. ej. un run entrenado antes de agregar `PrecisionCallback`).
    """
    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    if TAG_PRECISION not in ea.Tags().get("scalars", []):
        raise KeyError(f"El run {run_dir} no contiene la etiqueta '{TAG_PRECISION}'")

    eventos = ea.Scalars(TAG_PRECISION)
    steps = np.array([e.step for e in eventos], dtype=np.float64)
    precision_pct = np.array([e.value for e in eventos], dtype=np.float64) * 100.0
    return steps, precision_pct


def suavizar_ema(valores: np.ndarray, peso: float = 0.85) -> np.ndarray:
    """Suaviza una serie ruidosa con una media móvil exponencial (EMA).

    Misma técnica de suavizado que usa la UI de TensorBoard, para que la
    curva destacada sea legible sin ocultar la serie cruda de fondo.

    Args:
        valores: Serie original (p. ej. precisión por ventana de 100 steps).
        peso: Peso del valor suavizado previo en [0, 1). Más alto = más suave.

    Returns:
        np.ndarray: Serie suavizada, mismo largo que `valores`.
    """
    suavizada = np.empty_like(valores)
    acumulado = valores[0]
    for i, v in enumerate(valores):
        acumulado = peso * acumulado + (1 - peso) * v
        suavizada[i] = acumulado
    return suavizada


def generar_grafica_precision(run_dir: str, img_salida: str) -> None:
    """Genera la gráfica de precisión vs. steps de entrenamiento.

    Args:
        run_dir: Directorio del run de TensorBoard a graficar.
        img_salida: Ruta de archivo donde guardar el PNG.
    """
    steps, precision_pct = cargar_precision(run_dir)
    precision_suave = suavizar_ema(precision_pct)

    color_linea = PALETA_CATEGORICA[0]  # azul pastel, serie unica

    configurar_estilo_pastel()
    fig, ax = plt.subplots(figsize=(11, 6))

    # Serie cruda: fina y translúcida, de fondo (contexto/ruido de exploración).
    ax.plot(steps, precision_pct, color=color_linea, linewidth=1, alpha=0.35, zorder=2, label="Cruda (ventana de 100 steps)")
    # Serie suavizada (EMA): la curva que realmente se lee.
    ax.plot(steps, precision_suave, color=color_linea, linewidth=2.5, solid_capstyle="round", zorder=3, label="Suavizada (EMA)")

    valor_final = precision_suave[-1]
    ax.annotate(
        f"{valor_final:.1f}%",
        xy=(steps[-1], valor_final),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color=COLOR_TEXTO_PRIMARIO,
    )

    ax.set_title("Convergencia del Agente RL: Precisión vs. Steps de Entrenamiento", fontsize=14, fontweight="bold", color=COLOR_TEXTO_PRIMARIO)
    ax.set_xlabel("Steps de entrenamiento", fontsize=12)
    ax.set_ylabel("Precisión (% de decisiones óptimas)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1000:.0f}k")
    limpiar_bordes(ax)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(img_salida, dpi=300)
    print(f"[ÉXITO] Gráfica de convergencia guardada en: {img_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Directorio del run de TensorBoard (por defecto: el más reciente en logs_entrenamiento/)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs_entrenamiento")
    run_dir = args.run_dir or _detectar_run_mas_reciente(logs_dir)
    img_salida = os.path.join(base_dir, "convergencia_precision.png")

    print(f"Leyendo run: {run_dir}")
    generar_grafica_precision(run_dir, img_salida)
