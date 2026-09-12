"""
barrido_semillas.py

Entrena N agentes DQN independientes variando únicamente la semilla del algoritmo
de aprendizaje y reporta la dispersión de sus resultados (media ± desviación
estándar), como evidencia de reproducibilidad.

Motivación metodológica: el entrenamiento DQN es estocástico en la inicialización
de la red, en la exploración epsilon-greedy y en el muestreo de minibatches del
replay buffer. Una sola corrida es un único punto de muestra y no permite
distinguir una política robusta de una semilla afortunada (Henderson et al., 2018,
"Deep Reinforcement Learning that Matters"). Este script cuantifica esa varianza.

Aislamiento del experimento: la semilla que varía es EXCLUSIVAMENTE la del DQN.
El split del dataset y la cola de tareas de evaluación quedan fijos en BENCH_SEED,
de modo que las 5 corridas ven exactamente los mismos datos y se evalúan sobre
exactamente la misma carga de trabajo. Cualquier diferencia observada es atribuible
al proceso de aprendizaje, no a los datos.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Entorno.gym import PlanificadorEnv
from estilo_graficas import (
    COLOR_ACENTO_RL,
    COLOR_CPU,
    COLOR_GPU,
    COLOR_SUPERFICIE,
    COLOR_TEXTO_PRIMARIO,
    COLOR_TEXTO_SECUNDARIO,
    PALETA_CATEGORICA,
    configurar_estilo_pastel,
    limpiar_bordes,
)
from graficar_precision_entrenamiento import cargar_precision, suavizar_ema
from train import BENCH_SEED, entrenar_agente

# Semillas de las corridas independientes. La primera es BENCH_SEED para que la
# corrida de referencia del documento siga siendo parte del reporte estadístico.
SEMILLAS: list[int] = [42, 43, 44, 45, 46]

SUBDIR_MODELOS = "modelos_semillas"
SUBDIR_LOGS = "logs_semillas"


def construir_cola_evaluacion(csv_path: str, num_muestras: int) -> list[dict]:
    """Construye la cola de tareas de evaluación, idéntica para todas las semillas.

    Args:
        csv_path: Ruta del dataset de benchmark (`dataset_pacca.csv`).
        num_muestras: Cantidad de tareas a muestrear para la carga de trabajo.

    Returns:
        list[dict]: Tareas con su vector de observación y métricas CPU/GPU.
    """
    env = PlanificadorEnv(csv_path=csv_path, tamano_lote=num_muestras, shuffle=True)
    # Semilla fija: la carga de trabajo evaluada NO debe variar entre corridas.
    env.reset(seed=BENCH_SEED)
    return list(env.cola_tareas)


def evaluar_modelo(modelo_path: str, cola: list[dict]) -> dict[str, float]:
    """Simula la política determinista de un agente entrenado sobre la cola.

    Replica la contabilidad de `evaluar_agente.py`: CPU y GPU procesan sus colas
    en paralelo, por lo que el tiempo de sistema es el makespan (el dispositivo
    que termina último) y el EDP se calcula como Energía_total * Tiempo_total,
    ya que el EDP no es aditivo por tarea.

    Args:
        modelo_path: Ruta del modelo entrenado (con o sin extensión .zip).
        cola: Cola de tareas de evaluación.

    Returns:
        dict[str, float]: Métricas agregadas ('t', 'e', 'edp', 'precision',
            'frac_gpu') de esta corrida.
    """
    modelo = DQN.load(modelo_path)

    libre_cpu, libre_gpu = 0.0, 0.0
    energia_total = 0.0
    aciertos = 0
    asignaciones_gpu = 0

    for tarea in cola:
        accion, _ = modelo.predict(tarea["obs"], deterministic=True)
        accion_int = int(np.asarray(accion).item())

        m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
        # El óptimo verificable es el dispositivo de menor EDP (misma convención
        # de desempate que PlanificadorEnv.step: empate favorece a la CPU).
        accion_optima = 0 if float(m_cpu["edp"]) <= float(m_gpu["edp"]) else 1
        if accion_int == accion_optima:
            aciertos += 1

        m_elegida = tarea["metricas"][accion_int]
        if accion_int == 0:
            libre_cpu += m_elegida["tiempo"]
        else:
            libre_gpu += m_elegida["tiempo"]
            asignaciones_gpu += 1
        energia_total += m_elegida["energia"]

    tiempo_total = max(libre_cpu, libre_gpu)
    return {
        "t": tiempo_total,
        "e": energia_total,
        "edp": energia_total * tiempo_total,
        "precision": 100.0 * aciertos / len(cola),
        "frac_gpu": 100.0 * asignaciones_gpu / len(cola),
    }


def calcular_referencias(cola: list[dict]) -> dict[str, dict[str, float]]:
    """Calcula las líneas base deterministas Solo CPU y Solo GPU.

    Sirven como referencia invariante en las gráficas: al no depender de ningún
    entrenamiento, no tienen varianza entre semillas.

    Args:
        cola: Cola de tareas de evaluación.

    Returns:
        dict[str, dict[str, float]]: Métricas ('t', 'e', 'edp') por línea base.
    """
    referencias = {
        "Solo CPU": {"t": 0.0, "e": 0.0, "edp": 0.0},
        "Solo GPU": {"t": 0.0, "e": 0.0, "edp": 0.0},
    }
    for tarea in cola:
        m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
        referencias["Solo CPU"]["t"] += m_cpu["tiempo"]
        referencias["Solo CPU"]["e"] += m_cpu["energia"]
        referencias["Solo GPU"]["t"] += m_gpu["tiempo"]
        referencias["Solo GPU"]["e"] += m_gpu["energia"]

    for vals in referencias.values():
        vals["edp"] = vals["e"] * vals["t"]
    return referencias


def entrenar_barrido(
    semillas: list[int], timesteps: int, base_dir: str, reusar: bool, dataset: str | None = None
) -> dict[int, str]:
    """Entrena un agente independiente por cada semilla.

    Args:
        semillas: Semillas del DQN a evaluar.
        timesteps: Pasos de entrenamiento por corrida.
        base_dir: Directorio raíz del módulo Agente RL.
        reusar: Si es True, omite el entrenamiento de las semillas cuyo modelo
            .zip ya exista en disco (útil para re-generar solo las gráficas).
        dataset: Ruta al CSV codificado a usar para entrenar TODAS las semillas.
            Debe ser el mismo que luego se pase a construir_cola_evaluacion(), o
            la comparación entre semillas queda evaluada sobre una distribución
            distinta a la de entrenamiento.

    Returns:
        dict[int, str]: Semilla -> ruta del modelo entrenado (sin extensión).
    """
    os.makedirs(os.path.join(base_dir, SUBDIR_MODELOS), exist_ok=True)
    modelos: dict[int, str] = {}

    for i, semilla in enumerate(semillas, start=1):
        nombre = os.path.join(SUBDIR_MODELOS, f"modelo_dqn_seed_{semilla}")
        destino = os.path.join(base_dir, f"{nombre}.zip")

        if reusar and os.path.exists(destino):
            print(f"[{i}/{len(semillas)}] Semilla {semilla}: modelo existente, se reutiliza.")
            modelos[semilla] = destino
            continue

        print(f"\n[{i}/{len(semillas)}] Entrenando corrida independiente con semilla {semilla}...")
        try:
            modelos[semilla] = entrenar_agente(
                modelo_nombre=nombre,
                log_subdir=os.path.join(SUBDIR_LOGS, f"seed_{semilla}"),
                seed=semilla,
                timesteps=timesteps,
                dataset=dataset,
            )
        except Exception as exc:  # noqa: BLE001 - una corrida fallida no debe abortar el barrido
            print(f"[ERROR] La corrida con semilla {semilla} falló y se omite: {exc}")

    return modelos


def exportar_csv(
    resultados: dict[int, dict[str, float]],
    referencias: dict[str, dict[str, float]],
    csv_salida: str,
) -> None:
    """Persiste las métricas por semilla junto con la media y la desviación estándar.

    Args:
        resultados: Semilla -> métricas de esa corrida.
        referencias: Líneas base deterministas Solo CPU / Solo GPU.
        csv_salida: Ruta del CSV de salida.
    """
    campos = ["t", "e", "edp", "precision", "frac_gpu"]
    matriz = {c: np.array([resultados[s][c] for s in sorted(resultados)]) for c in campos}
    ref_edp = referencias["Solo CPU"]["edp"]

    with open(csv_salida, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Corrida",
                "Semilla DQN",
                "Tiempo Total (s)",
                "Energia Total (J)",
                "EDP Acumulado",
                "Precision (%)",
                "Tareas en GPU (%)",
                "Ahorro EDP vs Solo CPU (%)",
            ]
        )
        for i, semilla in enumerate(sorted(resultados), start=1):
            v = resultados[semilla]
            ahorro = (1 - v["edp"] / ref_edp) * 100 if ref_edp > 0 else 0.0
            writer.writerow(
                [
                    f"Run {i}",
                    semilla,
                    f"{v['t']:.4f}",
                    f"{v['e']:.4f}",
                    f"{v['edp']:.4f}",
                    f"{v['precision']:.2f}",
                    f"{v['frac_gpu']:.2f}",
                    f"{ahorro:.2f}",
                ]
            )

        ahorros = np.array(
            [(1 - resultados[s]["edp"] / ref_edp) * 100 if ref_edp > 0 else 0.0 for s in sorted(resultados)]
        )
        for etiqueta, func in (("Media", np.mean), ("Desv. estandar", np.std)):
            writer.writerow(
                [
                    etiqueta,
                    "-",
                    f"{func(matriz['t']):.4f}",
                    f"{func(matriz['e']):.4f}",
                    f"{func(matriz['edp']):.4f}",
                    f"{func(matriz['precision']):.2f}",
                    f"{func(matriz['frac_gpu']):.2f}",
                    f"{func(ahorros):.2f}",
                ]
            )

        for nombre, vals in referencias.items():
            writer.writerow(
                [nombre, "n/a (determinista)", f"{vals['t']:.4f}", f"{vals['e']:.4f}", f"{vals['edp']:.4f}", "-", "-",
                 f"{(1 - vals['edp'] / ref_edp) * 100 if ref_edp > 0 else 0.0:.2f}"]
            )

    print(f"[ÉXITO] Tabla por semilla generada en: {csv_salida}")


def _dibujar_panel_dispersion(
    ax: plt.Axes,
    etiquetas: list[str],
    valores: np.ndarray,
    formato: str,
    referencia: tuple[str, float, str] | None,
) -> None:
    """Dibuja un panel de dispersión por semilla con su banda media ± 1σ.

    Se usan marcadores en lugar de barras y el eje Y va ampliado al rango de los
    datos: la dispersión entre corridas es de orden 0.01%, así que un eje anclado
    en cero la volvería invisible. Los marcadores no arrastran la lectura de
    "área proporcional" que sí tendría una barra con base recortada.

    Args:
        ax: Eje de matplotlib sobre el que dibujar.
        etiquetas: Etiquetas del eje X (semillas).
        valores: Valor medido en cada corrida, en el orden de `etiquetas`.
        formato: Formato de texto para los valores (`str.format`).
        referencia: Tupla (nombre, valor, color) de una línea base determinista a
            superponer, o None si no aplica al panel.
    """
    media, desviacion = float(np.mean(valores)), float(np.std(valores))
    posiciones = np.arange(len(etiquetas))

    ax.axhspan(media - desviacion, media + desviacion, color=COLOR_TEXTO_PRIMARIO, alpha=0.10, zorder=1,
               label=f"Media ± 1σ (σ = {formato.format(desviacion)})")
    ax.axhline(media, color=COLOR_TEXTO_PRIMARIO, linewidth=1.6, zorder=3,
               label=f"Media = {formato.format(media)}")
    ax.scatter(posiciones, valores, s=140, color=COLOR_ACENTO_RL, edgecolor=COLOR_SUPERFICIE, linewidth=1.5,
               zorder=5, label="Corrida independiente")

    for x, valor in zip(posiciones, valores):
        # Etiqueta alternando arriba/abajo del marcador para que no choque con
        # la línea de la media cuando los valores casi coinciden.
        desplazamiento = 12 if valor >= media else -20
        ax.annotate(formato.format(valor), xy=(x, valor), xytext=(0, desplazamiento), textcoords="offset points",
                    ha="center", fontsize=9, color=COLOR_TEXTO_SECUNDARIO)

    valores_rango = [float(np.min(valores)), float(np.max(valores)), media - desviacion, media + desviacion]
    if referencia is not None:
        nombre, valor_ref, color = referencia
        valores_rango.append(valor_ref)
        ax.axhline(valor_ref, color=color, linewidth=1.8, linestyle="--", zorder=4, label=f"{nombre} = {valor_ref:.0f}")

    lo, hi = min(valores_rango), max(valores_rango)
    margen = (hi - lo) * 0.45 if hi > lo else max(abs(hi) * 0.001, 1.0)
    ax.set_ylim(lo - margen, hi + margen)
    ax.set_xticks(posiciones, etiquetas)
    ax.set_xlim(-0.6, len(etiquetas) - 0.4)
    limpiar_bordes(ax)


def graficar_dispersion(
    resultados: dict[int, dict[str, float]],
    referencias: dict[str, dict[str, float]],
    img_salida: str,
) -> None:
    """Genera la figura de dispersión entre corridas (EDP y precisión por semilla).

    Cada marcador es una corrida independiente y la banda sombreada es media ± 1σ.
    Una banda estrecha indica que el resultado del agente es reproducible y no
    producto de una semilla afortunada.

    Args:
        resultados: Semilla -> métricas de esa corrida.
        referencias: Líneas base deterministas Solo CPU / Solo GPU.
        img_salida: Ruta del PNG de salida.
    """
    semillas = sorted(resultados)
    etiquetas = [str(s) for s in semillas]
    edps = np.array([resultados[s]["edp"] for s in semillas])
    precisiones = np.array([resultados[s]["precision"] for s in semillas])

    configurar_estilo_pastel()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        f"Estabilidad del Agente RL entre {len(semillas)} Corridas Independientes",
        fontsize=17,
        fontweight="bold",
        color=COLOR_TEXTO_PRIMARIO,
    )

    _dibujar_panel_dispersion(
        axes[0], etiquetas, edps, "{:.0f}",
        referencia=("Solo GPU", referencias["Solo GPU"]["edp"], COLOR_GPU),
    )
    axes[0].set_title("Producto Energía-Retardo (EDP) del Sistema", fontsize=12)
    axes[0].set_ylabel("Magnitud EDP (eje ampliado)", fontsize=11)
    # Solo CPU queda un orden de magnitud por encima: se cita como texto en vez de
    # forzar la escala del eje, que aplastaría la dispersión que la figura reporta.
    axes[0].annotate(
        f"Solo CPU = {referencias['Solo CPU']['edp']:.0f} (fuera de escala)",
        xy=(0.02, 0.96), xycoords="axes fraction", ha="left", va="top",
        fontsize=9, fontweight="bold", color=COLOR_CPU,
    )

    _dibujar_panel_dispersion(axes[1], etiquetas, precisiones, "{:.1f}%", referencia=None)
    axes[1].set_title("Precisión de la Política Determinista", fontsize=12)
    axes[1].set_ylabel("% de decisiones óptimas (eje ampliado)", fontsize=11)

    for ax, ubicacion_leyenda in zip(axes, ("center right", "lower right")):
        ax.set_xlabel("Semilla del DQN", fontsize=11)
        ax.legend(loc=ubicacion_leyenda, fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(img_salida, dpi=300)
    print(f"[ÉXITO] Gráfica de dispersión guardada en: {img_salida}")


def graficar_convergencia_multisemilla(base_dir: str, semillas: list[int], img_salida: str) -> None:
    """Superpone las curvas de convergencia de todas las corridas con su banda ±1σ.

    Lee la etiqueta `metricas_personalizadas/precision` que `PrecisionCallback`
    registra en TensorBoard durante el entrenamiento de cada semilla.

    Args:
        base_dir: Directorio raíz del módulo Agente RL.
        semillas: Semillas cuyas curvas se van a graficar.
        img_salida: Ruta del PNG de salida.
    """
    series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for semilla in semillas:
        run_base = os.path.join(base_dir, "logs_entrenamiento", SUBDIR_LOGS, f"seed_{semilla}")
        try:
            # SB3 crea un subdirectorio DQN_<n> por invocación dentro del log_dir.
            candidatos = [d for d in os.listdir(run_base) if d.startswith("DQN_")]
            run_dir = os.path.join(run_base, max(candidatos, key=lambda d: os.path.getmtime(os.path.join(run_base, d))))
            series[semilla] = cargar_precision(run_dir)
        except (OSError, KeyError, ValueError) as exc:
            print(f"[AVISO] Sin curva de convergencia para la semilla {semilla}: {exc}")

    if not series:
        print("[AVISO] No se encontró ninguna curva de convergencia; se omite la figura.")
        return

    configurar_estilo_pastel()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    for i, (semilla, (steps, precision_pct)) in enumerate(sorted(series.items())):
        ax.plot(
            steps,
            suavizar_ema(precision_pct),
            color=PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)],
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
            label=f"Semilla {semilla}",
        )

    # Banda media ± 1σ sobre una malla común de steps (las corridas no comparten
    # exactamente los mismos puntos de registro, por eso se interpola).
    inicio = max(float(s[0]) for s, _ in series.values())
    fin = min(float(s[-1]) for s, _ in series.values())
    if fin > inicio:
        malla = np.linspace(inicio, fin, 300)
        interpoladas = np.vstack([np.interp(malla, steps, suavizar_ema(vals)) for steps, vals in series.values()])
        media, desviacion = interpoladas.mean(axis=0), interpoladas.std(axis=0)
        ax.plot(malla, media, color=COLOR_TEXTO_PRIMARIO, linewidth=2.8, zorder=4, label="Media entre semillas")
        ax.fill_between(malla, media - desviacion, media + desviacion, color=COLOR_TEXTO_PRIMARIO, alpha=0.12, zorder=2, label="± 1 desv. estándar")
        ax.annotate(
            f"{media[-1]:.1f}%",
            xy=(malla[-1], media[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color=COLOR_TEXTO_PRIMARIO,
        )

    ax.set_title(
        "Convergencia del Agente RL entre Corridas Independientes",
        fontsize=14,
        fontweight="bold",
        color=COLOR_TEXTO_PRIMARIO,
    )
    ax.set_xlabel("Steps de entrenamiento", fontsize=12)
    ax.set_ylabel("Precisión (% de decisiones óptimas, EMA)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1000:.0f}k")
    limpiar_bordes(ax)
    ax.legend(loc="lower right", ncol=2)

    plt.tight_layout()
    plt.savefig(img_salida, dpi=300)
    print(f"[ÉXITO] Gráfica de convergencia multi-semilla guardada en: {img_salida}")


def ejecutar_barrido(
    semillas: list[int], timesteps: int, num_muestras: int, reusar: bool, dataset: str | None = None
) -> None:
    """Orquesta el barrido completo: entrenamiento, evaluación, CSV y gráficas.

    Args:
        semillas: Semillas del DQN de las corridas independientes.
        timesteps: Pasos de entrenamiento por corrida.
        num_muestras: Tamaño de la carga de trabajo de evaluación.
        reusar: Reutiliza modelos ya entrenados en disco en lugar de re-entrenar.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = dataset if dataset else os.path.join(base_dir, "Entorno", "dataset_pacca.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")

    modelos = entrenar_barrido(semillas, timesteps, base_dir, reusar, dataset=csv_path)
    if not modelos:
        print("[ERROR] Ninguna corrida produjo un modelo; se aborta el reporte.")
        return

    print(f"\nEvaluando {len(modelos)} corridas sobre una carga idéntica de {num_muestras} tareas...")
    cola = construir_cola_evaluacion(csv_path, num_muestras)
    referencias = calcular_referencias(cola)

    resultados: dict[int, dict[str, float]] = {}
    for semilla, modelo_path in sorted(modelos.items()):
        try:
            resultados[semilla] = evaluar_modelo(modelo_path, cola)
            v = resultados[semilla]
            print(f"  Semilla {semilla}: EDP={v['edp']:.2f}  precisión={v['precision']:.2f}%  GPU={v['frac_gpu']:.1f}%")
        except Exception as exc:  # noqa: BLE001 - una evaluación fallida no debe abortar el reporte
            print(f"[ERROR] No se pudo evaluar la semilla {semilla}: {exc}")

    if not resultados:
        print("[ERROR] Ninguna corrida pudo evaluarse; se aborta el reporte.")
        return

    exportar_csv(resultados, referencias, os.path.join(base_dir, "resultados_semillas.csv"))
    graficar_dispersion(resultados, referencias, os.path.join(base_dir, "grafica_dispersion_semillas.png"))
    graficar_convergencia_multisemilla(base_dir, sorted(resultados), os.path.join(base_dir, "convergencia_multisemilla.png"))

    edps = np.array([v["edp"] for v in resultados.values()])
    precisiones = np.array([v["precision"] for v in resultados.values()])
    print(
        f"\n--- Resumen del barrido ({len(resultados)} corridas) ---\n"
        f"EDP:       media {edps.mean():.2f}  ±{edps.std():.2f} (CV {100 * edps.std() / edps.mean():.2f}%)\n"
        f"Precisión: media {precisiones.mean():.2f}%  ±{precisiones.std():.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barrido de semillas para el agente DQN.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Pasos de entrenamiento por corrida")
    parser.add_argument("-n", "--muestras", type=int, default=200, help="Tareas de la carga de evaluación")
    parser.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS, help="Semillas del DQN a entrenar")
    parser.add_argument("--reusar", action="store_true", help="Reutiliza modelos ya entrenados (solo re-genera reporte)")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Ruta al CSV codificado a usar para entrenar Y evaluar TODAS las semillas "
             "(debe ser el mismo en ambos pasos). Por defecto: Entorno/dataset_pacca.csv.",
    )
    args = parser.parse_args()

    ejecutar_barrido(args.semillas, args.timesteps, args.muestras, args.reusar, dataset=args.dataset)
