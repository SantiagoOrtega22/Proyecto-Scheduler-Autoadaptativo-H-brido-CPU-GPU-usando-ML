"""
evaluar_fase4_estadistica.py

Comparación estadística de la Fase 4 (OE4, revisión P2): EDP de sistema del agente
DQN frente a las seis políticas de referencia sobre 100 colas de tareas.

Diseño (fijado antes de ver los resultados)
--------------------------------------------
- Colas: 100 colas de 200 tareas, semillas 0 a 99, muestreo sin reemplazo dentro de
  cada cola (PlanificadorEnv.reset baraja los índices y toma los primeros 200).
- Fuente de tareas ('reservados', por defecto): tamaños no observados de la prueba de
  interpolación (P3). La cola de semilla s se arma con las tareas reservadas de la
  partición k = s mod 5, de modo que cada partición aporta 20 colas, y se evalúa con
  los 5 agentes entrenados sin esa partición (evaluar_interpolacion.py). El EDP del
  agente en la cola es la media de esos 5 agentes, igual que en evaluar_interpolacion.py.
  La fuente 'completo' (análisis de sensibilidad) toma las colas del conjunto completo
  y usa el modelo del libro (modelo_dqn_scheduler.zip).
- Métrica por cola: d = 100 * (EDP_sis(política) / EDP_sis(agente) - 1). d > 0 indica
  que el agente obtiene menor EDP de sistema que la política.
- Prueba: Wilcoxon de rangos con signo pareada por cola (H0: mediana de d = 0) para
  cada una de las 6 comparaciones (Solo CPU, Solo GPU, MET, MCT, Min-Min, Max-Min),
  aproximación normal con corrección por empates, sin corrección de continuidad, y
  corrección de Holm sobre las 6. Tamaño del efecto r = |Z| / sqrt(n).
- Umbral de relevancia práctica: |mediana de d| >= 1 % del EDP de sistema. Una
  diferencia significativa por debajo del umbral se reporta como no relevante.
- El oráculo por tarea se reporta solo de forma descriptiva (no es una política
  implementable, no entra en las 6 comparaciones).

Salidas (en resultados_fase4/<fuente>/)
---------------------------------------
- colas.csv: una fila por (cola, política) con T, E y EDP de sistema y d.
- resumen_estadistico.json: mediana, IQR, Z, p, p Holm, r y decisión por política.
- tabla_fase4_wilcoxon.tex: tabla del libro.
- fase4_robustez.png: distribución de d por política.

Uso
---
    python evaluar_fase4_estadistica.py                    # colas de tamaños reservados (P3)
    python evaluar_fase4_estadistica.py --fuente completo  # sensibilidad: conjunto completo
"""

import argparse
import csv
import json
import os
import sys
from typing import Callable

import numpy as np
from scipy.stats import norm, rankdata

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv  # noqa: E402
from evaluar_interpolacion import N_PARTICIONES, SEMILLAS, TAMANO_COLA, evaluar_cola  # noqa: E402

# Semillas de muestreo de las colas (una por cola). Son distintas de BENCH_SEED a
# propósito: cada cola debe ser una muestra independiente de la carga de trabajo.
# BENCH_SEED sigue siendo la única semilla de generación de datos y de entrenamiento.
SEMILLAS_COLAS: list[int] = list(range(100))

# Umbral de relevancia práctica, fijado antes de calcular las pruebas (revisión P2).
UMBRAL_RELEVANCIA_PCT = 1.0
ALFA = 0.05

AGENTE = "Agente DQN"
POLITICAS_COMPARADAS: list[str] = ["Solo CPU", "Solo GPU", "MET", "MCT", "Min-Min", "Max-Min"]
ORACULO = "Oráculo por tarea"

Predictor = Callable[[np.ndarray], np.ndarray]


# =============================================================================
# Agentes
# =============================================================================

def cargar_predictor(ruta_modelo: str) -> Predictor:
    """Carga un modelo DQN y devuelve su política determinista vectorizada.

    Args:
        ruta_modelo: Ruta del .zip de Stable-Baselines3.

    Returns:
        Predictor: Función que mapea observaciones (n x 23) a acciones (n,).

    Raises:
        FileNotFoundError: Si el modelo no existe.
    """
    from stable_baselines3 import DQN

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró {ruta_modelo}. Para la fuente 'reservados' corre antes "
            "evaluar_interpolacion.py (entrena los modelos de modelos_interpolacion/)."
        )
    modelo = DQN.load(ruta_modelo)

    def predecir(X: np.ndarray) -> np.ndarray:
        acciones, _ = modelo.predict(X.astype(np.float32), deterministic=True)
        return np.asarray(acciones, dtype=int).reshape(-1)

    return predecir


# =============================================================================
# Generación y evaluación de colas
# =============================================================================

def evaluar_colas(csv_path: str, fuente: str, base_dir: str) -> list[dict]:
    """Evalúa las 100 colas y devuelve una fila por (cola, política).

    Args:
        csv_path: Dataset codificado (el mismo con el que se entrenaron los modelos).
        fuente: 'reservados' (tamaños no observados, P3) o 'completo'.
        base_dir: Carpeta de Agente RL (para ubicar los modelos).

    Returns:
        list[dict]: Filas con cola, semilla, particion, politica, T_sis, E_sis, EDP_sis.
    """
    if fuente == "reservados":
        entornos = {
            k: PlanificadorEnv(csv_path=csv_path, tamano_lote=TAMANO_COLA, shuffle=True,
                               modo_particion="tamano", particion=k,
                               n_particiones=N_PARTICIONES, split="holdout")
            for k in range(N_PARTICIONES)
        }
        # Los agentes de la partición k nunca vieron sus tamaños durante el entrenamiento.
        agentes = {
            k: {f"{AGENTE} (s{s})": cargar_predictor(
                os.path.join(base_dir, "modelos_interpolacion", f"dqn_p{k}_s{s}.zip"))
                for s in SEMILLAS}
            for k in range(N_PARTICIONES)
        }
    else:
        entornos = {-1: PlanificadorEnv(csv_path=csv_path, tamano_lote=TAMANO_COLA, shuffle=True)}
        agentes = {-1: {f"{AGENTE} (libro)": cargar_predictor(
            os.path.join(base_dir, "modelo_dqn_scheduler.zip"))}}

    filas: list[dict] = []
    for cola_id, semilla in enumerate(SEMILLAS_COLAS):
        k = semilla % N_PARTICIONES if fuente == "reservados" else -1
        entornos[k].reset(seed=semilla)
        cola = list(entornos[k].cola_tareas)
        resultados = evaluar_cola(cola, agentes[k])

        edp_agentes = []
        for politica, (T, E, EDP, n_gpu) in resultados.items():
            filas.append({"cola": cola_id, "semilla": semilla, "particion": k, "politica": politica,
                          "T_sis": T, "E_sis": E, "EDP_sis": EDP, "tareas_gpu": n_gpu})
            if politica.startswith(AGENTE):
                edp_agentes.append(EDP)
        # Un único valor de agente por cola: media sobre los agentes evaluados.
        filas.append({"cola": cola_id, "semilla": semilla, "particion": k, "politica": AGENTE,
                      "T_sis": float("nan"), "E_sis": float("nan"),
                      "EDP_sis": float(np.mean(edp_agentes)), "tareas_gpu": float("nan")})
        if (cola_id + 1) % 20 == 0:
            print(f"  {cola_id + 1}/{len(SEMILLAS_COLAS)} colas evaluadas")

    # Diferencia relativa de cada fila frente al agente de su cola.
    edp_agente = {f["cola"]: f["EDP_sis"] for f in filas if f["politica"] == AGENTE}
    for f in filas:
        f["dif_vs_agente_pct"] = 100.0 * (f["EDP_sis"] / edp_agente[f["cola"]] - 1.0)
    return filas


def diferencias_por_politica(filas: list[dict], politica: str) -> np.ndarray:
    """Vector de d (una entrada por cola, en orden de cola) para una política."""
    sel = sorted((f for f in filas if f["politica"] == politica), key=lambda f: f["cola"])
    return np.array([f["dif_vs_agente_pct"] for f in sel])


# =============================================================================
# Estadística
# =============================================================================

def wilcoxon_signo_rango(d: np.ndarray) -> dict[str, float]:
    """Wilcoxon de rangos con signo de una muestra pareada (H0: mediana de d = 0).

    Se calcula explícitamente para reportar Z y r: se descartan las diferencias
    nulas (método de Wilcoxon), se promedian rangos en empates y se aplica la
    aproximación normal con corrección por empates, sin corrección de continuidad.
    Con n = 100 la aproximación normal es adecuada.

    Args:
        d: Diferencias pareadas por cola.

    Returns:
        dict: n (diferencias no nulas), W+ (suma de rangos positivos), Z, p
            bilateral y r = |Z| / sqrt(n).
    """
    d = d[d != 0.0]
    n = len(d)
    rangos = rankdata(np.abs(d))
    w_pos = float(rangos[d > 0].sum())
    media = n * (n + 1) / 4.0
    _, conteos = np.unique(rangos, return_counts=True)
    varianza = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(conteos ** 3 - conteos)) / 48.0
    z = (w_pos - media) / np.sqrt(varianza)
    return {"n": n, "W_pos": w_pos, "Z": float(z), "p": float(2.0 * norm.sf(abs(z))),
            "r": float(abs(z) / np.sqrt(n))}


def corregir_holm(p_valores: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (step-down) para comparaciones múltiples.

    Args:
        p_valores: p sin corregir, en cualquier orden.

    Returns:
        list[float]: p corregidos, en el mismo orden de entrada.
    """
    m = len(p_valores)
    orden = np.argsort(p_valores)
    corregidos = np.empty(m)
    acumulado = 0.0
    for paso, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - paso) * p_valores[i]))
        corregidos[i] = acumulado
    return corregidos.tolist()


def resumir(filas: list[dict], fuente: str, csv_path: str) -> dict:
    """Aplica las 6 pruebas, la corrección de Holm y el umbral de relevancia.

    Args:
        filas: Salida de evaluar_colas.
        fuente: Fuente de las colas (se guarda en el resumen).
        csv_path: Dataset usado (se guarda en el resumen).

    Returns:
        dict: Resumen con el diseño del experimento y una entrada por política.
    """
    resumen: dict = {
        "fuente": fuente, "dataset": os.path.abspath(csv_path),
        "n_colas": len(SEMILLAS_COLAS), "tareas_por_cola": TAMANO_COLA,
        "semillas_colas": [SEMILLAS_COLAS[0], SEMILLAS_COLAS[-1]],
        "agente": ("media de los 5 agentes entrenados sin la partición de la cola, semillas "
                   f"{SEMILLAS}") if fuente == "reservados" else "modelo_dqn_scheduler.zip",
        "metrica": "d = 100 * (EDP_sis(politica) / EDP_sis(agente) - 1); d > 0: el agente es mejor",
        "prueba": "Wilcoxon de rangos con signo pareada por cola, aproximación normal, Holm sobre 6 comparaciones",
        "umbral_relevancia_pct": UMBRAL_RELEVANCIA_PCT, "alfa": ALFA, "politicas": {},
    }
    pruebas = {pol: wilcoxon_signo_rango(diferencias_por_politica(filas, pol)) for pol in POLITICAS_COMPARADAS}
    p_holm = dict(zip(POLITICAS_COMPARADAS, corregir_holm([pruebas[p]["p"] for p in POLITICAS_COMPARADAS])))

    for pol in POLITICAS_COMPARADAS + [ORACULO]:
        d = diferencias_por_politica(filas, pol)
        q1, mediana, q3 = np.percentile(d, [25, 50, 75])
        entrada = {"mediana": float(mediana), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
                   "colas_agente_mejor": int(np.sum(d > 0)), "colas_politica_mejor": int(np.sum(d < 0))}
        if pol in pruebas:
            significativa = p_holm[pol] < ALFA
            relevante = abs(mediana) >= UMBRAL_RELEVANCIA_PCT
            entrada.update(pruebas[pol])
            entrada.update({
                "p_holm": p_holm[pol], "significativa": bool(significativa),
                "supera_umbral": bool(relevante),
                "conclusion": ("sin diferencia significativa" if not significativa
                               else "significativa, no relevante" if not relevante
                               else "agente mejor" if mediana > 0 else "política mejor"),
            })
        resumen["politicas"][pol] = entrada

    if fuente == "reservados":
        # Sensibilidad a la semilla del DQN: mediana de d usando cada agente por separado.
        resumen["sensibilidad_por_semilla_dqn"] = {}
        for s in SEMILLAS:
            edp_s = {f["cola"]: f["EDP_sis"] for f in filas if f["politica"] == f"{AGENTE} (s{s})"}
            resumen["sensibilidad_por_semilla_dqn"][str(s)] = {
                pol: float(np.median([100.0 * (f["EDP_sis"] / edp_s[f["cola"]] - 1.0)
                                      for f in filas if f["politica"] == pol]))
                for pol in POLITICAS_COMPARADAS
            }
    return resumen


# =============================================================================
# Exportación
# =============================================================================

def exportar_csv(filas: list[dict], ruta: str) -> None:
    """Escribe una fila por (cola, política)."""
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def _formato_p(p: float) -> str:
    """p con la notación del libro (punto decimal): '< 0.001' por debajo de 10^-3."""
    return "$< 0.001$" if p < 1e-3 else f"${p:.3f}$"


def _formato_num(x: float, dec: int = 2, signo: bool = True) -> str:
    """Número en modo matemático con punto decimal, como el resto del libro."""
    return f"${x:+.{dec}f}$" if signo else f"${x:.{dec}f}$"


def exportar_tabla_latex(resumen: dict, ruta: str) -> None:
    """Tabla del libro: política | mediana | IQR | p (Holm) | r | ¿supera el umbral?

    Se envuelve en \\resizebox{\\textwidth}, igual que las demás tablas anchas del libro,
    para que no exceda el ancho de página; los encabezados y la columna de decisión se
    mantienen cortos para que el escalado no reduzca demasiado la fuente.
    """
    origen = ("tamaños no observados" if resumen["fuente"] == "reservados"
              else "configuraciones medidas (conjunto completo)")
    umbral = f"{resumen['umbral_relevancia_pct']:g}"
    lineas = [
        "\\begin{table}[H]", "\\centering", "\\renewcommand{\\arraystretch}{1.3}",
        f"\\caption[Comparación estadística del EDP de sistema sobre {resumen['n_colas']} colas "
        f"({origen}).]{{Diferencia relativa de EDP de sistema de cada política respecto del agente sobre "
        f"{resumen['n_colas']} colas de {resumen['tareas_por_cola']} tareas de {origen}. "
        f"Valores positivos: el agente obtiene menor EDP. Prueba de Wilcoxon de rangos con signo "
        f"pareada por cola con corrección de Holm; $r = |Z|/\\sqrt{{n}}$; umbral de relevancia "
        f"práctica: {umbral}\\,\\%.}}",
        "\\label{tab:fase4_wilcoxon}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|r|c|c|c|c|}", "\\hline",
        "\\textbf{Política} & \\textbf{Mediana (\\%)} & \\textbf{IQR (\\%)} & "
        f"\\textbf{{$p$ (Holm)}} & \\textbf{{$r$}} & \\textbf{{¿Supera {umbral}\\,\\%?}} \\\\ \\hline",
    ]
    for pol in POLITICAS_COMPARADAS:
        v = resumen["politicas"][pol]
        decision = ("No" if not v["supera_umbral"] else
                    "Sí, agente mejor" if v["mediana"] > 0 else "Sí, política mejor")
        lineas.append(f"{pol} & {_formato_num(v['mediana'])} & "
                      f"[{_formato_num(v['q1'])}; {_formato_num(v['q3'])}] & "
                      f"{_formato_p(v['p_holm'])} & {_formato_num(v['r'], signo=False)} & {decision} \\\\ \\hline")
    o = resumen["politicas"][ORACULO]
    lineas.append(f"\\textit{{Oráculo (referencia)}} & {_formato_num(o['mediana'], 3)} & "
                  f"[{_formato_num(o['q1'], 3)}; {_formato_num(o['q3'], 3)}] & --- & --- & --- \\\\ \\hline")
    lineas += ["\\end{tabular}%", "}", "\\end{table}",
               "% FUENTE: evaluar_fase4_estadistica.py -> resumen_estadistico.json"]
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")


def graficar_robustez(filas: list[dict], resumen: dict, ruta: str) -> None:
    """Distribución por cola de d para cada política (una fila por política).

    Solo CPU va en un panel aparte con su propia escala: sus diferencias (~10^3 %)
    aplastarían a las demás (~1 %) en un eje común, y un eje logarítmico no admite
    las diferencias negativas.

    Args:
        filas: Salida de evaluar_colas.
        resumen: Salida de resumir (medianas y conclusiones).
        ruta: PNG de salida.
    """
    import matplotlib.pyplot as plt
    from estilo_graficas import (COLOR_EJE, COLOR_GRID, COLOR_TEXTO_PRIMARIO, COLOR_TEXTO_SECUNDARIO,
                                 PALETA_CATEGORICA, configurar_estilo_pastel, limpiar_bordes)

    configurar_estilo_pastel()
    color_puntos = PALETA_CATEGORICA[0]
    principales = [p for p in POLITICAS_COMPARADAS if p != "Solo CPU"]
    rng = np.random.default_rng(0)  # Solo para el jitter vertical de los puntos (estético).

    fig, (ax_cpu, ax) = plt.subplots(
        2, 1, figsize=(10, 6.2), dpi=300, gridspec_kw={"height_ratios": [1, len(principales)]})

    def dibujar(eje: plt.Axes, politicas: list[str]) -> None:
        for y, pol in enumerate(politicas):
            d = diferencias_por_politica(filas, pol)
            eje.boxplot(d, positions=[y], vert=False, widths=0.55, showfliers=False, patch_artist=True,
                        boxprops={"facecolor": "none", "edgecolor": COLOR_TEXTO_SECUNDARIO, "linewidth": 1.2},
                        medianprops={"color": COLOR_TEXTO_PRIMARIO, "linewidth": 2},
                        whiskerprops={"color": COLOR_TEXTO_SECUNDARIO, "linewidth": 1.2},
                        capprops={"color": COLOR_TEXTO_SECUNDARIO, "linewidth": 1.2}, zorder=3)
            eje.scatter(d, y + rng.uniform(-0.18, 0.18, len(d)), s=14, color=color_puntos,
                        alpha=0.55, edgecolors="none", zorder=2)
            v = resumen["politicas"][pol]
            etiqueta = f"mediana {v['mediana']:+.2f} %  ·  {v['conclusion']}".replace(".", ",")
            eje.annotate(etiqueta, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
                         textcoords="offset points", va="center", fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
        eje.set_yticks(range(len(politicas)), politicas)
        eje.set_ylim(len(politicas) - 0.5, -0.5)
        eje.grid(axis="y", visible=False)
        limpiar_bordes(eje)

    umbral = resumen["umbral_relevancia_pct"]
    ax.axvspan(-umbral, umbral, color=COLOR_GRID, alpha=0.55, zorder=0, linewidth=0)
    ax.axvline(0, color=COLOR_EJE, linewidth=1.2, zorder=1)
    dibujar(ax, principales)
    ax.set_xlabel("Diferencia de EDP de sistema respecto del agente (%)")
    ax.text(0, -0.62, f"±{umbral:g} % (umbral de relevancia)".replace(".", ","), ha="center",
            va="bottom", fontsize=8.5, color=COLOR_TEXTO_SECUNDARIO)

    dibujar(ax_cpu, ["Solo CPU"])
    ax_cpu.set_xlabel("%", fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
    ax_cpu.set_title("Solo CPU (escala propia)", loc="left", fontsize=10, fontweight="normal",
                     color=COLOR_TEXTO_SECUNDARIO)

    origen = "tamaños no observados" if resumen["fuente"] == "reservados" else "conjunto completo"
    fig.suptitle(f"EDP de sistema por cola: {resumen['n_colas']} colas de {resumen['tareas_por_cola']} "
                 f"tareas ({origen})", fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.04)
    fig.text(0.02, 0.995, "Cada punto es una cola. Valores > 0: el agente obtiene menor EDP que la política.",
             ha="left", fontsize=9.5, color=COLOR_TEXTO_SECUNDARIO)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def imprimir_tabla(resumen: dict) -> None:
    """Tabla de resultados en consola."""
    print(f"\n{'Política':20s} {'Mediana':>9s} {'Q1':>9s} {'Q3':>9s} {'p Holm':>10s} {'r':>6s}  Conclusión")
    for pol in POLITICAS_COMPARADAS:
        v = resumen["politicas"][pol]
        print(f"{pol:20s} {v['mediana']:+9.3f} {v['q1']:+9.3f} {v['q3']:+9.3f} {v['p_holm']:10.2e} "
              f"{v['r']:6.2f}  {v['conclusion']}")
    o = resumen["politicas"][ORACULO]
    print(f"{ORACULO:20s} {o['mediana']:+9.3f} {o['q1']:+9.3f} {o['q3']:+9.3f}  (referencia descriptiva)")


def ejecutar(csv_path: str, fuente: str, salida_dir: str) -> dict:
    """Corre la evaluación completa y escribe las salidas.

    Args:
        csv_path: Dataset codificado.
        fuente: 'reservados' o 'completo'.
        salida_dir: Carpeta de resultados.

    Returns:
        dict: Resumen estadístico.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(salida_dir, exist_ok=True)
    print(f"Evaluando {len(SEMILLAS_COLAS)} colas de {TAMANO_COLA} tareas (fuente: {fuente})...")
    filas = evaluar_colas(csv_path, fuente, base_dir)
    resumen = resumir(filas, fuente, csv_path)

    exportar_csv(filas, os.path.join(salida_dir, "colas.csv"))
    with open(os.path.join(salida_dir, "resumen_estadistico.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    exportar_tabla_latex(resumen, os.path.join(salida_dir, "tabla_fase4_wilcoxon.tex"))
    graficar_robustez(filas, resumen, os.path.join(salida_dir, "fase4_robustez.png"))
    imprimir_tabla(resumen)
    print(f"\nResultados en {salida_dir}")
    return resumen


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Comparación estadística de la Fase 4 (revisión P2).")
    parser.add_argument("--fuente", choices=["reservados", "completo"], default="reservados",
                        help="'reservados': tamaños no observados de P3 (default); 'completo': sensibilidad.")
    parser.add_argument("--dataset", default=os.path.join(base, "Entorno", "dataset_rl.csv"),
                        help="CSV codificado; debe ser el mismo con el que se entrenaron los modelos.")
    parser.add_argument("--salida", default=None, help="Default: resultados_fase4/<fuente>/")
    args = parser.parse_args()
    ejecutar(args.dataset, args.fuente, args.salida or os.path.join(base, "resultados_fase4", args.fuente))
