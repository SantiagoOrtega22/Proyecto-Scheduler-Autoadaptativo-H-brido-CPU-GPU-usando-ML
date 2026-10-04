"""
comparar_versiones_dataset.py

Compara dos versiones del agente DQN entrenadas con train.py (mismos
hiperparámetros, misma semilla) sobre dos versiones del dataset codificado:

- Actual: modelo_dqn_scheduler.zip, entrenado con Entorno/dataset_rl.csv
  (GEMM de benchmarkpacca_gemm.csv, energía por muestreo).
- Nuevo:  modelo_dqn_counter.zip, entrenado con Entorno/dataset_rl_counter.csv
  (GEMM de gemm_counter_full.csv, energía por contadores RAPL/NVML). El bloque
  FFT es idéntico en ambos datasets.

Secciones
---------
1. Convergencia: recompensa media de los últimos episodios (monitor.csv),
   expresada también como fracción de la recompensa del oráculo del dataset.
2. Decisión por tarea (metricas() de evaluar_interpolacion.py): precisión frente
   al oráculo, precisión por carga, % a GPU, sobrecosto geométrico de EDP y
   pérdida de recompensa. Se evalúa la matriz cruzada modelo x dataset: la
   diagonal es cada modelo sobre su propio dataset; el modelo actual sobre el
   dataset nuevo mide cuánto de su política sigue siendo válida con las nuevas
   mediciones de GEMM.
3. Cambio en los datos: para las configuraciones GEMM presentes en ambos
   datasets, cuántas cambian de dispositivo óptimo.
4. Fase 4: EDP de sistema en 100 colas de 200 tareas (semillas 0..99, misma
   lógica que evaluar_fase4_estadistica.py --fuente completo). Mediana de
   d = 100 * (EDP_sis(política) / EDP_sis(agente) - 1) por política.

Ambos modelos se entrenaron con el dataset completo (sin holdout), así que las
secciones 2 y 4 miden ajuste a la frontera de decisión, no generalización.

Uso
---
    python comparar_versiones_dataset.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv  # noqa: E402
from evaluar_fase4_estadistica import SEMILLAS_COLAS, cargar_predictor  # noqa: E402
from evaluar_interpolacion import CARGAS, TAMANO_COLA, cargar_arreglos, evaluar_cola, metricas  # noqa: E402

VENTANA_CONVERGENCIA = 50  # Episodios finales promediados de monitor.csv
POLITICAS_REFERENCIA: list[str] = ["Solo CPU", "Solo GPU", "MET", "MCT", "Min-Min", "Max-Min", "Oráculo por tarea"]


def recompensa_final(ruta_monitor: str, ventana: int = VENTANA_CONVERGENCIA) -> tuple[float, int]:
    """Recompensa media por paso en los últimos `ventana` episodios de monitor.csv.

    Args:
        ruta_monitor: monitor.csv de Stable-Baselines3 (columnas r, l, t).
        ventana: Episodios finales a promediar.

    Returns:
        tuple[float, int]: (recompensa media por paso, episodios totales).
    """
    with open(ruta_monitor, encoding="utf-8") as f:
        next(f)  # Línea de metadatos JSON del Monitor
        filas = list(csv.DictReader(f))
    por_paso = [float(r["r"]) / float(r["l"]) for r in filas]
    return float(np.mean(por_paso[-ventana:])), len(filas)


def recompensa_oraculo(d: dict) -> float:
    """Recompensa media por paso que obtendría el oráculo (techo del dataset)."""
    edp = d["edp"]
    peor = edp.max(axis=1)
    return float(np.mean(np.where(peor > 0, 1.0 - edp.min(axis=1) / np.where(peor > 0, peor, 1.0), 1.0)))


def cambio_oraculo_gemm(d_actual: dict, d_nuevo: dict) -> dict:
    """Configuraciones GEMM comunes a ambos datasets y cuántas cambian de óptimo.

    La configuración se identifica por el vector de observación completo, que en
    GEMM codifica M, N, K, precisión y transposiciones.
    """
    def indice(d: dict) -> dict[tuple, int]:
        sel = np.flatnonzero(d["carga"] == "GEMM")
        return {tuple(np.round(d["X"][i], 6)): int(d["oraculo"][i]) for i in sel}

    a, n = indice(d_actual), indice(d_nuevo)
    comunes = a.keys() & n.keys()
    cpu_a_gpu = sum(1 for k in comunes if a[k] == 0 and n[k] == 1)
    gpu_a_cpu = sum(1 for k in comunes if a[k] == 1 and n[k] == 0)
    return {
        "configs_actual": len(a),
        "configs_nuevo": len(n),
        "configs_comunes": len(comunes),
        "cambian_cpu_a_gpu": cpu_a_gpu,
        "cambian_gpu_a_cpu": gpu_a_cpu,
        "pct_cambian": 100.0 * (cpu_a_gpu + gpu_a_cpu) / len(comunes) if comunes else float("nan"),
    }


def fase4(csv_path: str, predictores: dict) -> dict[str, dict[str, float]]:
    """EDP de sistema en 100 colas y mediana de d de cada política frente a cada agente.

    Args:
        csv_path: Dataset del que se muestrean las colas.
        predictores: nombre -> función obs (n x 23) -> acciones (n,).

    Returns:
        dict: agente -> {política: mediana de d en %, ...}, más 'pct_gpu' del agente.
    """
    env = PlanificadorEnv(csv_path=csv_path, tamano_lote=TAMANO_COLA, shuffle=True)
    edp_por_politica: dict[str, list[float]] = {}
    gpu_por_agente: dict[str, list[int]] = {k: [] for k in predictores}
    for semilla in SEMILLAS_COLAS:
        env.reset(seed=semilla)
        resultados = evaluar_cola(list(env.cola_tareas), predictores)
        for politica, (_, _, edp, n_gpu) in resultados.items():
            edp_por_politica.setdefault(politica, []).append(edp)
            if politica in gpu_por_agente:
                gpu_por_agente[politica].append(n_gpu)

    salida = {}
    for agente in predictores:
        ref = np.array(edp_por_politica[agente])
        salida[agente] = {
            p: float(np.median(100.0 * (np.array(edp_por_politica[p]) / ref - 1.0)))
            for p in POLITICAS_REFERENCIA + [a for a in predictores if a != agente]
        }
        salida[agente]["pct_gpu"] = 100.0 * float(np.mean(gpu_por_agente[agente])) / TAMANO_COLA
    return salida


def imprimir_resumen(r: dict) -> None:
    """Tablas de consola con las cuatro secciones."""
    print("\n=== 1. Convergencia (últimos %d episodios) ===" % VENTANA_CONVERGENCIA)
    for nombre, c in r["convergencia"].items():
        print(f"  {nombre:<8} recompensa/paso = {c['recompensa_paso']:.4f}  "
              f"oráculo = {c['recompensa_oraculo']:.4f}  ({c['fraccion_oraculo']:.1%})  "
              f"episodios = {c['episodios']}")

    print("\n=== 2. Decisión por tarea (modelo -> dataset) ===")
    cols = ["precision"] + [f"precision_{c}" for c in CARGAS] + ["pct_gpu", "sobrecosto_geom", "perdida_recompensa"]
    print(f"  {'modelo -> dataset':<20}" + "".join(f"{c.replace('precision_', 'prec_').replace('sobrecosto_geom', 'sobrec_geom_%').replace('perdida_recompensa', 'perdida_R'):>14}" for c in cols))
    for clave, m in r["por_tarea"].items():
        print(f"  {clave:<20}" + "".join(f"{m[c]:>14.3f}" for c in cols))

    g = r["cambio_oraculo_gemm"]
    print("\n=== 3. Cambio del óptimo en GEMM (configuraciones comunes) ===")
    print(f"  comunes = {g['configs_comunes']} (actual {g['configs_actual']}, nuevo {g['configs_nuevo']})  "
          f"CPU->GPU = {g['cambian_cpu_a_gpu']}  GPU->CPU = {g['cambian_gpu_a_cpu']}  "
          f"({g['pct_cambian']:.1f} %)")

    print("\n=== 4. Fase 4: mediana de d = EDP_sis(política)/EDP_sis(agente) - 1 [%] (100 colas) ===")
    for dataset, por_agente in r["fase4"].items():
        print(f"  Colas de {dataset}:")
        for agente, d in por_agente.items():
            otros = "  ".join(f"{p}={v:+.2f}" for p, v in d.items() if p != "pct_gpu")
            print(f"    {agente:<16} (%GPU {d['pct_gpu']:.1f})  {otros}")


def ejecutar(args: argparse.Namespace) -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    versiones = {
        "actual": {"modelo": args.modelo_actual, "dataset": args.dataset_actual, "monitor": args.monitor_actual},
        "nuevo": {"modelo": args.modelo_nuevo, "dataset": args.dataset_nuevo, "monitor": args.monitor_nuevo},
    }
    datos = {v: cargar_arreglos(cfg["dataset"]) for v, cfg in versiones.items()}
    predictores = {v: cargar_predictor(cfg["modelo"]) for v, cfg in versiones.items()}

    resumen: dict = {"entradas": versiones, "convergencia": {}, "por_tarea": {}}

    for v, cfg in versiones.items():
        r_paso, episodios = recompensa_final(cfg["monitor"])
        r_opt = recompensa_oraculo(datos[v])
        resumen["convergencia"][v] = {"recompensa_paso": r_paso, "recompensa_oraculo": r_opt,
                                      "fraccion_oraculo": r_paso / r_opt, "episodios": episodios}

    # Matriz cruzada: cada modelo sobre cada dataset (todas las tareas).
    for v_modelo in versiones:
        for v_datos, d in datos.items():
            acciones = predictores[v_modelo](d["X"])
            mascara = np.ones(len(acciones), dtype=bool)
            resumen["por_tarea"][f"{v_modelo} -> {v_datos}"] = metricas(acciones, d, mascara)

    resumen["cambio_oraculo_gemm"] = cambio_oraculo_gemm(datos["actual"], datos["nuevo"])

    # Fase 4 con cada modelo en su propio dataset, y ambos modelos sobre las colas
    # del dataset nuevo (comparación directa con las mediciones vigentes).
    nombres = {"actual": "DQN actual", "nuevo": "DQN counter"}
    resumen["fase4"] = {
        "dataset_rl.csv": fase4(versiones["actual"]["dataset"], {nombres["actual"]: predictores["actual"]}),
        "dataset_rl_counter.csv": fase4(versiones["nuevo"]["dataset"],
                                        {nombres[v]: predictores[v] for v in versiones}),
    }

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.salida, "por_tarea.csv"), "w", newline="", encoding="utf-8") as f:
        campos = ["modelo_dataset"] + list(next(iter(resumen["por_tarea"].values())).keys())
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for clave, m in resumen["por_tarea"].items():
            w.writerow({"modelo_dataset": clave, **m})

    imprimir_resumen(resumen)
    print(f"\nResultados en {args.salida}")
    return resumen


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Compara el DQN actual contra el entrenado con dataset_rl_counter.csv.")
    p.add_argument("--modelo-actual", default=os.path.join(base, "modelo_dqn_scheduler.zip"))
    p.add_argument("--dataset-actual", default=os.path.join(base, "Entorno", "dataset_rl.csv"))
    p.add_argument("--monitor-actual", default=os.path.join(base, "logs_entrenamiento", "monitor.csv"))
    p.add_argument("--modelo-nuevo", default=os.path.join(base, "modelo_dqn_counter.zip"))
    p.add_argument("--dataset-nuevo", default=os.path.join(base, "Entorno", "dataset_rl_counter.csv"))
    p.add_argument("--monitor-nuevo", default=os.path.join(base, "logs_entrenamiento", "counter", "monitor.csv"))
    p.add_argument("--salida", default=os.path.join(base, "resultados_comparacion_counter"))
    ejecutar(p.parse_args())
