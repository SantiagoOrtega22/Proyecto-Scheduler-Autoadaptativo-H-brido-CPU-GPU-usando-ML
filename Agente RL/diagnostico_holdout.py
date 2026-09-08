"""
diagnostico_holdout.py

Diagnostico de memorizacion vs. interpolacion para el Agente RL (DQN).

Entrena una copia del agente reservando un 10% del dataset fuera del
entrenamiento (holdout), y compara la precision (% de decisiones optimas,
verificables directamente contra argmin(edp_cpu, edp_gpu)) entre el split de
entrenamiento y el holdout nunca visto.

No reemplaza el modelo de tesis (modelo_dqn_scheduler.zip): guarda un modelo
separado (modelo_dqn_scheduler_holdout_check.zip) exclusivamente para esta
comprobacion. El objetivo NO es medir generalizacion a tamaños N fuera del
alcance del proyecto (ver CLAUDE.md #3), sino verificar si la red aprendio
una frontera de decision en funcion de las features, o si memorizo filas
exactas del CSV.
"""

import os
import sys

import numpy as np
from stable_baselines3 import DQN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv
from train import entrenar_agente, BENCH_SEED

HOLDOUT_FRACTION = 0.10


def medir_precision_split(
    modelo: DQN, csv_path: str, split: str, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[float, int]:
    """Mide el % de decisiones optimas del modelo sobre un split fijo del dataset.

    Recorre cada tarea del split exactamente una vez (sin muestreo aleatorio ni
    reemplazo) y compara la accion predicha contra el optimo verificable
    (argmin(edp_cpu, edp_gpu)).

    Args:
        modelo: Modelo DQN entrenado a evaluar.
        csv_path: Ruta al dataset_pacca.csv.
        split: 'train' o 'holdout', debe coincidir con el split usado al entrenar.
        holdout_fraction: Debe coincidir con la fraccion usada al entrenar el modelo.

    Returns:
        tuple[float, int]: (precision en [0.0, 1.0], numero de tareas evaluadas).
    """
    env = PlanificadorEnv(
        csv_path=csv_path,
        tamano_lote=10_000_000,
        shuffle=False,
        holdout_fraction=holdout_fraction,
        split=split,
        split_seed=BENCH_SEED,
    )
    env.reset(seed=BENCH_SEED)

    total = len(env.cola_tareas)
    aciertos = 0
    for tarea in env.cola_tareas:
        edp_cpu = tarea["metricas"][0]["edp"]
        edp_gpu = tarea["metricas"][1]["edp"]
        accion_optima = 0 if edp_cpu <= edp_gpu else 1

        accion, _ = modelo.predict(tarea["obs"], deterministic=True)
        accion_predicha = int(np.asarray(accion).item())

        if accion_predicha == accion_optima:
            aciertos += 1

    return (aciertos / total if total > 0 else 0.0), total


def ejecutar_diagnostico() -> None:
    """Entrena el modelo de holdout y reporta la precision train vs. holdout."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Entorno", "dataset_pacca.csv")

    print(f"--- Entrenando modelo de diagnostico (holdout={HOLDOUT_FRACTION:.0%}) ---")
    modelo_path = entrenar_agente(
        holdout_fraction=HOLDOUT_FRACTION,
        split="train",
        modelo_nombre="modelo_dqn_scheduler_holdout_check",
        log_subdir="holdout_check",
    )

    print("\n--- Midiendo precision por split ---")
    modelo = DQN.load(modelo_path)

    precision_train, n_train = medir_precision_split(modelo, csv_path, split="train")
    precision_holdout, n_holdout = medir_precision_split(modelo, csv_path, split="holdout")

    print(f"\nPrecision en TRAIN   ({n_train} tareas, vistas en entrenamiento): {precision_train:.4%}")
    print(f"Precision en HOLDOUT ({n_holdout} tareas, nunca vistas):          {precision_holdout:.4%}")
    print(f"Diferencia (train - holdout): {(precision_train - precision_holdout):.4%}")


if __name__ == "__main__":
    ejecutar_diagnostico()
