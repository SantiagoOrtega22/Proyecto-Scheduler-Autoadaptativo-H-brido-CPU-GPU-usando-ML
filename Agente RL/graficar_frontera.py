import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3 import DQN

# Agregar el directorio principal al PATH para importar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.codificador_csv import MAX_LOG2_DIM
from Entorno.gym import PlanificadorEnv
from estilo_graficas import (
    COLOR_CPU,
    COLOR_GPU,
    COLOR_TEXTO_PRIMARIO,
    configurar_estilo_pastel,
    limpiar_bordes,
)

def generar_grafica_frontera(dataset: str | None = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(base_dir, "modelo_dqn_scheduler.zip")
    # Debe apuntar al MISMO dataset con el que se entrenó el modelo (el nombre de
    # archivo no queda registrado dentro del .zip). Por defecto dataset_pacca.csv
    # por compatibilidad con corridas previas; pasar --dataset para un dataset
    # FFT-only, GEMM-only o mixto distinto.
    csv_path = dataset if dataset else os.path.join(base_dir, "Entorno", "dataset_pacca.csv")

    if not os.path.exists(modelo_path):
        print(f"Error: No se encontró el modelo entrenado en {modelo_path}")
        return

    print("Cargando el Agente Scheduler (DQN)...")
    modelo = DQN.load(modelo_path)

    # Cargamos TODO el dataset sin mezclarlo (shuffle=False)
    print("Cargando dataset para extraer tareas...")
    env = PlanificadorEnv(csv_path=csv_path, tamano_lote=15000, shuffle=True)
    env.reset()

    tamanos = []
    decisiones = []

    # Si el dataset trae GEMM se prioriza esa frontera (Precisión Simple, como
    # siempre); si es FFT-only (is_GEMM=0 en todas las filas) se cae a FFT
    # Precisión Simple para no devolver "sin tareas" en una corrida solo-FFT.
    hay_gemm_s = any(t["obs"][0] == 1.0 and t["obs"][6] == 1.0 for t in env.cola_tareas)
    if hay_gemm_s:
        tipo_frontera, titulo_algo = "GEMM", "GEMM"
    else:
        tipo_frontera, titulo_algo = "FFT", "FFT"
    img_salida = os.path.join(base_dir, f"frontera_decision_{tipo_frontera.lower()}.png")

    for tarea in env.cola_tareas:
        obs = tarea["obs"]
        # obs[0] es is_GEMM, obs[1] es is_FFT, obs[6] es Prec_S (Float32)
        es_del_tipo = (obs[0] == 1.0) if tipo_frontera == "GEMM" else (obs[1] == 1.0)
        prec_s = obs[6] == 1.0

        if es_del_tipo and prec_s:
            # Revertimos la escala log2 para sacar el tamaño real N. Usa la misma
            # constante que el codificador: si divergen, el eje X queda mal escalado.
            # obs[2] es M (GEMM) o Nx (FFT) en ambos casos (ver DISENO_ENTORNO.md).
            N = int(round(2 ** (obs[2] * MAX_LOG2_DIM)))

            # La IA toma la decisión
            accion, _ = modelo.predict(obs, deterministic=True)
            accion_int = int(np.asarray(accion).item())

            tamanos.append(N)
            decisiones.append(accion_int) # 0 = CPU, 1 = GPU

    if len(tamanos) == 0:
        print(f"No se encontraron tareas {titulo_algo} de Precisión Simple en el dataset.")
        return

    # Ordenar los datos por tamaño N de menor a mayor
    tamanos, decisiones = zip(*sorted(zip(tamanos, decisiones)))

    # Generación de la gráfica
    configurar_estilo_pastel()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Dibujamos los puntos: CPU (0) en rojo pastel, GPU (1) en verde pastel.
    # Semántica y colores consistentes con evaluar_agente.py (estilo_graficas.py).
    colores = [COLOR_CPU if d == 0 else COLOR_GPU for d in decisiones]

    # Borde blanco (en vez de negro puro) para separar visualmente los puntos
    # superpuestos sin el contraste duro de un borde negro sobre relleno pastel.
    ax.scatter(tamanos, decisiones, c=colores, s=110, edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3)

    # Formato del gráfico
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CPU', 'GPU'], fontsize=12, fontweight='bold')
    ax.set_ylim(-0.5, 1.5)
    ax.set_xscale('log', base=2)  # Escala logarítmica para ver bien los tamaños pequeños y masivos

    ax.set_xlabel('Tamaño de la Matriz (N)', fontsize=12)
    ax.set_title(f'Frontera de Decisión del Agente RL ({titulo_algo} - Float32)', fontsize=14, fontweight='bold', color=COLOR_TEXTO_PRIMARIO)
    ax.grid(True, which="both", axis='x', ls="--", alpha=0.5)
    ax.grid(False, axis='y')
    limpiar_bordes(ax)

    leyenda = [
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor=COLOR_CPU, markeredgecolor='white', markersize=10, label='CPU'),
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor=COLOR_GPU, markeredgecolor='white', markersize=10, label='GPU'),
    ]
    ax.legend(handles=leyenda, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Dispositivo')

    plt.tight_layout()
    plt.savefig(img_salida, dpi=300)
    print(f"[ÉXITO] Gráfica de frontera de decisión guardada en: {img_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Ruta al CSV codificado a usar. Debe coincidir con el dataset usado "
             "para entrenar el modelo. Por defecto: Entorno/dataset_pacca.csv.",
    )
    args = parser.parse_args()
    generar_grafica_frontera(dataset=args.dataset)

