import os
import sys
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# Agregar el directorio principal al PATH para importar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv
from estilo_graficas import (
    COLOR_ACENTO_RL,
    COLOR_CPU,
    COLOR_GPU,
    PALETA_CATEGORICA,
    COLOR_SUPERFICIE,
    COLOR_TEXTO_PRIMARIO,
    configurar_estilo_pastel,
    etiquetar_barras,
    limpiar_bordes,
    resaltar_barra,
)

BENCH_SEED = 42  # Semilla determinista (CLAUDE.md #9): fija qué tareas caen en la muestra evaluada.


def generar_grafica_comparativa(
    resultados: dict,
    claves: list[str],
    colores: list[str],
    titulo: str,
    img_salida: str,
    resaltar: str | None = None,
) -> None:
    """Genera una figura de 3 paneles (Tiempo, Energía, EDP) para un subconjunto de políticas.

    Args:
        resultados: Diccionario politica -> {'t', 'e', 'edp'} con todas las políticas simuladas.
        claves: Subconjunto y orden de políticas a graficar (debe existir en `resultados`).
        colores: Un color por cada entrada de `claves`, mismo orden.
        titulo: Título general de la figura.
        img_salida: Ruta de archivo donde guardar el PNG.
        resaltar: Nombre de la política a resaltar con borde marcado (p. ej. 'Agente RL'), o None.
    """
    tiempos = [resultados[k]['t'] for k in claves]
    energias = [resultados[k]['e'] for k in claves]
    edps = [resultados[k]['edp'] for k in claves]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(titulo, fontsize=18, fontweight='bold', color=COLOR_TEXTO_PRIMARIO)

    datos_por_panel = [
        (axes[0], tiempos, 'Tiempo Total de Ejecución', 'Segundos', '{:.1f}'),
        (axes[1], energias, 'Consumo de Energía Acumulado', 'Joules', '{:.0f}'),
        (axes[2], edps, 'Producto Energía-Retraso (EDP)', 'Magnitud EDP', '{:.0f}'),
    ]

    for ax, valores, subtitulo, etiqueta_y, formato in datos_por_panel:
        # Borde de superficie (en vez de negro puro) entre barras adyacentes:
        # separa los rellenos pastel sin el contraste duro de un borde negro.
        barras = ax.bar(claves, valores, color=colores, edgecolor=COLOR_SUPERFICIE, linewidth=1.5, width=0.65)
        if resaltar is not None:
            # Resalta la barra indicada (p. ej. Agente RL, el resultado central de la tesis)
            # con un borde marcado para que destaque frente a las demás.
            resaltar_barra(barras, claves.index(resaltar))
        etiquetar_barras(ax, barras, formato)

        ax.set_title(subtitulo, fontsize=12)
        ax.set_ylabel(etiqueta_y, fontsize=11)
        ax.set_ylim(top=max(valores) * 1.15)  # margen para que las etiquetas no choquen con el título
        limpiar_bordes(ax)
        ax.tick_params(axis='x', rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(img_salida, dpi=300)
    print(f"[ÉXITO] Gráfica generada en: {img_salida}")

def evaluar_comparativa_extendida(num_muestras: int = 200):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(base_dir, "modelo_dqn_scheduler.zip")
    csv_path = os.path.join(base_dir, "Entorno", "dataset_pacca.csv")
    img_salida_dispositivos = os.path.join(base_dir, "grafica_comparativa_dispositivos.png")
    img_salida_heuristicas = os.path.join(base_dir, "grafica_comparativa_heuristicas.png")
    csv_salida = os.path.join(base_dir, "tabla_comparativa_tesis.csv")

    if not os.path.exists(modelo_path):
        print(f"Error: No se encontró el modelo en {modelo_path}")
        return

    print("Cargando el Agente Scheduler (DQN)...")
    modelo = DQN.load(modelo_path)

    print(f"Generando carga de trabajo sintética de {num_muestras} tareas...")
    env = PlanificadorEnv(csv_path=csv_path, tamano_lote=num_muestras, shuffle=True)
    env.reset(seed=BENCH_SEED)
    cola_simulacion = list(env.cola_tareas)
    
    # Estructura para almacenar resultados de las 7 políticas
    resultados = {
        'Solo CPU': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'Solo GPU': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'MET': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'MCT': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'Min-Min': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'Max-Min': {'t': 0.0, 'e': 0.0, 'edp': 0.0},
        'Agente RL': {'t': 0.0, 'e': 0.0, 'edp': 0.0}
    }

    print("Simulando heurísticas estáticas y Agente RL...\n")

    # ==========================================
    # SALVAGUARDA DE ESTRUCTURA DE DATOS
    # ==========================================
    # Garantizamos que la estructura del entorno no haya mutado silenciosamente.
    assert len(cola_simulacion) > 0, "La cola de tareas está vacía."
    assert len(cola_simulacion[0]["metricas"]) == 2, "Se esperaban exactamente 2 métricas (Índice 0: CPU, Índice 1: GPU)."

    # 1 y 2. SOLO CPU y SOLO GPU
    # Un solo dispositivo procesa toda la cola: aqui la suma serial de tiempos SI es
    # el makespan real (no hay una segunda cola en paralelo con la que comparar).
    for tarea in cola_simulacion:
        m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]

        resultados['Solo CPU']['t'] += m_cpu["tiempo"]
        resultados['Solo CPU']['e'] += m_cpu["energia"]

        resultados['Solo GPU']['t'] += m_gpu["tiempo"]
        resultados['Solo GPU']['e'] += m_gpu["energia"]
    # EDP de sistema = Energia_total * Tiempo_total (no la suma de EDP por tarea: el EDP
    # no es aditivo, ver nota en MCT).
    resultados['Solo CPU']['edp'] = resultados['Solo CPU']['e'] * resultados['Solo CPU']['t']
    resultados['Solo GPU']['edp'] = resultados['Solo GPU']['e'] * resultados['Solo GPU']['t']

    # 3. MET (Minimum Execution Time) - Greedy puro de tiempo (decide por tarea, sin
    # conocimiento de la cola), pero igual reparte trabajo entre CPU y GPU en paralelo:
    # el tiempo de sistema es su makespan, igual que en MCT/Min-Min/Max-Min.
    libre_cpu, libre_gpu = 0.0, 0.0
    for tarea in cola_simulacion:
        m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
        # En caso de empate (<=), la convención estricta delega la tarea a la CPU.
        if m_cpu["tiempo"] <= m_gpu["tiempo"]:
            m_elegida = m_cpu
            libre_cpu += m_cpu["tiempo"]
        else:
            m_elegida = m_gpu
            libre_gpu += m_gpu["tiempo"]
        resultados['MET']['e'] += m_elegida["energia"]
    resultados['MET']['t'] = max(libre_cpu, libre_gpu)
    resultados['MET']['edp'] = resultados['MET']['e'] * resultados['MET']['t']

    # 4. MCT (Minimum Completion Time) - Greedy con conocimiento de colas
    libre_cpu, libre_gpu = 0.0, 0.0
    for tarea in cola_simulacion:
        m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
        if (libre_cpu + m_cpu["tiempo"]) <= (libre_gpu + m_gpu["tiempo"]):
            m_elegida = m_cpu
            libre_cpu += m_cpu["tiempo"]
        else:
            m_elegida = m_gpu
            libre_gpu += m_gpu["tiempo"]
        resultados['MCT']['e'] += m_elegida["energia"]
    # CPU y GPU procesan sus colas en paralelo: el tiempo de sistema es el makespan
    # (el dispositivo que termina último), no la suma serial de las tareas asignadas.
    resultados['MCT']['t'] = max(libre_cpu, libre_gpu)
    resultados['MCT']['edp'] = resultados['MCT']['e'] * resultados['MCT']['t']

    # 5. MIN-MIN (Heurística Batch)
    libre_cpu, libre_gpu = 0.0, 0.0
    # Shallow copy segura (solo se extraen punteros con pop, no se mutan los diccionarios)
    unmapped = list(cola_simulacion)
    while unmapped:
        best_idx, best_dev, global_min_ct = -1, -1, float('inf')
        for i, tarea in enumerate(unmapped):
            m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
            ct_cpu, ct_gpu = libre_cpu + m_cpu["tiempo"], libre_gpu + m_gpu["tiempo"]
            
            local_min = min(ct_cpu, ct_gpu)
            # En Min-Min, la desigualdad estricta (<) en local_min < global_min_ct asegura 
            # que en caso de empate global, gana la primera tarea hallada iterativamente.
            if local_min < global_min_ct:
                global_min_ct = local_min
                best_idx = i
                best_dev = 0 if ct_cpu <= ct_gpu else 1
        
        tarea = unmapped.pop(best_idx)
        m_elegida = tarea["metricas"][best_dev]
        if best_dev == 0: libre_cpu += m_elegida["tiempo"]
        else: libre_gpu += m_elegida["tiempo"]

        resultados['Min-Min']['e'] += m_elegida["energia"]
    # Makespan de las colas paralelas CPU/GPU, no suma serial (ver nota en MCT).
    resultados['Min-Min']['t'] = max(libre_cpu, libre_gpu)
    resultados['Min-Min']['edp'] = resultados['Min-Min']['e'] * resultados['Min-Min']['t']

    # 6. MAX-MIN (Heurística Batch)
    libre_cpu, libre_gpu = 0.0, 0.0
    unmapped = list(cola_simulacion)
    while unmapped:
        best_idx, best_dev, global_max_min_ct = -1, -1, -1.0
        for i, tarea in enumerate(unmapped):
            m_cpu, m_gpu = tarea["metricas"][0], tarea["metricas"][1]
            ct_cpu, ct_gpu = libre_cpu + m_cpu["tiempo"], libre_gpu + m_gpu["tiempo"]
            local_min = min(ct_cpu, ct_gpu)
            if local_min > global_max_min_ct:
                global_max_min_ct = local_min
                best_idx = i
                best_dev = 0 if ct_cpu <= ct_gpu else 1
        
        tarea = unmapped.pop(best_idx)
        m_elegida = tarea["metricas"][best_dev]
        if best_dev == 0: libre_cpu += m_elegida["tiempo"]
        else: libre_gpu += m_elegida["tiempo"]

        resultados['Max-Min']['e'] += m_elegida["energia"]
    # Makespan de las colas paralelas CPU/GPU, no suma serial (ver nota en MCT).
    resultados['Max-Min']['t'] = max(libre_cpu, libre_gpu)
    resultados['Max-Min']['edp'] = resultados['Max-Min']['e'] * resultados['Max-Min']['t']

    # 7. AGENTE RL (IA - DQN) - misma logica de makespan que las demas politicas hibridas.
    libre_cpu, libre_gpu = 0.0, 0.0
    for tarea in cola_simulacion:
        obs = tarea["obs"]
        accion, _ = modelo.predict(obs, deterministic=True)
        accion_int = int(np.asarray(accion).item())
        m_elegida = tarea["metricas"][accion_int]
        if accion_int == 0:
            libre_cpu += m_elegida["tiempo"]
        else:
            libre_gpu += m_elegida["tiempo"]
        resultados['Agente RL']['e'] += m_elegida["energia"]
    resultados['Agente RL']['t'] = max(libre_cpu, libre_gpu)
    resultados['Agente RL']['edp'] = resultados['Agente RL']['e'] * resultados['Agente RL']['t']

    # ==========================================
    # GENERACIÓN DE TABLA CSV
    # ==========================================
    with open(csv_salida, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Politica", "Tiempo Total (s)", "Energia Total (J)", "EDP Acumulado", "Ahorro vs Solo CPU (%)"])
        
        ref_edp = resultados['Solo CPU']['edp']
        for pol, vals in resultados.items():
            ahorro_vs_cpu = (1 - vals['edp'] / ref_edp) * 100 if ref_edp > 0 else 0.0
            writer.writerow([
                pol, 
                f"{vals['t']:.4f}", 
                f"{vals['e']:.4f}", 
                f"{vals['edp']:.4f}", 
                f"{ahorro_vs_cpu:.2f}%"
            ])
    print(f"[ÉXITO] Tabla generada en: {csv_salida}")

    # ==========================================
    # GENERACIÓN DE GRÁFICAS PARA LA TESIS (separadas: dispositivos vs. heurísticas)
    # ==========================================
    configurar_estilo_pastel()

    generar_grafica_comparativa(
        resultados,
        claves=['Solo CPU', 'Solo GPU', 'Agente RL'],
        colores=[COLOR_CPU, COLOR_GPU, COLOR_ACENTO_RL],
        titulo='Referencia de Dispositivo Único vs. Agente RL',
        img_salida=img_salida_dispositivos,
        resaltar='Agente RL',
    )

    generar_grafica_comparativa(
        resultados,
        claves=['MET', 'MCT', 'Min-Min', 'Max-Min', 'Agente RL'],
        colores=PALETA_CATEGORICA[:5],
        titulo='Heurísticas Clásicas vs. Scheduler RL',
        img_salida=img_salida_heuristicas,
        resaltar='Agente RL',
    )

    print("\n¡Simulación completada! Revisa los archivos .png y .csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--muestras", type=int, default=200, help="Cantidad de tareas")
    args = parser.parse_args()
    evaluar_comparativa_extendida(num_muestras=args.muestras)
