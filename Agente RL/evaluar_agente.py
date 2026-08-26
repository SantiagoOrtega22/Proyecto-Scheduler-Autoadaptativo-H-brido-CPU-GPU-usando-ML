import os
import sys
import numpy as np
from stable_baselines3 import DQN

# Agregar el directorio principal al PATH para importar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv

import argparse

def evaluar_agente(num_muestras: int = 15):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(base_dir, "modelo_dqn_scheduler.zip")
    csv_path = os.path.join(base_dir, "Entorno", "dataset_pacca.csv")

    if not os.path.exists(modelo_path):
        print(f"Error: No se encontró el modelo entrenado en {modelo_path}")
        print("¡Asegúrate de correr train.py primero!")
        return

    print("Cargando el Cerebro del Scheduler (DQN)...")
    modelo = DQN.load(modelo_path)

    # Inicializamos el entorno con la cantidad de muestras solicitada
    print(f"Simulando llegada de {num_muestras} tareas al servidor...\n")
    env = PlanificadorEnv(csv_path=csv_path, tamano_lote=num_muestras, shuffle=True)
    obs, info = env.reset()
    
    print("==================================================================")
    print("               SCHEDULER AUTODAPTATIVO EN ACCIÓN")
    print("==================================================================")

    aciertos = 0
    paso = 1
    done = False
    
    while not done:
        # Extraemos la información real de la tarea antes de consumirla
        tarea_actual = env.cola_tareas[0]
        
        # Deduciendo el tipo de tarea desde el vector de estado (obs)
        is_gemm = obs[0] == 1.0
        # obs[2] es log2(Dim_1) normalizado entre 0 y 24.
        # Recuperamos el tamaño aproximado de la matriz/arreglo:
        tamano = int(2 ** (obs[2] * 24.0)) 
        nombre_tarea = f"GEMM (Matriz de {tamano}x{tamano})" if is_gemm else f"FFT (Arreglo de {tamano})"

        # 1. El modelo evalúa el estado y toma una decisión determinista (sin explorar)
        accion, _ = modelo.predict(obs, deterministic=True)
        accion_int = int(np.asarray(accion).item())
        eleccion_ia = "CPU" if accion_int == 0 else "GPU"
        
        # 2. Ejecutamos la acción en el entorno
        obs, reward, terminated, truncated, info = env.step(accion_int)
        
        # 3. Analizamos si la IA tomó la decisión correcta comparando con los datos reales
        edp_cpu = float(tarea_actual["metricas"][0]["edp"])
        edp_gpu = float(tarea_actual["metricas"][1]["edp"])
        ideal = "CPU" if edp_cpu <= edp_gpu else "GPU"
        es_optimo = info.get("es_optimo", 0.0) == 1.0
        
        if es_optimo:
            aciertos += 1
            resultado_str = "✅ ¡Decisión Óptima!"
        else:
            resultado_str = f"❌ Error (Ideal era {ideal})"

        # 4. Imprimir resultados del paso
        print(f"Tarea #{paso}: {nombre_tarea}")
        print(f"   ├─ EDP si usara CPU: {edp_cpu:.5f}")
        print(f"   ├─ EDP si usara GPU: {edp_gpu:.5f}")
        print(f"   └─ SCHEDULER ASIGNÓ A -> [ {eleccion_ia} ] {resultado_str}")
        print("-" * 66)
        
        paso += 1
        done = terminated or truncated

    precision = (aciertos / (paso - 1)) * 100
    print(f"==================================================================")
    print(f"  RESUMEN FINAL: {aciertos} de {paso - 1} tareas asignadas perfectamente ({precision:.1f}%)")
    print(f"==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar el Agente Scheduler DQN en vivo.")
    parser.add_argument(
        "-n", "--muestras",
        type=int,
        default=15,
        help="Cantidad de tareas a evaluar (por defecto: 15)"
    )
    args = parser.parse_args()
    evaluar_agente(num_muestras=args.muestras)

