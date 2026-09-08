import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Agregar el directorio principal al PATH para poder importar Entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Entorno.gym import PlanificadorEnv

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

BENCH_SEED = 42  # Semilla determinista unica (CLAUDE.md #9), reutilizada en split y en el DQN.

class PrecisionCallback(BaseCallback):
    """
    Callback personalizado para calcular la precisión (% de veces que el agente 
    elige el dispositivo con el menor EDP) y registrarla en TensorBoard.
    """
    def __init__(self, verbose=0):
        super(PrecisionCallback, self).__init__(verbose)
        self.es_optimo_history = []
        
    def _on_step(self) -> bool:
        # Extraer el flag 'es_optimo' del diccionario 'info' del entorno
        for info in self.locals.get("infos", []):
            if "es_optimo" in info:
                self.es_optimo_history.append(info["es_optimo"])
                
        # Calcular y registrar en TensorBoard cada 100 pasos
        if len(self.es_optimo_history) >= 100:
            precision = np.mean(self.es_optimo_history)
            self.logger.record("metricas_personalizadas/precision", precision)
            self.es_optimo_history = []  # Limpiar buffer
            
        return True

def entrenar_agente(
    holdout_fraction: float = 0.0,
    split: str = "all",
    modelo_nombre: str = "modelo_dqn_scheduler",
    log_subdir: str | None = None,
) -> str:
    """
    Instancia el entorno PlanificadorEnv y entrena un agente DQN
    para tomar decisiones de asignación entre CPU y GPU buscando minimizar el EDP.

    Guarda los registros (TensorBoard) y exporta el modelo entrenado (.zip).

    Args:
        holdout_fraction: Fracción de tareas a reservar fuera del entrenamiento
            (0.0 = comportamiento original, entrena con el dataset completo).
        split: 'train' entrena solo con el complemento del holdout; 'all' (default)
            usa el dataset completo, igual que antes de este parámetro existir.
        modelo_nombre: Nombre base (sin extensión) del archivo .zip de salida.
        log_subdir: Subcarpeta de logs_entrenamiento para TensorBoard/Monitor.
            Si es None, usa logs_entrenamiento directamente (comportamiento original).

    Returns:
        str: Ruta absoluta del modelo guardado (sin extensión .zip).
    """
    print("--- Iniciando Configuración del Entrenamiento ---")

    # 1. Definir rutas relativas al proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Entorno", "dataset_pacca.csv")
    log_dir = os.path.join(base_dir, "logs_entrenamiento", log_subdir) if log_subdir else os.path.join(base_dir, "logs_entrenamiento")
    modelo_path = os.path.join(base_dir, modelo_nombre)

    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")

    # 2. Instanciar y envolver el entorno
    # Monitor: Guarda las recompensas y duraciones de los episodios en CSVs para TensorBoard/Gráficas
    env_base = PlanificadorEnv(
        csv_path=csv_path,
        holdout_fraction=holdout_fraction,
        split=split,
        split_seed=BENCH_SEED,
    )
    env = Monitor(env_base, log_dir)

    # 3. Configurar la arquitectura y los hiperparámetros del Agente DQN
    print("Inicializando arquitectura Deep Q-Network...")
    modelo = DQN(
        policy="MlpPolicy",          # Red Neuronal Multicapa estándar (Perceptrón Multicapa)
        env=env,
        learning_rate=1e-4,          # Tasa de aprendizaje de la red neuronal
        buffer_size=10000,           # Capacidad de la memoria de repetición (Experience Replay)
        learning_starts=50,          # Acciones iniciales aleatorias para llenar el buffer
        batch_size=128,               # Tamaño de lote (batch) al retropropagar gradientes
        gamma=0.0,                  # Factor de descuento (visión a futuro de la política)
        exploration_fraction=0.3,    # Explorar (aleatorio) durante el 30% del entrenamiento
        exploration_initial_eps=1.0, # Comenzar 100% aleatorio (exploración pura)
        exploration_final_eps=0.05,  # Terminar con un 5% mínimo de exploración continua
        tensorboard_log=log_dir,     # Ruta para ver las curvas de recompensa
        verbose=1,                   # Nivel de detalle en la consola
        seed=BENCH_SEED               # Semilla determinista (CLAUDE.md #9)
    )

    # 4. Ciclo de Entrenamiento
    # Entrenaremos durante 5000 pasos lógicos a modo de prueba inicial.
    # Dado que ahora el dataset tiene 9 filas, el agente iterará muchas veces sobre él
    # simulando una cola infinita de tareas entrantes.
    timesteps = 100000
    print(f"\n--- Iniciando Aprendizaje por {timesteps} Pasos ---")

    callback_precision = PrecisionCallback()
    modelo.learn(total_timesteps=timesteps, progress_bar=True, callback=callback_precision)

    # 5. Guardar el modelo entrenado a disco
    modelo.save(modelo_path)
    print(f"\n[Éxito] Modelo de pesos guardado en: {modelo_path}.zip")

    # 6. Prueba rápida de Inferencia
    print("\n--- Inferencia de Prueba (One-Shot) ---")
    obs, info = env.reset()
    # Usar el modelo de forma determinista (explotación pura, sin aleatoriedad)
    accion, _ = modelo.predict(obs, deterministic=True)

    dispositivo_str = "CPU" if accion == 0 else "GPU"
    print(f"Estado recibido (Características 22D):")
    print(obs)
    print(f"Decisión del Scheduler para esta tarea: {dispositivo_str}")

    return modelo_path

if __name__ == "__main__":
    entrenar_agente()
