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
    seed: int = BENCH_SEED,
    timesteps: int = 100000,
    dataset: str | None = None,
    modo_particion: str = "filas",
    particion: int = 0,
    n_particiones: int = 5,
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
        seed: Semilla del DQN (init de pesos, exploración epsilon-greedy y muestreo
            del replay buffer). Por defecto BENCH_SEED, para no alterar el modelo de
            tesis vigente; `barrido_semillas.py` la varía para medir la varianza
            entre corridas independientes. El split del dataset NO usa esta semilla:
            queda fijo en BENCH_SEED para que todas las corridas vean los mismos datos.
        timesteps: Pasos de entrenamiento del ciclo `learn()`.
        dataset: Ruta al CSV codificado a usar. Si es None (default), usa
            Entorno/dataset_rl.csv: GEMM + FFT de la campaña principal (FFT 1D
            con N >= 4096). La campaña FFT 1D de tamaños pequeños queda fuera
            porque se midió en otro trabajo con un desfase de potencia frente a
            la principal. Cualquier script que evalúe el modelo resultante debe
            usar este MISMO CSV, o la observación no coincidirá con lo aprendido.
        modo_particion: 'filas' (default, comportamiento original) o 'tamano'
            (partición por tamaño de problema para la prueba de interpolación,
            ver PlanificadorEnv.asignar_particiones_por_tamano y
            evaluar_interpolacion.py).
        particion: Partición reservada en modo 'tamano'.
        n_particiones: Número de particiones en modo 'tamano'.

    Returns:
        str: Ruta absoluta del modelo guardado (sin extensión .zip).
    """
    print("--- Iniciando Configuración del Entrenamiento ---")

    # 1. Definir rutas relativas al proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = dataset if dataset else os.path.join(base_dir, "Entorno", "dataset_rl.csv")
    log_dir = os.path.join(base_dir, "logs_entrenamiento", log_subdir) if log_subdir else os.path.join(base_dir, "logs_entrenamiento")
    modelo_path = os.path.join(base_dir, modelo_nombre)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(modelo_path), exist_ok=True)  # modelo_nombre puede incluir subcarpeta

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {csv_path}")

    # 2. Instanciar y envolver el entorno
    # Monitor: Guarda las recompensas y duraciones de los episodios en CSVs para TensorBoard/Gráficas
    env_base = PlanificadorEnv(
        csv_path=csv_path,
        holdout_fraction=holdout_fraction,
        split=split,
        split_seed=BENCH_SEED,
        modo_particion=modo_particion,
        particion=particion,
        n_particiones=n_particiones,
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
        seed=seed                    # Semilla determinista (CLAUDE.md #9); default BENCH_SEED
    )

    # 4. Ciclo de Entrenamiento
    # El agente itera muchas veces sobre el dataset, simulando una cola infinita
    # de tareas entrantes.
    print(f"\n--- Iniciando Aprendizaje por {timesteps} Pasos (seed={seed}) ---")

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
    print(f"Estado recibido (Características {env_base.observation_space.shape[0]}D):")
    print(obs)
    print(f"Decisión del Scheduler para esta tarea: {dispositivo_str}")

    return modelo_path

if __name__ == "__main__":
    entrenar_agente()
