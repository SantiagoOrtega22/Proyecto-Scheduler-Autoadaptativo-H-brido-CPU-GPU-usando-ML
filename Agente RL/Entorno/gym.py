import csv
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PlanificadorEnv(gym.Env):
    """Entorno Gymnasium para planificar tareas GEMM y FFT en CPU/GPU optimizando EDP mediante datos de benchmark."""

    def __init__(
        self,
        csv_path: str | None = None,
        tamano_lote: int = 100,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.tamano_lote = tamano_lote
        self.shuffle = shuffle

        # Espacio de acciones: 0 = CPU, 1 = GPU
        self.action_space = spaces.Discrete(2)

        # Espacio de estados: Vector unificado 22D [0.0, 1.0] (Ver DISENO_ENTORNO.md)
        self.observation_space = spaces.Box(
            low=np.zeros(22, dtype=np.float32),
            high=np.ones(22, dtype=np.float32),
            dtype=np.float32,
        )

        self.dataset_tareas: list[dict] = []
        if csv_path and os.path.exists(csv_path):
            self._cargar_dataset_csv(csv_path)
        else:
            self._generar_dataset_sintetico()

        self.cola_tareas: list[dict] = []
        self.estado_actual: np.ndarray | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Selección de tareas para el episodio
        indices = np.arange(len(self.dataset_tareas))
        if self.shuffle:
            self.np_random.shuffle(indices)
        indices_seleccionados = indices[: min(self.tamano_lote, len(self.dataset_tareas))]

        self.cola_tareas = [self.dataset_tareas[i] for i in indices_seleccionados]
        self.estado_actual = self.cola_tareas[0]["obs"] if self.cola_tareas else np.zeros(22, dtype=np.float32)

        return self.estado_actual, {}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Convertir a entero nativo de Python en caso de recibir numpy.ndarray o escalar
        action_idx = int(np.asarray(action).item())

        tarea_actual = self.cola_tareas.pop(0)

        # Extracción de métricas para la acción ejecutada y cálculo del óptimo
        edp_cpu = float(tarea_actual["metricas"][0]["edp"])
        edp_gpu = float(tarea_actual["metricas"][1]["edp"])
        
        # El óptimo es el dispositivo con el menor EDP
        accion_optima = 0 if edp_cpu <= edp_gpu else 1
        es_optimo = 1.0 if action_idx == accion_optima else 0.0

        metricas = tarea_actual["metricas"][action_idx]
        energia_joules = float(metricas["energia"])
        tiempo_segundos = float(metricas["tiempo"])
        edp_medido = float(metricas["edp"])

        # Función de recompensa logarítmica negativa
        reward = -np.log(edp_medido + 1e-5)

        # Transición al siguiente estado
        terminated = len(self.cola_tareas) == 0
        truncated = False
        self.estado_actual = self.cola_tareas[0]["obs"] if not terminated else np.zeros(22, dtype=np.float32)

        info = {
            "dispositivo": "cpu" if action == 0 else "gpu",
            "edp": edp_medido,
            "energia_J": energia_joules,
            "tiempo_s": tiempo_segundos,
            "es_optimo": es_optimo
        }
        return self.estado_actual, float(reward), terminated, truncated, info

    def _cargar_dataset_csv(self, csv_path: str) -> None:
        """Carga y parsea el CSV con vectores de 22D y métricas CPU/GPU."""
        self.dataset_tareas.clear()
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                # Definición de nombres descriptivos One-Hot y Log2
                columnas_obs = [
                    "is_GEMM", "is_FFT", 
                    "Dim_1_log2", "Dim_2_log2", "Dim_3_log2", "Batch_log2",
                    "Prec_S", "Prec_D", "Prec_C", "Prec_Z",
                    "OpA_N", "OpA_T", "OpA_C",
                    "OpB_N", "OpB_T", "OpB_C",
                    "Dom_C2C", "Dom_R2C",
                    "Dir_F", "Dir_I",
                    "Layout_I", "Layout_O"
                ]
                
                # Extracción del vector de observación 22D
                obs = np.array([float(fila[col]) for col in columnas_obs], dtype=np.float32)

                # Extracción de métricas de rendimiento por dispositivo
                metricas = {
                    0: {  # CPU
                        "edp": float(fila["cpu_edp"]),
                        "energia": float(fila.get("cpu_energy", fila.get("cpu_energy_j", 0.0))),
                        "tiempo": float(fila.get("cpu_time", fila.get("cpu_time_sec", 0.0))),
                    },
                    1: {  # GPU
                        "edp": float(fila["gpu_edp"]),
                        "energia": float(fila.get("gpu_energy", fila.get("gpu_energy_j", 0.0))),
                        "tiempo": float(fila.get("gpu_time", fila.get("gpu_time_sec", 0.0))),
                    },
                }

                self.dataset_tareas.append({"obs": obs, "metricas": metricas})

    def _generar_dataset_sintetico(self) -> None:
        """Genera datos de respaldo si no se proporciona un CSV."""
        self.dataset_tareas.clear()
        for i in range(20):
            obs = np.zeros(22, dtype=np.float32)
            obs[0] = 1.0  # is_GEMM
            obs[2] = (i + 1) / 20.0  # Dim_1 escalada
            metricas = {
                0: {"edp": 1e-4 * (i + 1), "energia": 5.0, "tiempo": 0.02},
                1: {"edp": 1e-5 * (i + 1), "energia": 15.0, "tiempo": 0.001},
            }
            self.dataset_tareas.append({"obs": obs, "metricas": metricas})