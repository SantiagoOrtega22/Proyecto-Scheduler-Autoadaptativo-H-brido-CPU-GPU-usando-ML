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
        holdout_fraction: float = 0.0,
        split: str = "all",
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.tamano_lote = tamano_lote
        self.shuffle = shuffle

        # Espacio de acciones: 0 = CPU, 1 = GPU
        self.action_space = spaces.Discrete(2)

        # Espacio de estados: Vector unificado 23D [0.0, 1.0] (Ver DISENO_ENTORNO.md)
        self.observation_space = spaces.Box(
            low=np.zeros(23, dtype=np.float32),
            high=np.ones(23, dtype=np.float32),
            dtype=np.float32,
        )

        self.dataset_tareas: list[dict] = []
        if csv_path and os.path.exists(csv_path):
            self._cargar_dataset_csv(csv_path)
        else:
            self._generar_dataset_sintetico()

        if holdout_fraction > 0.0:
            self.dataset_tareas = self._dividir_dataset(
                self.dataset_tareas, holdout_fraction, split, split_seed
            )

        self.cola_tareas: list[dict] = []
        self.estado_actual: np.ndarray | None = None

    @staticmethod
    def _dividir_dataset(
        tareas: list[dict], holdout_fraction: float, split: str, seed: int
    ) -> list[dict]:
        """Separa deterministamente el dataset en un subconjunto de entrenamiento y
        uno de reserva (holdout), para diagnosticar memorización vs. interpolación
        de la frontera de decisión (ver DISENO_ENTORNO.md, diagnostico de holdout).

        Args:
            tareas: Lista completa de tareas cargadas del CSV.
            holdout_fraction: Fracción (0.0-1.0) de tareas a reservar como holdout.
            split: 'train' devuelve el complemento del holdout, 'holdout' devuelve
                solo el holdout, cualquier otro valor devuelve todas las tareas.
            seed: Semilla del muestreo (reutilizar BENCH_SEED para reproducibilidad).

        Returns:
            list[dict]: Subconjunto de tareas correspondiente al split solicitado.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(tareas))
        n_holdout = int(round(len(tareas) * holdout_fraction))
        holdout_idx = set(indices[:n_holdout].tolist())

        if split == "holdout":
            return [t for i, t in enumerate(tareas) if i in holdout_idx]
        if split == "train":
            return [t for i, t in enumerate(tareas) if i not in holdout_idx]
        return tareas

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Selección de tareas para el episodio
        indices = np.arange(len(self.dataset_tareas))
        if self.shuffle:
            self.np_random.shuffle(indices)
        indices_seleccionados = indices[: min(self.tamano_lote, len(self.dataset_tareas))]

        self.cola_tareas = [self.dataset_tareas[i] for i in indices_seleccionados]
        self.estado_actual = self.cola_tareas[0]["obs"] if self.cola_tareas else np.zeros(23, dtype=np.float32)

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

        # Recompensa normalizada contra el peor dispositivo de ESTA tarea.
        #
        #   R(s,a) = (EDP_max(s) - EDP(s,a)) / EDP_max(s)
        #
        # donde EDP_max(s) = max(edp_cpu, edp_gpu) para la tarea actual.
        #
        # Sustituye a R = -ln(EDP + eps). Esa fórmula tomaba el valor ABSOLUTO del
        # EDP, que en este dataset abarca ~32 nats entre la tarea más pequeña y la
        # más grande: una variación que depende del tamaño de N, no de si la
        # decisión CPU/GPU fue correcta. Como la red comparte pesos entre todas las
        # tareas y el objetivo de regresión de Q es el reward crudo (gamma=0), esa
        # amplitud ahogaba la señal que sí importa (0.4-3.3 nats de diferencia entre
        # acertar y fallar), sesgando el aprendizaje hacia acertar la MAGNITUD de
        # las tareas grandes en vez de el ORDEN entre los dos dispositivos.
        #
        # Al dividir por el máximo de la propia tarea, el reward queda acotado en
        # [0, 1] sea cual sea la escala absoluta: 0 al elegir el peor dispositivo y
        # (1 - EDP_min/EDP_max) al elegir el mejor. Además, por ser una función
        # saturante del cociente (y no logarítmica), comprime los casos extremos:
        # una tarea donde la GPU gana por 20.000x deja de pesar desproporcionadamente
        # más que una donde gana por 1.5x. En el dataset mixto esto equilibra el
        # margen entre la clase mayoritaria y la minoritaria de 0.98x a 1.00x, y en
        # FFT lo baja de 8.05x a 2.85x.
        edp_peor = max(edp_cpu, edp_gpu)
        # Si ambos dispositivos miden exactamente 0 el cociente es indefinido; se
        # entrega el reward máximo porque cualquier acción es óptima en ese caso.
        reward = (edp_peor - edp_medido) / edp_peor if edp_peor > 0.0 else 1.0

        # Transición al siguiente estado
        terminated = len(self.cola_tareas) == 0
        truncated = False
        self.estado_actual = self.cola_tareas[0]["obs"] if not terminated else np.zeros(23, dtype=np.float32)

        info = {
            "dispositivo": "cpu" if action == 0 else "gpu",
            "edp": edp_medido,
            "energia_J": energia_joules,
            "tiempo_s": tiempo_segundos,
            "es_optimo": es_optimo
        }
        return self.estado_actual, float(reward), terminated, truncated, info

    # Nombres de columnas del vector de observación, en el mismo orden en que
    # codificador_csv.py los escribe. Se mantiene una única copia a nivel de
    # clase (no repetida en cada fila del bucle) para que una futura dimensión
    # nueva solo tenga que actualizarse aquí.
    COLUMNAS_OBS = [
        "is_GEMM", "is_FFT",
        "Dim_1_log2", "Dim_2_log2", "Dim_3_log2", "Batch_log2",
        "Prec_S", "Prec_D", "Prec_C", "Prec_Z",
        "OpA_N", "OpA_T", "OpA_C",
        "OpB_N", "OpB_T", "OpB_C",
        "Dom_C2C", "Dom_R2C",
        "Dir_F", "Dir_I",
        "Layout_I", "Layout_O",
        "Radix_log2",
    ]

    def _cargar_dataset_csv(self, csv_path: str) -> None:
        """Carga y parsea el CSV con vectores de observación y métricas CPU/GPU."""
        self.dataset_tareas.clear()
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                # Extracción del vector de observación. Se valida contra
                # observation_space.shape en vez de asumir 23 a mano, para que un
                # CSV desalineado falle aquí con un mensaje claro en vez de
                # reventar mas adelante dentro de stable-baselines3 (ValueError de
                # broadcasting sin contexto, como ocurrió con un CSV de 22D contra
                # un espacio de 23D).
                obs = np.array([float(fila[col]) for col in self.COLUMNAS_OBS], dtype=np.float32)
                if obs.shape != self.observation_space.shape:
                    raise ValueError(
                        f"El CSV '{csv_path}' produce vectores de observación de "
                        f"forma {obs.shape}, pero observation_space espera "
                        f"{self.observation_space.shape}. Regenera el dataset con "
                        "codificador_csv.py (la version actual)."
                    )

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
            obs = np.zeros(23, dtype=np.float32)
            obs[0] = 1.0  # is_GEMM
            obs[2] = (i + 1) / 20.0  # Dim_1 escalada
            metricas = {
                0: {"edp": 1e-4 * (i + 1), "energia": 5.0, "tiempo": 0.02},
                1: {"edp": 1e-5 * (i + 1), "energia": 15.0, "tiempo": 0.001},
            }
            self.dataset_tareas.append({"obs": obs, "metricas": metricas})