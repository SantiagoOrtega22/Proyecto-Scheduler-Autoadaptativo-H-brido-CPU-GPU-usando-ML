import csv
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:  # Mismo techo de normalización que usa el codificador (Dim_k_log2 = log2(N) / MAX_LOG2_DIM).
    from Entorno.codificador_csv import MAX_LOG2_DIM
except ImportError:  # gym.py importado fuera del paquete Entorno
    from codificador_csv import MAX_LOG2_DIM


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
        modo_particion: str = "filas",
        particion: int = 0,
        n_particiones: int = 5,
    ) -> None:
        """
        Args:
            csv_path: CSV codificado por codificador_csv.py.
            tamano_lote: Tareas por episodio.
            shuffle: Si True, cada episodio muestrea tareas al azar.
            holdout_fraction: Fracción reservada en el modo 'filas' (0.0 = sin reserva,
                comportamiento original).
            split: 'train', 'holdout' o 'all' (default, dataset completo).
            split_seed: Semilla de la partición por filas.
            modo_particion: 'filas' (default, partición aleatoria por filas, igual que
                antes de existir este parámetro) o 'tamano' (partición por tamaño de
                problema, ver `asignar_particiones_por_tamano`). En modo 'tamano' la
                reserva se activa con split='train' o 'holdout' e ignora holdout_fraction.
            particion: Índice (0..n_particiones-1) de la partición reservada en modo 'tamano'.
            n_particiones: Número de particiones en modo 'tamano'.
        """
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

        if modo_particion == "tamano":
            if split in ("train", "holdout"):
                self.dataset_tareas = self._dividir_por_tamano(
                    self.dataset_tareas, particion, n_particiones, split
                )
        elif modo_particion != "filas":
            raise ValueError(f"modo_particion debe ser 'filas' o 'tamano', no '{modo_particion}'.")
        elif holdout_fraction > 0.0:
            self.dataset_tareas = self._dividir_dataset(
                self.dataset_tareas, holdout_fraction, split, split_seed
            )

        self.cola_tareas: list[dict] = []
        self.estado_actual: np.ndarray | None = None

    @staticmethod
    def clave_tamano(obs: np.ndarray) -> tuple[str, float]:
        """Identifica la carga y el tamaño de problema de una tarea a partir de su vector.

        Todas las variantes de un mismo tamaño (precisión, transposición, dominio,
        dirección, layout) comparten la clave. El tamaño se lee de Dim_1_log2
        (obs[2]): N en GEMM (M = N = K) y en FFT 2D/3D (Nx = Ny = Nz), Nx en FFT 1D.
        Se usa el valor codificado tal cual, sin reconstruir N, para que la clave sea
        exacta aunque la observación venga en float32.

        Args:
            obs: Vector de observación de 23 dimensiones.

        Returns:
            tuple[str, float]: (carga, Dim_1_log2), con carga en
                {'GEMM', 'FFT1D', 'FFT2D', 'FFT3D'}.
        """
        if obs[0] > 0.5:
            carga = "GEMM"
        elif obs[4] > 0.0:
            carga = "FFT3D"
        elif obs[3] > 0.0:
            carga = "FFT2D"
        else:
            carga = "FFT1D"
        return carga, float(obs[2])

    @staticmethod
    def asignar_particiones_por_tamano(tareas: list[dict], n_particiones: int = 5) -> list[int]:
        """Asigna cada tarea a una partición según su tamaño de problema.

        Unidad de partición: el tamaño, no la fila. Todas las variantes de un tamaño
        caen en la misma partición, para que ninguna configuración reservada tenga
        casi duplicados en el entrenamiento. Dentro de cada carga y de cada octava
        [2^k, 2^(k+1)), los tamaños se ordenan y se reparten de forma alterna
        (0, 1, ..., n-1, 0, 1, ...), de modo que cada partición cubre todo el rango
        del barrido y los vecinos inmediatos de un tamaño reservado quedan en el
        entrenamiento: es una prueba de interpolación dentro de los rangos medidos.
        El punto de inicio de la alternancia rota con la octava para equilibrar el
        número de tareas entre particiones. El tamaño mínimo y el máximo de cada
        carga se dejan siempre en el entrenamiento (partición -1): reservarlos sería
        extrapolar fuera del rango, no interpolar. La asignación es determinista y
        no usa semilla.

        Args:
            tareas: Tareas cargadas del CSV (cada una con clave 'obs').
            n_particiones: Número de particiones.

        Returns:
            list[int]: Partición (0..n_particiones-1) de cada tarea, en el mismo orden,
                o -1 para los extremos del rango, que nunca se reservan.
        """
        claves = [PlanificadorEnv.clave_tamano(t["obs"]) for t in tareas]
        # Octava de cada tamaño: floor(log2 N) = floor(Dim_1_log2 * MAX_LOG2_DIM).
        # El 1e-4 absorbe el redondeo float32 en las potencias de dos exactas.
        grupos: dict[tuple[str, int], set[float]] = {}
        extremos: dict[str, tuple[float, float]] = {}
        for carga, valor in claves:
            octava = int(np.floor(valor * MAX_LOG2_DIM + 1e-4))
            grupos.setdefault((carga, octava), set()).add(valor)
            lo, hi = extremos.get(carga, (valor, valor))
            extremos[carga] = (min(lo, valor), max(hi, valor))

        particion_de: dict[tuple[str, float], int] = {}
        for (carga, octava), valores in grupos.items():
            for rango, valor in enumerate(sorted(valores)):
                particion_de[(carga, valor)] = (rango + octava) % n_particiones
        for carga, (lo, hi) in extremos.items():
            particion_de[(carga, lo)] = -1
            particion_de[(carga, hi)] = -1

        return [particion_de[c] for c in claves]

    @staticmethod
    def _dividir_por_tamano(
        tareas: list[dict], particion: int, n_particiones: int, split: str
    ) -> list[dict]:
        """Devuelve las tareas de entrenamiento o las reservadas de la partición indicada.

        Args:
            tareas: Lista completa de tareas.
            particion: Índice de la partición reservada.
            n_particiones: Número de particiones.
            split: 'train' (todas menos la reservada) u 'holdout' (solo la reservada).

        Returns:
            list[dict]: Subconjunto solicitado.
        """
        if not 0 <= particion < n_particiones:
            raise ValueError(f"particion debe estar en [0, {n_particiones - 1}].")
        asignacion = PlanificadorEnv.asignar_particiones_por_tamano(tareas, n_particiones)
        if split == "holdout":
            return [t for t, p in zip(tareas, asignacion) if p == particion]
        return [t for t, p in zip(tareas, asignacion) if p != particion]

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