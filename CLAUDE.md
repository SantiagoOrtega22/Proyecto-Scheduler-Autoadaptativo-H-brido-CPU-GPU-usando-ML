# CLAUDE.md — Benchmark de Recolección de Datos para Agente de RL (CPU-GPU EDP Scheduler)

Este archivo da contexto persistente a Claude Code para este repositorio. Colócalo en la raíz del proyecto (o del subdirectorio del benchmark) para que se cargue automáticamente en cada sesión, tanto en terminal como en la extensión de VS Code.

## 1. Contexto del trabajo de grado

**Título:** Diseño, Implementación y Evaluación de un Planificador de Tareas Basado en Aprendizaje por Refuerzo para la Optimización del Producto Energía-Retardo (EDP) en Sistemas CPU-GPU.
**Universidad:** UIS — Escuela de Ingeniería de Sistemas e Informática.
**Infraestructura de validación:** Supercomputador SC3 (clúster Guane), nodo heterogéneo Intel Xeon + NVIDIA.

Este módulo (**benchmark**) es la Fase 1 del proyecto: caracteriza el comportamiento energético y de latencia de las cargas de trabajo para establecer la línea base (ground truth) contra la que luego se comparará el agente de RL. Los datos que produce alimentan el entrenamiento del planificador (Fases 2-3), así que su corrección y reproducibilidad son críticas para el resto del trabajo.

## 2. Objetivo del módulo

Recolectar mediciones de CPU y GPU resultantes de ejecutar **GEMM** y **FFT** en distintos tamaños, con y sin monitoreo de energía activo, para calcular Tiempo, Potencia, Energía, EDP y GFLOPS por ejecución.

## 3. Alcance y límites acordados (no expandir sin confirmarlo con el usuario)

Estos límites vienen de las aclaraciones entregadas al comité evaluador sobre el plan de trabajo de grado. Claude Code **no debe sugerir ni implementar** funcionalidades que los contradigan salvo que el usuario lo pida explícitamente:

- **Sin variable de saturación externa en el estado del MDP.** Las reservas de recursos en SC3 son exclusivas (el nodo/GPU asignado no se comparte), así que no hace falta modelar contención externa.
- **Fase 2 prioriza convergencia sobre optimización del motor de IA.** No es necesario perfilar/optimizar agresivamente el overhead de inferencia de las DQN; el foco es validar que la política de asignación converge.
- **Solo dos cargas de trabajo:** GEMM (compute-bound) y FFT (memory-bound). No agregar cargas con flujos de control estocásticos: excede el alcance metodológico y las 16 semanas del cronograma. Eso queda como trabajo futuro.
- **Telemetría limitada a RAPL (CPU) y NVML (GPU).** No se requiere instrumentación física externa (multímetros, etc.); el proyecto asume que estas interfaces mantienen calibración de fábrica.
- **Planificación estrictamente intra-nodo.** No implementar ni sugerir orquestación distribuida (MPI, Slurm, PBS) ni integración con SYCL, optimizadores bayesianos, o arquitecturas específicas como A100 fuera de lo ya disponible en SC3.
- **Sin modificaciones al kernel.** El planificador vive enteramente en espacio de usuario; no tocar el scheduler CFS de Linux ni la gestión de memoria del SO.
- **Soft real-time, no hard real-time.** Tolerancias de milisegundos son aceptables; no se garantiza determinismo a nivel de microsegundos.
- **Sin garantía de generalización zero-shot.** El agente se valida solo con los perfiles de GEMM/FFT usados en entrenamiento.

## 4. Funcionalidades del benchmark

- **Fuente de datos:** obtiene matrices/arreglos desde el banco de datos, o los genera en memoria de forma determinista con semilla fija (`BENCH_SEED = 42`) para garantizar reproducibilidad científica entre corridas CPU/GPU.
- **Variantes GEMM:** `sgemm`, `dgemm`, `cgemm`, `zgemm`, variando parámetros internos como transposiciones.
- **Variantes FFT:** 1D, 2D, 3D, con variantes `R2C` y `C2C`.
- **Medición de métricas:**
  - GPU: tiempo vía **NVML**, incluyendo tiempos de transferencia.
  - CPU: tiempo vía **chronos**.
- **Modos de ejecución:** ejecuta los binarios correspondientes en cada dispositivo según el modo solicitado; varía el tamaño `N` (matriz NxN para GEMM, tamaño de arreglo para FFT); si el modo involucra ambos dispositivos, alterna la ejecución entre ellos para cada `N`.
- **Protocolo de medición riguroso:**
  1. 4 ejecuciones de warm-up antes de cada medición.
  2. Primera ejecución sin herramientas de energía activas → mide solo tiempo y calcula GFLOPS.
  3. Segunda ejecución idéntica (mismo `N`) con monitoreo de potencia continuo activo (RAPL o `nvmlDeviceGetPowerUsage`).
- **Salida y reporte:**
  - Campos requeridos: Device, N, Time (ms), Average Power (W), Energy (J), EDP (J·s), GFLOPS.
  - Impresión en tiempo real en consola (formato tabla) + guardado a `.csv`.
  - El `.csv` debe incluir metadata de los parámetros de ejecución (p. ej. para GEMM: `sgemm`, `transA: N`, `transB: T`, etc.).

## 5. Modos de ejecución (definidos por el usuario)

| Modo | Descripción |
|---|---|
| Full Execution (default) | Barrido completo: todos los algoritmos, todos los dispositivos, todos los parámetros |
| Quick Test | Tamaños de `N` pequeños, sin variar parámetros |
| Only CPU / Only GPU | Restringe el dispositivo de ejecución |
| Only GEMM / Only FFT | Restringe el algoritmo |

## 6. Arquitectura y stack tecnológico

- **Orquestador del benchmark:** Python.
- **GEMM CPU:** MKL / OpenBLAS. **GEMM GPU:** cuBLAS.
- **FFT CPU:** FFTW. **FFT GPU:** cuFFT.
- **Medición:** RAPL (CPU), NVML (GPU), chronos.
- El sistema consta de un banco de datos con las matrices/arreglos por tamaño y tipo de dato, códigos específicos por función GEMM/FFT, y el orquestador que despacha ejecuciones, recolecta mediciones y las persiste.

## 7. Estándares de código

### 7.1 Calidad de código
- **SRP estricto:** telemetría (RAPL/NVML), orquestación de ejecución, y exportación de datos van en clases/módulos separados.
- **Nomenclatura:** `snake_case` para variables/funciones, `PascalCase` para clases, `UPPER_SNAKE_CASE` para constantes. Nombres descriptivos (`measure_gpu_power()`, no `get_pwr()`).
- **Manejo de errores:** `try-except` explícito en llamadas a hardware (init de NVML, acceso a archivos RAPL). Loguear errores; si falla una medición, saltar esa iteración de forma segura sin abortar el barrido completo.
- **Type hints:** obligatorios en todas las funciones Python (p. ej. `def run_warmup(n_size: int) -> None:`).

### 7.2 Rigor HPC
- **Gestión de recursos:** usar siempre context managers (`with`) para archivos y conexiones.
- **Sincronización explícita:** las tareas GPU deben ir precedidas y seguidas de `cudaDeviceSynchronize()` para que el timer del host refleje la finalización real del kernel.
- **Gestión de memoria:** liberar explícitamente matrices/arreglos grandes entre corridas para evitar leaks y errores OOM en barridos largos.
- **Generación determinista de datos:** matrices y vectores generados con un PRNG determinista y semilla fija (`BENCH_SEED = 42`) tanto en CPU como en GPU, para reproducibilidad estricta e inputs de prueba idénticos entre corridas.

### 7.3 Documentación
- **Docstrings** estilo Google o NumPy en todas las clases/métodos: propósito, argumentos (tipo y descripción), valores de retorno, excepciones.
- **Comentarios inline** explicando el *por qué* de decisiones de protocolo HPC (p. ej. `# Aislamiento estricto: herramienta de energía desactivada para medir GFLOPS pico`).
- **README.md en la raíz** con: requisitos de hardware (GPU NVIDIA, soporte Intel RAPL), dependencias de software (`requirements.txt`), y guía de ejecución paso a paso.

## 8. Requisitos no funcionales

- **Confiabilidad:** todas las mediciones deben ser confiables y estar libres de errores o riesgos de medición (ver protocolo de aislamiento en la sección 4).

## 9. Cómo trabajar en este repo (para Claude Code)

- Antes de tocar el código de medición, revisa si el cambio afecta el protocolo de aislamiento (warm-up → medición sin energía → medición con energía). No lo reordenes ni lo simplifiques sin confirmarlo.
- Si una tarea implica ampliar el alcance descrito en la Sección 3 (nuevas cargas de trabajo, nuevos dispositivos, orquestación distribuida, etc.), señálalo explícitamente y pregunta antes de implementarlo — el alcance ya fue acordado con el comité evaluador y cambiarlo tiene implicaciones sobre el resto del trabajo de grado.
- Mantén la semilla `BENCH_SEED = 42` como constante única y reutilizada en todo el código que genere datos aleatorios; no la dupliques con valores distintos en otros módulos.
- Al añadir nuevas métricas o columnas al `.csv`, actualiza también el README y cualquier metadata documentada de los parámetros de ejecución.
