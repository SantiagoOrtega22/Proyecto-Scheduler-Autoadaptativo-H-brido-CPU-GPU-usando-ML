# Especificación Técnica y Diseño del Entorno de Aprendizaje por Refuerzo (`PlanificadorEnv`)

## 1. Visión General del Entorno

El entorno `PlanificadorEnv` implementa la interfaz estándar de **Gymnasium** para modelar el problema de planificación y despacho autoadaptativo de tareas de Álgebra Lineal densa (**GEMM**) y Transformadas Rápidas de Fourier (**FFT**) sobre una arquitectura heterogénea conformada por **CPU** y **GPU**.

El entorno opera en modalidad **Data-Driven (Guiado por Datasets de Benchmarks)**, cargando directamente las mediciones reales de telemetría de archivos `.csv` (`benchmark_results.csv`, `fft_benchmark_results.csv`), permitiendo un entrenamiento ultra rápido ($O(1)$ en memoria) para algoritmos como **DQN** en **Stable-Baselines3**.

El objetivo del agente es aprender una política $\pi(a|s)$ que minimice el **Energy-Delay Product (EDP)**:

$$\text{EDP} = \text{Energía (Joules)} \times \text{Tiempo (segundos)}$$

---

## 2. Formulación como Proceso de Decisión de Markov (MDP)

### 2.1. Espacio de Acciones ($\mathcal{A}$)
Espacio discreto binario: `spaces.Discrete(2)`.
* **$a = 0$**: Despachar y ejecutar la tarea en la **CPU** (OpenBLAS / MKL / FFTW).
* **$a = 1$**: Despachar y ejecutar la tarea en la **GPU** (cuBLAS / cuFFT).

### 2.2. Espacio de Estados / Observaciones ($\mathcal{S}$)
Espacio continuo acotado: `spaces.Box(low=0.0, high=1.0, shape=(22,), dtype=np.float32)`.

Vector unificado de **22 dimensiones** con **One-Hot Encoding** para variables categóricas y **escalado $\log_2$ normalizado** para dimensiones cuantitativas.

### 2.3. Función de Recompensa ($\mathcal{R}$)
$$R(s, a) = -\ln(\text{EDP} + \epsilon)$$

Donde $\epsilon = 10^{-9}$. Maximizar $R(s, a)$ equivale estrictamente a minimizar el EDP medido en el CSV.

### 2.4. Política de Episodios One-Shot
* **Episodio**: Configurable mediante `tasks_per_episode` (por defecto $1$ para decisiones puras One-Shot por tarea, o secuencias de $N$ tareas).
* **Consulta en $O(1)$**: Al recibir la acción `action` (CPU o GPU), se consulta la fila correspondiente en la tabla de métricas indexada en RAM y se extraen `Energy_J`, `Time_sec`, `EDP`, `GFLOPS` y `Avg_Power_W`.

---

## 3. Estructura del Vector de Observación (22D)

| Rango de Índices | Característica | Tipo de Codificación | Valores / Regla de Transformación | Descripción |
| :---: | :--- | :---: | :--- | :--- |
| **`[0..1]`** | **Tipo de Kernel** | One-Hot (2D) | `GEMM` $\rightarrow [1, 0]$<br>`FFT` $\rightarrow [0, 1]$ | Identificador del tipo de algoritmo a ejecutar. |
| **`[2]`** | **Dimensión 1** | Continuo $\log_2$ | $\frac{\log_2(\max(1, M \text{ ó } N_x))}{26.0}$ | Filas $M$ (GEMM) o dimensión $N_x$ (FFT). |
| **`[3]`** | **Dimensión 2** | Continuo $\log_2$ | $\frac{\log_2(N)}{26.0}$ (GEMM)<br>$\frac{\log_2(N_y)}{26.0}$ si $N_y > 0$ else $0.0$ (FFT) | Columnas $N$ (GEMM) o dimensión $N_y$ en FFT 2D/3D. |
| **`[4]`** | **Dimensión 3** | Continuo $\log_2$ | $\frac{\log_2(K)}{26.0}$ (GEMM)<br>$\frac{\log_2(N_z)}{26.0}$ si $N_z > 0$ else $0.0$ (FFT) | Dimensión común $K$ (GEMM) o dimensión $N_z$ en FFT 3D. |
| **`[5]`** | **Batch Size** | Continuo $\log_2$ | $0.0$ (GEMM)<br>$\frac{\log_2(\max(1, \text{Batch}))}{16.0}$ (FFT) | Lote de transformadas FFT. |
| **`[6..9]`** | **Precisión** | One-Hot (4D) | `S` (Float32) $\rightarrow [1, 0, 0, 0]$<br>`D` (Float64) $\rightarrow [0, 1, 0, 0]$<br>`C` (Complex64) $\rightarrow [0, 0, 1, 0]$<br>`Z` (Complex128) $\rightarrow [0, 0, 0, 1]$ | Precisión aritmética compartida. |
| **`[10..12]`**| **GEMM OpA** | One-Hot (3D) | `N` $\rightarrow [1, 0, 0]$, `T` $\rightarrow [0, 1, 0]$, `C` $\rightarrow [0, 0, 1]$<br>*(En FFT se llena con $[0, 0, 0]$)* | Operación sobre matriz $A$. |
| **`[13..15]`**| **GEMM OpB** | One-Hot (3D) | `N` $\rightarrow [1, 0, 0]$, `T` $\rightarrow [0, 1, 0]$, `C` $\rightarrow [0, 0, 1]$<br>*(En FFT se llena con $[0, 0, 0]$)* | Operación sobre matriz $B$. |
| **`[16..17]`**| **FFT Dominio** | One-Hot (2D) | `C2C` $\rightarrow [1, 0]$, `R2C` $\rightarrow [0, 1]$<br>*(En GEMM se llena con $[0, 0]$)* | Complejo a Complejo o Real a Complejo. |
| **`[18..19]`**| **FFT Dirección**| One-Hot (2D) | `Forward (F)` $\rightarrow [1, 0]$, `Inverse (I)` $\rightarrow [0, 1]$<br>*(En GEMM se llena con $[0, 0]$)* | Dirección de la transformada FFT. |
| **`[20..21]`**| **FFT Layout** | One-Hot (2D) | `In-Place (I)` $\rightarrow [1, 0]$, `Out-of-Place (O)` $\rightarrow [0, 1]$<br>*(En GEMM se llena con $[0, 0]$)* | Ubicación en memoria de los buffers. |
