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
Espacio continuo acotado: `spaces.Box(low=0.0, high=1.0, shape=(23,), dtype=np.float32)`.

Vector unificado de **23 dimensiones** con **One-Hot Encoding** para variables categóricas y **escalado $\log_2$ normalizado** para dimensiones cuantitativas.

### 2.3. Función de Recompensa ($\mathcal{R}$)
$$R(s, a) = \frac{\text{EDP}_{max}(s) - \text{EDP}(s, a)}{\text{EDP}_{max}(s)}$$

Donde $\text{EDP}_{max}(s) = \max(\text{EDP}_{cpu}(s), \text{EDP}_{gpu}(s))$ es el EDP del **peor**
dispositivo para esa misma tarea. El resultado queda acotado en $[0, 1]$: vale $0$ al elegir el
peor dispositivo y $1 - \text{EDP}_{min}/\text{EDP}_{max}$ al elegir el mejor. Maximizar
$R(s,a)$ sigue equivaliendo estrictamente a minimizar el EDP medido en el CSV, porque
$\text{EDP}_{max}(s)$ es constante dentro de un mismo estado.

#### Por qué se reemplazó $R = -\ln(\text{EDP} + \epsilon)$

La fórmula anterior usaba el valor **absoluto** del EDP. En el dataset de `paccaA100` ese valor
abarca ~32 nats entre la tarea más pequeña y la más grande — una amplitud que depende del tamaño
de $N$, no de si la decisión CPU/GPU fue correcta. Como la red comparte pesos entre todas las
tareas y, con $\gamma = 0$, el objetivo de regresión de $Q$ es el reward inmediato, esa variación
de escala ahogaba la señal útil: la diferencia de recompensa entre acertar y fallar es de solo
$0.4$–$3.3$ nats. El resultado medido fue un agente que colapsaba a "siempre GPU" ($99.91\%$ de
sus decisiones) y se estancaba en $\approx 93\%$ de precisión, sin mejorar al triplicar los pasos
de entrenamiento.

La normalización por $\text{EDP}_{max}(s)$ corrige dos cosas a la vez:

1. **Elimina la varianza de escala entre tareas.** Al dividir por una cantidad de la propia tarea,
   el reward vive siempre en $[0,1]$ sin importar si la tarea tarda microsegundos o segundos.
2. **Comprime los extremos.** Por ser una función saturante del cociente (y no logarítmica), una
   tarea donde la GPU gana por $20{,}000\times$ deja de pesar desproporcionadamente más que una
   donde gana por $1.5\times$.

Efecto medido sobre la razón entre el margen de la clase mayoritaria y el de la minoritaria
(mediana del margen de recompensa entre acertar y fallar):

| Dataset | $-\ln(\text{EDP})$ | Normalizada |
| :--- | :---: | :---: |
| FFT | $8.05\times$ | $2.85\times$ |
| GEMM | $0.53\times$ | $0.82\times$ |
| Mixto | $0.98\times$ | $1.00\times$ |

> **Compatibilidad:** el cambio de escala de la recompensa invalida los modelos entrenados con la
> fórmula anterior y las curvas de `ep_rew_mean` en TensorBoard no son comparables entre ambas.
> La métrica `metricas_personalizadas/precision` sí sigue siendo comparable, porque se calcula
> contra $\arg\min(\text{EDP}_{cpu}, \text{EDP}_{gpu})$ y no depende de la recompensa.

### 2.4. Política de Episodios One-Shot
* **Episodio**: Configurable mediante `tasks_per_episode` (por defecto $1$ para decisiones puras One-Shot por tarea, o secuencias de $N$ tareas).
* **Consulta en $O(1)$**: Al recibir la acción `action` (CPU o GPU), se consulta la fila correspondiente en la tabla de métricas indexada en RAM y se extraen `Energy_J`, `Time_sec`, `EDP`, `GFLOPS` y `Avg_Power_W`.

---

## 3. Estructura del Vector de Observación (23D)

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
| **`[22]`** | **FFT Estructura de Radix** | Continuo $\log_2$ | $\frac{\log_2(\max\text{-primo}(N_x, N_y, N_z))}{26.0}$<br>*(En GEMM se llena con $0.0$)* | Mayor factor primo de las dimensiones. Determina si FFTW usa codelet nativo o cae a Rader/Bluestein. |


### 3.1. Nota sobre `[22]` — Estructura de Radix

Se añadió tras auditar el barrido FFT de `paccaA100` (job 6924, 25.184 filas). El
rendimiento de CPU relativo a un tamaño 5-smooth de la misma octava decae de forma
**monótona** con el mayor factor primo de las dimensiones:

| Mayor factor primo | GFLOPS relativos CPU | GFLOPS relativos GPU |
| :--- | :---: | :---: |
| $\le 13$ (codelet nativo FFTW) | 1.00 | 1.00 |
| 17 – 31 (radix genérico) | 0.84 | 1.02 |
| 37 – 127 (Rader/Bluestein) | 0.49 | 1.02 |
| $> 127$ | **0.15** | 0.89 |

cuFFT es prácticamente insensible, así que esta variable **es** la frontera CPU/GPU en
FFT. Sin ella, dos tamaños separados un 2–4 % (indistinguibles en `obs[2]`) pueden tener
EDP de CPU que difieren por factores de hasta $10^4$, y el agente no dispone de ninguna
señal para separarlos: la recompensa le resulta ruido.

Se codifica **continua y no como one-hot de clase** porque la penalización sigue creciendo
dentro de la clase `bluestein` (0.49 frente a 0.15), gradiente que un one-hot descartaría;
además cuesta 1 dimensión en lugar de 4.

> **Compatibilidad:** el cambio de 22D a 23D invalida los checkpoints entrenados con el
> espacio anterior. Hay que re-entrenar. `codificador_csv.feature_radix()` deriva el valor
> de `Nx/Ny/Nz`, así que los CSV anteriores a la columna `Radix_Class` siguen siendo
> utilizables sin regenerarlos.
