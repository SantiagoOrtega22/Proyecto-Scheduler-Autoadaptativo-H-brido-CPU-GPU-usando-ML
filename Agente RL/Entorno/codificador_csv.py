import csv
import math
import os
from collections import defaultdict

# Techos de normalizacion logaritmica de las dimensiones. Deben cubrir el tamaño
# maximo que el generador de cargas puede emitir, o el feature normalizado se
# saldria del rango [0,1] que declara observation_space en gym.py.
# MAX_LOG2_DIM = 26 corresponde a fft_max_n = 2**26 en benchmark_runner.py
# (el techo de FFT 1D); GEMM tope en 2**14, muy por debajo.
# Constante unica: graficar_frontera.py la importa para la transformacion inversa.
MAX_LOG2_DIM = 26.0
MAX_LOG2_BATCH = 16.0


def max_prime_factor(n):
    """Devuelve el mayor factor primo de n (1 para n <= 1).

    Args:
        n (int): Entero positivo a factorizar.

    Returns:
        int: El mayor factor primo de n, o 1 si n <= 1.
    """
    n = int(n)
    if n <= 1:
        return 1
    mayor = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            mayor = d
            n //= d
        d += 1 if d == 2 else 2
    return max(mayor, n)


def feature_radix(nx, ny=0, nz=0):
    """Codifica la hostilidad de una forma de FFT para la librería, en [0, 1].

    FFTW resuelve a velocidad plena las longitudes cuyos factores primos tienen
    codelet nativo (<= 13) y cae a Rader/Bluestein en cuanto aparece un primo
    mayor. En el barrido de paccaA100 el rendimiento relativo de CPU decae de
    forma monótona con el mayor factor primo (1.00 para <= 13, 0.84 en 17..31,
    0.49 en 37..127, 0.15 por encima), mientras cuFFT se mantiene plano. Esa
    asimetría es la que decide la frontera CPU/GPU en FFT.

    Se codifica de forma CONTINUA, no como one-hot de clase: la penalización es
    monótona en log2 del mayor primo y sigue creciendo dentro de la clase
    "bluestein" (0.49 frente a 0.15), gradiente que un one-hot descartaría.
    Se normaliza con MAX_LOG2_DIM porque el mayor primo nunca excede la propia
    dimensión, así que el cociente queda acotado a [0, 1] como exige gym.py.

    Args:
        nx (int): Dimensión Nx de la transformada.
        ny (int): Dimensión Ny, o 0 si la FFT es 1D.
        nz (int): Dimensión Nz, o 0 si la FFT no es 3D.

    Returns:
        float: log2(mayor factor primo) / MAX_LOG2_DIM, en [0, 1]. El minimo
            alcanzable por una FFT real es 1/26 ~ 0.0385 (potencias de 2, mayor
            primo = 2); solo devuelve 0.0 exacto si no hay dimension valida, que
            es tambien el valor que toma el feature en tareas GEMM.
    """
    dims = [int(d) for d in (nx, ny, nz) if int(d) > 0]
    if not dims:
        return 0.0
    mayor = max(max_prime_factor(d) for d in dims)
    return math.log2(mayor) / MAX_LOG2_DIM if mayor > 1 else 0.0


def generar_vector_23d(task_info):
    """
    Toma un diccionario con los parámetros de la tarea (GEMM o FFT)
    y retorna el vector de 23 dimensiones One-Hot según DISENO_ENTORNO.md.
    """
    obs = [0.0] * 23
    tipo = task_info['tipo']
    max_log2_dim = MAX_LOG2_DIM
    max_log2_batch = MAX_LOG2_BATCH

    if tipo == 'GEMM':
        obs[0] = 1.0  # is_GEMM
        obs[2] = math.log2(max(1, int(task_info['M']))) / max_log2_dim
        obs[3] = math.log2(max(1, int(task_info['N']))) / max_log2_dim
        obs[4] = math.log2(max(1, int(task_info['K']))) / max_log2_dim
        obs[5] = 0.0  # Batch no aplica en GEMM
        
        opa = str(task_info.get('OpA', 'N')).upper()
        if opa == 'N': obs[10] = 1.0
        elif opa == 'T': obs[11] = 1.0
        elif opa == 'C': obs[12] = 1.0
        
        opb = str(task_info.get('OpB', 'N')).upper()
        if opb == 'N': obs[13] = 1.0
        elif opb == 'T': obs[14] = 1.0
        elif opb == 'C': obs[15] = 1.0
    else:
        obs[1] = 1.0  # is_FFT
        obs[2] = math.log2(max(1, int(task_info['Nx']))) / max_log2_dim
        
        ny = int(task_info.get('Ny', 0))
        nz = int(task_info.get('Nz', 0))
        obs[3] = math.log2(ny) / max_log2_dim if ny > 0 else 0.0
        obs[4] = math.log2(nz) / max_log2_dim if nz > 0 else 0.0
        
        batch = int(task_info.get('Batch', 1))
        obs[5] = math.log2(max(1, batch)) / max_log2_batch
        
        dom = str(task_info.get('Domain', 'C2C')).upper()
        if dom == 'C2C': obs[16] = 1.0
        elif dom == 'R2C': obs[17] = 1.0
        
        direction = str(task_info.get('Direction', 'F')).upper()
        if direction == 'F': obs[18] = 1.0
        elif direction == 'I': obs[19] = 1.0
        
        layout = str(task_info.get('Layout', 'I')).upper()
        if layout == 'I': obs[20] = 1.0
        elif layout == 'O': obs[21] = 1.0

        # Estructura de radix. Se deriva de las dimensiones en vez de leer la
        # columna Radix_Class del CSV, para que el codificador siga funcionando
        # con los datasets generados antes de que esa columna existiera.
        obs[22] = feature_radix(task_info['Nx'], ny, nz)
        
    prec = str(task_info.get('Precision', 'S')).upper()
    if prec == 'S': obs[6] = 1.0
    elif prec == 'D': obs[7] = 1.0
    elif prec == 'C': obs[8] = 1.0
    elif prec == 'Z': obs[9] = 1.0
    
    return obs

def procesar_csvs(csv_gemm=None, csv_fft=None, csv_salida=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    
    # Si no se pasó ninguno de los dos, usar los defaults originales
    if csv_gemm is None and csv_fft is None:
        csv_gemm = os.path.join(project_root, "benchmark_results.csv")
        csv_fft = os.path.join(project_root, "fft_benchmark_results.csv")
    if csv_salida is None:
        csv_salida = os.path.join(base_dir, "dataset_rl.csv")

    gemm_tasks = defaultdict(dict)
    fft_tasks = defaultdict(dict)

    # Procesar GEMM
    if csv_gemm and os.path.exists(csv_gemm):
        print(f"Procesando {csv_gemm}...")
        with open(csv_gemm, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filtrar rápido en caso de que le hayan pasado un CSV que no es de GEMM
                if 'M' not in row:
                    continue
                # La clave identifica unívocamente la configuración de la tarea
                key = (row.get('M', 1), row.get('N', 1), row.get('K', 1), row.get('Precision', 'S'), row.get('OpA', 'N'), row.get('OpB', 'N'))
                device = row.get('Device', 'cpu').lower()
                gemm_tasks[key][device] = row
    elif csv_gemm:
        print(f"Advertencia: No se encontró {csv_gemm}")

    # Procesar FFT
    if csv_fft and os.path.exists(csv_fft):
        print(f"Procesando {csv_fft}...")
        with open(csv_fft, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Nx' not in row:
                    continue
                key = (row.get('Nx', 1), row.get('Ny', 0), row.get('Nz', 0), row.get('Batch', 1), row.get('Precision', 'S'), row.get('Domain', 'C2C'), row.get('Direction', 'F'), row.get('Layout', 'I'))
                device = row.get('Device', 'cpu').lower()
                fft_tasks[key][device] = row
    elif csv_fft:
        print(f"Advertencia: No se encontró {csv_fft}")

    # Escribir salida combinada
    print(f"Escribiendo dataset unificado en {csv_salida}...")
    
    # Nombres descriptivos semánticos en lugar de obs_0...obs_21
    columnas_obs = [
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
    # El volcado se arma con enumerate(columnas_obs), que trunca en silencio si la
    # lista se queda corta respecto al vector. Se verifica explicitamente para que
    # una futura dimension nueva falle de inmediato en vez de perderse sin aviso.
    assert len(columnas_obs) == len(generar_vector_23d(
        {"tipo": "FFT", "Nx": 2, "Ny": 0, "Nz": 0}
    )), "columnas_obs no coincide con la longitud del vector de observacion"
    columnas_metricas = ["cpu_time", "cpu_energy", "cpu_edp", "gpu_time", "gpu_energy", "gpu_edp"]
    
    with open(csv_salida, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=columnas_obs + columnas_metricas)
        writer.writeheader()
        
        filas_escritas = 0

        # Exportar tareas GEMM
        for key, devices in gemm_tasks.items():
            if 'cpu' not in devices or 'gpu' not in devices:
                continue
                
            task_info = {
                'tipo': 'GEMM',
                'M': key[0], 'N': key[1], 'K': key[2],
                'Precision': key[3], 'OpA': key[4], 'OpB': key[5]
            }
            obs = generar_vector_23d(task_info)
            
            fila_salida = {col: obs[i] for i, col in enumerate(columnas_obs)}
            fila_salida.update({
                "cpu_time": devices['cpu']['Time_sec'],
                "cpu_energy": devices['cpu']['Energy_J'],
                "cpu_edp": devices['cpu']['EDP'],
                "gpu_time": devices['gpu']['Time_sec'],
                "gpu_energy": devices['gpu']['Energy_J'],
                "gpu_edp": devices['gpu']['EDP'],
            })
            writer.writerow(fila_salida)
            filas_escritas += 1

        # Exportar tareas FFT
        for key, devices in fft_tasks.items():
            if 'cpu' not in devices or 'gpu' not in devices:
                continue
                
            task_info = {
                'tipo': 'FFT',
                'Nx': key[0], 'Ny': key[1], 'Nz': key[2], 'Batch': key[3],
                'Precision': key[4], 'Domain': key[5], 'Direction': key[6], 'Layout': key[7]
            }
            obs = generar_vector_23d(task_info)
            
            fila_salida = {col: obs[i] for i, col in enumerate(columnas_obs)}
            fila_salida.update({
                "cpu_time": devices['cpu']['Time_sec'],
                "cpu_energy": devices['cpu']['Energy_J'],
                "cpu_edp": devices['cpu']['EDP'],
                "gpu_time": devices['gpu']['Time_sec'],
                "gpu_energy": devices['gpu']['Energy_J'],
                "gpu_edp": devices['gpu']['EDP'],
            })
            writer.writerow(fila_salida)
            filas_escritas += 1

    print(f"Proceso completado. Se escribieron {filas_escritas} tareas válidas emparejadas en el dataset.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convierte resultados de Benchmark CSV a un Dataset codificado para RL (One-Hot 22D).")
    parser.add_argument("--gemm", type=str, help="Ruta al archivo CSV de resultados GEMM", default=None)
    parser.add_argument("--fft", type=str, help="Ruta al archivo CSV de resultados FFT", default=None)
    parser.add_argument("--out", type=str, help="Ruta al archivo CSV codificado de salida", default=None)
    
    args = parser.parse_args()
    procesar_csvs(csv_gemm=args.gemm, csv_fft=args.fft, csv_salida=args.out)
