import csv
import math
import os
from collections import defaultdict

def generar_vector_22d(task_info):
    """
    Toma un diccionario con los parámetros de la tarea (GEMM o FFT)
    y retorna el vector de 22 dimensiones One-Hot según DISENO_ENTORNO.md.
    """
    obs = [0.0] * 22
    tipo = task_info['tipo']
    max_log2_dim = 24.0
    max_log2_batch = 16.0

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
        
    prec = str(task_info.get('Precision', 'S')).upper()
    if prec == 'S': obs[6] = 1.0
    elif prec == 'D': obs[7] = 1.0
    elif prec == 'C': obs[8] = 1.0
    elif prec == 'Z': obs[9] = 1.0
    
    return obs

def procesar_csvs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    
    csv_gemm = os.path.join(project_root, "benchmark_results.csv")
    csv_fft = os.path.join(project_root, "fft_benchmark_results.csv")
    csv_salida = os.path.join(base_dir, "dataset_rl.csv")

    gemm_tasks = defaultdict(dict)
    fft_tasks = defaultdict(dict)

    # Procesar GEMM
    if os.path.exists(csv_gemm):
        print(f"Procesando {csv_gemm}...")
        with open(csv_gemm, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # La clave identifica unívocamente la configuración de la tarea
                key = (row['M'], row['N'], row['K'], row['Precision'], row['OpA'], row['OpB'])
                device = row['Device'].lower()
                gemm_tasks[key][device] = row
    else:
        print(f"Advertencia: No se encontró {csv_gemm}")

    # Procesar FFT
    if os.path.exists(csv_fft):
        print(f"Procesando {csv_fft}...")
        with open(csv_fft, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['Nx'], row['Ny'], row['Nz'], row['Batch'], row['Precision'], row['Domain'], row['Direction'], row['Layout'])
                device = row['Device'].lower()
                fft_tasks[key][device] = row
    else:
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
        "Layout_I", "Layout_O"
    ]
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
            obs = generar_vector_22d(task_info)
            
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
            obs = generar_vector_22d(task_info)
            
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
    procesar_csvs()
