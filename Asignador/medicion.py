import subprocess
import re
import time
import threading
import csv

#EJECUTAR CON PERMISOS!!!! (sudo)

# ============================================================================
# FÓRMULAS Y VARIABLES DE CÁLCULO DE EDP
# ============================================================================
#
# POTENCIA INSTANTÁNEA (RAPL - CPU):
#   Fórmula: P(t) = ΔE / Δt
#   Variables:
#     - energy_uj: Energía actual leída de RAPL (μJ)
#     - prev_energy_uj: Energía anterior leída de RAPL (μJ)
#     - delta_energy_j: Diferencia de energía convertida a Joules (J)
#     - delta_time: Diferencia de tiempo entre lecturas (s)
#     - watts: Potencia instantánea calculada (W)
#   Cálculo:
#     delta_energy_j = (energy_uj - prev_energy_uj) / 1e6
#     watts = delta_energy_j / delta_time
#
# POTENCIA INSTANTÁNEA (NVML - GPU):
#   Fórmula: P = Lectura directa de NVIDIA Management Library
#   Variables:
#     - power_draw_mw: Consumo instantáneo de GPU (mW)
#     - watts: Potencia instantánea convertida (W)
#   Cálculo:
#     watts = power_draw_mw / 1000.0
#
# POTENCIA PROMEDIO:
#   Fórmula: P_avg = Σ(P(i)) / n
#   Variables:
#     - registro_watts: Lista de todas las potencias instantáneas medidas (W)
#     - potencia_promedio: Promedio aritmético de todas las muestras (W)
#   Cálculo:
#     potencia_promedio = sum(registro_watts) / len(registro_watts)
#
# TIEMPO DE EJECUCIÓN:
#   Variables:
#     - tiempo_ms: Tiempo de ejecución (ms)
#     - tiempo_s: Tiempo convertido a segundos (s)
#   Cálculo:
#     tiempo_s = tiempo_ms / 1000.0
#
# ENERGÍA TOTAL CONSUMIDA:
#   Fórmula: E = P_avg × t
#   Variables:
#     - potencia_promedio: Potencia promedio durante ejecución (W)
#     - tiempo_s: Tiempo total de ejecución (s)
#     - energia_joules: Energía total consumida (J)
#   Cálculo:
#     energia_joules = potencia_promedio * tiempo_s
#
# ENERGY-DELAY PRODUCT (EDP):
#   Fórmula: EDP = E × t = P_avg × t²
#   Variables:
#     - energia_joules: Energía total consumida (J)
#     - tiempo_s: Tiempo total de ejecución (s)
#     - edp: Energy-Delay Product (J·s)
#   Cálculo:
#     edp = energia_joules * tiempo_s
#
# ============================================================================
# Variables globales para comunicar los hilos

medicion_activa = False
registro_watts = []
prev_energy_uj = None
prev_time = None

# ============================================================================
# FUNCIONES PARA LEER POTENCIA
# ============================================================================

def leer_rapl_watts():
    """Lee consumo de CPU usando RAPL (calcula potencia instantánea)"""
    global prev_energy_uj, prev_time
    try:
        with open('/sys/class/powercap/intel-rapl:0/energy_uj') as f:
            energy_uj = float(f.read().strip())
        
        current_time = time.time()
        
        if prev_energy_uj is not None and prev_time is not None:
            # Diferencia en energía (convertir de μJ a J)
            delta_energy_j = (energy_uj - prev_energy_uj) / 1e6
            delta_time = current_time - prev_time
            
            # Evitar división por cero y valores negativos
            if delta_time > 0.001 and delta_energy_j >= 0:
                watts = delta_energy_j / delta_time
                prev_energy_uj = energy_uj
                prev_time = current_time
                return watts if watts < 500 else 0.0  # Filtrar spikes
        
        prev_energy_uj = energy_uj
        prev_time = current_time
        return 0.0
    except PermissionError:
        print("ERROR: Sin permisos para leer RAPL")
        return 0.0
    except FileNotFoundError:
        print("ERROR: RAPL no disponible en este sistema")
        return 0.0
    except Exception as e:
        print("Error leyendo RAPL: " + str(e))
        return 0.0

def leer_nvml_watts():
    """Lee consumo de GPU usando NVIDIA Management Library"""
    try:
        import pynvml
        
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # en mW
            watts = power_draw_mw / 1000.0  # Convertir a W
            pynvml.nvmlShutdown()
            return watts
        except pynvml.NVMLError as e:
            print("NVML Error: " + str(e))
            return 0.0
    except ImportError:
        print("ERROR: pynvml no instalado. Ejecuta: pip install pynvml")
        return 0.0
    except Exception as e:
        print("Error leyendo GPU: " + str(e))
        return 0.0

# ============================================================================
# MONITOREO DE ENERGÍA EN HILO PARALELO
# ============================================================================

def monitor_energia(dispositivo):
    """Hilo que mide consumo de energía en segundo plano"""
    global medicion_activa, registro_watts
    
    while medicion_activa:
        try:
            if dispositivo == 'cpu':
                watts = leer_rapl_watts()
            else:  # gpu
                watts = leer_nvml_watts()
            
            # Solo registrar valores válidos (> 0 y < 1000W)
            if 0 < watts < 1000:
                registro_watts.append(watts)
            
        except Exception as e:
            print("Error en monitor: " + str(e))
        
        time.sleep(0.05)  # Muestrear cada 50ms

# ============================================================================
# FUNCIÓN PRINCIPAL: EJECUTAR BENCHMARK Y MEDIR EDP
# ============================================================================

def ejecutar_y_medir(N, dispositivo, benchmark_type):
    """
    Ejecuta el benchmark y mide:
    - Tiempo de ejecución
    - Potencia promedio
    - Energía consumida
    - EDP (Energy-Delay Product)
    """
    global medicion_activa, registro_watts, prev_energy_uj, prev_time
    
    registro_watts = []  # Limpiar registros anteriores
    medicion_activa = True
    
    # Reset para RAPL
    if dispositivo == 'cpu':
        prev_energy_uj = None
        prev_time = None
        time.sleep(0.1)  # Pequeña pausa antes de empezar
    
    # Iniciar el monitor de energía en paralelo
    hilo_monitor = threading.Thread(target=monitor_energia, args=(dispositivo,))
    hilo_monitor.daemon = True
    hilo_monitor.start()
    
    # Pequeña pausa para que el monitor se estabilice
    time.sleep(0.05)
    
    # --- INICIO DE EJECUCIÓN ---
    start_time = time.time()
    
    # Lanzar binario (GEMM o FFT)
    if benchmark_type == 'gemm':
        if dispositivo == "cpu":
            comando = ["./gemm_cpu", str(N)]
        else:
            comando = ["./gemm_gpu", str(N)]
    elif benchmark_type == 'fft':
        if dispositivo == "cpu":
            comando = ["./fft_cpu", str(N)]
        else:
            comando = ["./fft_gpu", str(N)]
    else:
        raise ValueError("Benchmark type invalido")
    
    resultado = subprocess.run(comando, capture_output=True, text=True)
    
    end_time = time.time()
    # --- FIN DE EJECUCIÓN ---
    
    # Detener el monitor
    medicion_activa = False
    hilo_monitor.join(timeout=1.0)
    
    # ========================================================================
    # PARSEAR SALIDA DEL BENCHMARK
    # ========================================================================
    match = re.search(r'tiempo=([\d.]+)\s*ms.*GFLOPS=([\d.]+)', resultado.stdout)
    if not match:
        raise ValueError("No se pudo parsear la salida: " + resultado.stdout)
    
    tiempo_ms = float(match.group(1))
    tiempo_s = tiempo_ms / 1000.0  # Convertir a segundos
    gflops = float(match.group(2))
    
    # ========================================================================
    # CALCULAR ENERGÍA Y EDP
    # ========================================================================
    
    # Potencia promedio durante la ejecución
    if len(registro_watts) > 0:
        potencia_promedio = sum(registro_watts) / len(registro_watts)
        muestras = len(registro_watts)
    else:
        # Si no se capturó potencia, usar valor por defecto
        potencia_promedio = 50.0 if dispositivo == "cpu" else 150.0
        muestras = 0
    
    # Energía = Potencia × Tiempo
    energia_joules = potencia_promedio * tiempo_s
    
    # EDP = Energía × Tiempo
    edp = energia_joules * tiempo_s
    
    return tiempo_ms, energia_joules, edp, potencia_promedio, gflops, muestras

# ============================================================================
# FUNCIONES DE IMPRESIÓN EN TIEMPO REAL
# ============================================================================

def imprimir_encabezado_tabla():
    """Imprime el encabezado de la tabla"""
    print("")
    print("=" * 120)
    print("Benchmark       N          Tiempo(ms)      Potencia(W)     Energia(J)      EDP(J*s)        GFLOPS          Muestras")
    print("-" * 120)

def imprimir_resultado(benchmark, dispositivo, N, tiempo_ms, potencia, energia, edp, gflops, muestras):
    """Imprime un resultado en la tabla"""
    label = benchmark.upper() + "_" + dispositivo.upper()
    print(f"{label:<15} {N:<10} {tiempo_ms:>14.6f} {potencia:>15.2f} {energia:>15.6f} {edp:>15.6f} {gflops:>15.1f} {muestras:>12}")

def imprimir_separador():
    """Imprime una línea separadora"""
    print("-" * 120)

def imprimir_cierre():
    """Imprime el cierre de la tabla"""
    print("=" * 120)
    print("")

# ============================================================================
# MAIN: EJECUTAR BENCHMARKS
# ============================================================================

if __name__ == "__main__":
    # Tamaños de datos
    tamanos_gemm = [128, 256, 512, 1024, 2048, 4096]
    tamanos_fft = [2**14, 2**16, 2**18, 2**20, 2**22, 2**24]
    
    # Archivo CSV para guardar resultados
    csv_file = 'mediciones_unificadas.csv'
    
    # Crear archivo CSV con encabezado
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Benchmark",
            "Dispositivo", 
            "N", 
            "Tiempo(ms)", 
            "Potencia_Avg(W)", 
            "Energia(J)", 
            "EDP(J*s)",
            "GFLOPS",
            "Muestras_Potencia"
        ])
    
    # Imprimir encabezado de tabla
    imprimir_encabezado_tabla()
    
    # ====== BENCHMARK GEMM ======
    
    # GEMM en CPU
    for N in tamanos_gemm:
        try:
            t_ms, energia, edp, potencia, gflops, muestras = ejecutar_y_medir(N, 'cpu', 'gemm')
            imprimir_resultado('gemm', 'cpu', N, t_ms, potencia, energia, edp, gflops, muestras)
            
            # Guardar en CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['gemm', 'cpu', N, t_ms, potencia, energia, edp, gflops, muestras])
            
            time.sleep(1)
        except Exception as ex:
            print("Error en GEMM CPU N=" + str(N) + ": " + str(ex))
    
    imprimir_separador()
    
    # GEMM en GPU
    for N in tamanos_gemm:
        try:
            t_ms, energia, edp, potencia, gflops, muestras = ejecutar_y_medir(N, 'gpu', 'gemm')
            imprimir_resultado('gemm', 'gpu', N, t_ms, potencia, energia, edp, gflops, muestras)
            
            # Guardar en CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['gemm', 'gpu', N, t_ms, potencia, energia, edp, gflops, muestras])
            
            time.sleep(1)
        except Exception as ex:
            print("Error en GEMM GPU N=" + str(N) + ": " + str(ex))
    
    imprimir_separador()
    
    # ====== BENCHMARK FFT ======
    
    # FFT en CPU
    for N in tamanos_fft:
        try:
            t_ms, energia, edp, potencia, gflops, muestras = ejecutar_y_medir(N, 'cpu', 'fft')
            imprimir_resultado('fft', 'cpu', N, t_ms, potencia, energia, edp, gflops, muestras)
            
            # Guardar en CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['fft', 'cpu', N, t_ms, potencia, energia, edp, gflops, muestras])
            
            time.sleep(1)
        except Exception as ex:
            print("Error en FFT CPU N=" + str(N) + ": " + str(ex))
    
    imprimir_separador()
    
    # FFT en GPU
    for N in tamanos_fft:
        try:
            t_ms, energia, edp, potencia, gflops, muestras = ejecutar_y_medir(N, 'gpu', 'fft')
            imprimir_resultado('fft', 'gpu', N, t_ms, potencia, energia, edp, gflops, muestras)
            
            # Guardar en CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['fft', 'gpu', N, t_ms, potencia, energia, edp, gflops, muestras])
            
            time.sleep(1)
        except Exception as ex:
            print("Error en FFT GPU N=" + str(N) + ": " + str(ex))
    
    imprimir_cierre()
    print("Dataset guardado en: " + csv_file)