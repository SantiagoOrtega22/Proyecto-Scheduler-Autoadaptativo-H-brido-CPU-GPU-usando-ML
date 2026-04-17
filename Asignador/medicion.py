import subprocess
import re
import time
import threading
import csv
import os

#EJECUTAR CON PERMISOS!!!! (SUDO)
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
#     - muestras: Número total de muestras capturadas (-)
#   Cálculo:
#     potencia_promedio = sum(registro_watts) / len(registro_watts)
#     muestras = len(registro_watts)
#
# TIEMPO DE EJECUCIÓN:
#   Variables:
#     - tiempo_ms: Tiempo de ejecución del programa GEMM (ms)
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
# THROUGHPUT:
#   Fórmula: GFLOPS = (2 × N³) / (tiempo × 1e6)
#   Variables:
#     - N: Tamaño de matriz cuadrada (-)
#     - gflops: Giga operaciones por segundo (GFLOPS)
#   Nota: Se extrae del output del programa GEMM
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
        print("❌ ERROR: Sin permisos para leer RAPL")
        return 0.0
    except FileNotFoundError:
        print("❌ ERROR: RAPL no disponible en este sistema")
        return 0.0
    except Exception as e:
        print(f"⚠️  Error leyendo RAPL: {e}")
        return 0.0

def leer_nvml_watts():
    """Lee consumo de GPU usando NVIDIA Management Library"""
    try:
        # Importar dentro de la función para evitar problemas con sudo
        import pynvml
        
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # en mW
            watts = power_draw_mw / 1000.0  # Convertir a W
            pynvml.nvmlShutdown()
            return watts
        except pynvml.NVMLError as e:
            print(f"⚠️  NVML Error: {e}")
            return 0.0
    except ImportError:
        print("❌ pynvml no instalado. Ejecuta: pip install pynvml")
        return 0.0
    except Exception as e:
        print(f"⚠️  Error leyendo GPU: {e}")
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
            print(f"Error en monitor: {e}")
        
        time.sleep(0.05)  # Muestrear cada 50ms

# ============================================================================
# FUNCIÓN PRINCIPAL: EJECUTAR BENCHMARK Y MEDIR EDP
# ============================================================================

def ejecutar_y_medir(N, dispositivo):
    """
    Ejecuta el benchmark GEMM y mide:
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
    
    # Lanzar binario GEMM (CPU o GPU)
    if dispositivo == "cpu":
        comando = ["./gemm_cpu", str(N)]
    elif dispositivo == "gpu":
        comando = ["./gemm_gpu", str(N)]
    else:
        raise ValueError("Dispositivo no válido")
    
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
        raise ValueError(f"No se pudo parsear la salida: {resultado.stdout}")
    
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
        print(f"   ⚠️  Sin datos de potencia (0 muestras), usando valor por defecto: {potencia_promedio}W")
    
    # Energía = Potencia × Tiempo
    energia_joules = potencia_promedio * tiempo_s
    
    # EDP = Energía × Tiempo
    edp = energia_joules * tiempo_s
    
    return tiempo_ms, energia_joules, edp, potencia_promedio, gflops, muestras

# ============================================================================
# MAIN: EJECUTAR BENCHMARKS Y GUARDAR DATASET
# ============================================================================

if __name__ == "__main__":
    # Tamaños de matrices a evaluar
    tamanos_N = [128, 256, 512, 1024, 2048, 4096]
    
    # Archivo CSV para guardar resultados
    csv_file = 'dataset_gemm_edp.csv'
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Dispositivo", 
            "N", 
            "Tiempo(ms)", 
            "Potencia_Avg(W)", 
            "Energia(J)", 
            "EDP(J*s)",
            "GFLOPS",
            "Muestras_Potencia"
        ])
        
        # ====== BENCHMARKS EN CPU ======
        print("="*70)
        print("--- Iniciando Benchmarks en CPU ---")
        print("="*70)
        for N in tamanos_N:
            try:
                t_ms, e, edp, p_avg, gflops, muestras = ejecutar_y_medir(N, 'cpu')
                print(f"CPU | N={N:5d} | T:{t_ms:8.3f}ms | P:{p_avg:7.2f}W | E:{e:8.4f}J | EDP:{edp:10.6f} | GFLOPS:{gflops:7.1f} | Muestras:{muestras}")
                writer.writerow(['cpu', N, t_ms, p_avg, e, edp, gflops, muestras])
                file.flush()
                time.sleep(1)  # Pausa para enfriar hardware
            except Exception as ex:
                print(f"❌ Error en CPU N={N}: {ex}")
        
        # ====== BENCHMARKS EN GPU ======
        print("\n" + "="*70)
        print("--- Iniciando Benchmarks en GPU ---")
        print("="*70)
        for N in tamanos_N:
            try:
                t_ms, e, edp, p_avg, gflops, muestras = ejecutar_y_medir(N, 'gpu')
                print(f"GPU | N={N:5d} | T:{t_ms:8.3f}ms | P:{p_avg:7.2f}W | E:{e:8.4f}J | EDP:{edp:10.6f} | GFLOPS:{gflops:7.1f} | Muestras:{muestras}")
                writer.writerow(['gpu', N, t_ms, p_avg, e, edp, gflops, muestras])
                file.flush()
                time.sleep(1)
            except Exception as ex:
                print(f"❌ Error en GPU N={N}: {ex}")
    
    print("\n" + "="*70)
    print(f"✓ Dataset guardado en '{csv_file}'")
    print("="*70)