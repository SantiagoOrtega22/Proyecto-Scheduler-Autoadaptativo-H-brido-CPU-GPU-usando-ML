import subprocess
import re

def ejecutar_benchmark(dispositivo, N):
    # Elegir el binario correcto
    if dispositivo == "cpu":
        comando = ["./gemm_cpu", str(N)]
    elif dispositivo == "gpu":
        comando = ["./gemm_gpu", str(N)]
    else:
        raise ValueError("Dispositivo no válido")

    # Ejecutar y capturar lo que el programa imprime en consola
    resultado = subprocess.run(comando, capture_output=True, text=True)
    
    # Parsear la salida con regex para extraer tiempo y GFLOPS
    # Formato esperado: "N=5000 | tiempo=0.123 ms | GFLOPS=456.7"
    match = re.search(r'tiempo=([\d.]+)\s*ms.*GFLOPS=([\d.]+)', resultado.stdout)
    
    if not match:
        raise ValueError(f"No se pudo parsear la salida: {resultado.stdout}")
    
    tiempo_ms = float(match.group(1))
    gflops = float(match.group(2))
    
    return tiempo_ms, gflops

# Ejemplo de uso:
dispositivo = "cpu"
tiempo_ms, gflops = ejecutar_benchmark(dispositivo, 2048)
print(f"{dispositivo} tardó {tiempo_ms:.3f} ms con {gflops:.1f} GFLOPS")