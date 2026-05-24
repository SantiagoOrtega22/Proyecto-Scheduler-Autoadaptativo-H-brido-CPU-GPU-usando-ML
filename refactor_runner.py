import re

with open("benchmark_runner.py", "r") as f:
    content = f.read()

# Change default gemm-warmup to 4
content = content.replace('default=3,\n        help="Ejecuciones de warmup previas a GEMM"', 'default=4,\n        help="Ejecuciones de warmup previas a GEMM"')
# Change default fft-warmup to 4
content = content.replace('default=3,\n        help="Iteraciones de warmup FFT"', 'default=4,\n        help="Iteraciones de warmup FFT"')

# --- Refactor run_single_case (GEMM) ---
gemm_old = """        if warmup_runs > 0:
            run_gemm_warmup(cmd, timeout, warmup_runs, matrix_file)

        power_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        monitor_thread = None
        start_wall = time.perf_counter()

        if device == "gpu":
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            monitor_thread = threading.Thread(
                target=monitor_power_gpu,
                args=(handle, stop_event, power_queue),
                daemon=True,
            )
        else:
            rapl = find_rapl_energy_path()
            if rapl:
                monitor_thread = threading.Thread(
                    target=monitor_power_cpu,
                    args=(rapl, stop_event, power_queue),
                    daemon=True,
                )
            else:
                warn_rapl_missing_once()

        if monitor_thread is not None:
            monitor_thread.start()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        finally:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join()

        end_wall = time.perf_counter()
        samples = power_queue.get() if not power_queue.empty() else []

        if proc.returncode != 0:
            raise RuntimeError(
                "Fallo en binario para "
                f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\\n"
                f"STDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
            )

        match = TIME_PATTERN.search(proc.stdout)
        if not match:
            raise RuntimeError(
                "No se pudo parsear Time_sec de la salida del binario.\\n"
                f"Salida:\\n{proc.stdout}"
            )

        time_sec = float(match.group(1))"""

gemm_new = """        # 1. Warmups
        if warmup_runs > 0:
            run_gemm_warmup(cmd, timeout, warmup_runs, matrix_file)

        # 2. Metric Isolation Execution (Solo Tiempo)
        proc_iso = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc_iso.returncode != 0:
            raise RuntimeError(
                "Fallo en binario para "
                f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\\n"
                f"STDOUT:\\n{proc_iso.stdout}\\nSTDERR:\\n{proc_iso.stderr}"
            )

        match = TIME_PATTERN.search(proc_iso.stdout)
        if not match:
            raise RuntimeError(
                "No se pudo parsear Time_sec de la salida del binario en aislamiento.\\n"
                f"Salida:\\n{proc_iso.stdout}"
            )

        time_sec = float(match.group(1))

        # 3. Power Monitoring Execution (Segunda ejecucion identica con monitor activo)
        power_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        monitor_thread = None
        start_wall = time.perf_counter()

        if device == "gpu":
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            monitor_thread = threading.Thread(
                target=monitor_power_gpu,
                args=(handle, stop_event, power_queue),
                daemon=True,
            )
        else:
            rapl = find_rapl_energy_path()
            if rapl:
                monitor_thread = threading.Thread(
                    target=monitor_power_cpu,
                    args=(rapl, stop_event, power_queue),
                    daemon=True,
                )
            else:
                warn_rapl_missing_once()

        if monitor_thread is not None:
            monitor_thread.start()

        try:
            proc_pwr = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        finally:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join()

        end_wall = time.perf_counter()
        samples = power_queue.get() if not power_queue.empty() else []

        if proc_pwr.returncode != 0:
            raise RuntimeError(
                "Fallo en binario para ejecucion de monitoreo "
                f"M={m}, N={n}, K={k}, P={precision}, OpA={op_a}, OpB={op_b}.\\n"
                f"STDOUT:\\n{proc_pwr.stdout}\\nSTDERR:\\n{proc_pwr.stderr}"
            )"""

content = content.replace(gemm_old, gemm_new)

# --- Refactor run_single_case_fft ---
fft_old = """    if matrix_file:
        cmd.append(matrix_file)

    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    monitor_thread = None
    start_wall = time.perf_counter()

    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu,
            args=(handle, stop_event, power_queue),
            daemon=True,
        )
    else:
        rapl = find_rapl_energy_path()
        if rapl:
            monitor_thread = threading.Thread(
                target=monitor_power_cpu,
                args=(rapl, stop_event, power_queue),
                daemon=True,
            )
        else:
            warn_rapl_missing_once()

    if monitor_thread is not None:
        monitor_thread.start()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join()

    end_wall = time.perf_counter()
    samples = power_queue.get() if not power_queue.empty() else []

    if proc.returncode != 0:
        raise RuntimeError(
            "Fallo en binario FFT para "
            f"Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, Dir={direction}, L={layout}.\\n"
            f"STDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
        )

    match = FFT_TIME_PATTERN.search(proc.stdout)
    if not match:
        raise RuntimeError(
            "No se pudo parsear tiempo de la salida FFT.\\n"
            f"Salida:\\n{proc.stdout}"
        )

    if match.group(1) is not None:
        time_sec = float(match.group(1))
    else:
        time_sec = float(match.group(2)) / 1e3"""

fft_new = """    if matrix_file:
        cmd.append(matrix_file)

    # 1. Metric Isolation Execution (Solo Tiempo)
    proc_iso = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if proc_iso.returncode != 0:
        raise RuntimeError(
            "Fallo en binario FFT para "
            f"Nx={nx}, Ny={ny}, Nz={nz}, Batch={batch}, P={precision}, D={domain}, Dir={direction}, L={layout}.\\n"
            f"STDOUT:\\n{proc_iso.stdout}\\nSTDERR:\\n{proc_iso.stderr}"
        )

    match = FFT_TIME_PATTERN.search(proc_iso.stdout)
    if not match:
        raise RuntimeError(
            "No se pudo parsear tiempo de la salida FFT.\\n"
            f"Salida:\\n{proc_iso.stdout}"
        )

    if match.group(1) is not None:
        time_sec = float(match.group(1))
    else:
        time_sec = float(match.group(2)) / 1e3

    # 2. Power Monitoring Execution (Segunda ejecucion con monitor activo)
    power_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    monitor_thread = None
    start_wall = time.perf_counter()

    if device == "gpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        monitor_thread = threading.Thread(
            target=monitor_power_gpu,
            args=(handle, stop_event, power_queue),
            daemon=True,
        )
    else:
        rapl = find_rapl_energy_path()
        if rapl:
            monitor_thread = threading.Thread(
                target=monitor_power_cpu,
                args=(rapl, stop_event, power_queue),
                daemon=True,
            )
        else:
            warn_rapl_missing_once()

    if monitor_thread is not None:
        monitor_thread.start()

    try:
        proc_pwr = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join()

    end_wall = time.perf_counter()
    samples = power_queue.get() if not power_queue.empty() else []
    
    if proc_pwr.returncode != 0:
        raise RuntimeError("Fallo en ejecucion de monitoreo de FFT.")"""

content = content.replace(fft_old, fft_new)

with open("benchmark_runner.py", "w") as f:
    f.write(content)

print("Done runner")
