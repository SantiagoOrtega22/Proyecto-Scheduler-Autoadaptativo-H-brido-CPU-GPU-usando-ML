#!/usr/bin/env python3
"""Module for measuring idle power of the CPU PKG domain.

This module provides tools to read the Intel RAPL PKG energy counter from sysfs
and compute the average power consumed over a baseline idle window.
"""

import os
import time
from typing import Optional


def find_pkg_energy_paths() -> list[str]:
    """Finds all paths to the Intel RAPL PKG energy files in sysfs.

    Returns:
        list[str]: Absolute paths to all Package energy_uj files found and readable.
    """
    base_dir = "/sys/class/powercap"
    paths = []
    if not os.path.isdir(base_dir):
        return paths

    def is_readable(p):
        return os.path.isfile(p) and os.access(p, os.R_OK)

    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("intel-rapl"):
                    continue

                name_path = os.path.join(base_dir, entry.name, "name")
                energy_path = os.path.join(base_dir, entry.name, "energy_uj")
                if is_readable(name_path) and is_readable(energy_path):
                    try:
                        with open(name_path, "r") as f:
                            name_val = f.read().strip().lower()
                        if "package" in name_val:
                            paths.append(energy_path)
                    except Exception:
                        continue
    except OSError as e:
        print(f"Error scanning for RAPL PKG domain: {e}")

    if not paths:
        fallback = os.path.join(base_dir, "intel-rapl:0", "energy_uj")
        if is_readable(fallback):
            paths.append(fallback)

    return sorted(paths)


def measure_idle_power(duration_sec: float = 2.0) -> float:
    """Measures the average idle power consumption of all detected CPU PKG domains.

    Reads starting energy values from all sockets, sleeps for the specified
    duration, reads final energy values, and calculates the total average power
    in Watts.

    Args:
        duration_sec (float): Duration in seconds of the idle measurement
            window. Defaults to 2.0.

    Returns:
        float: Total average idle power consumption in Watts (W) across all sockets.

    Raises:
        RuntimeError: If no RAPL paths are found, or if a counter rollover occurs.
    """
    energy_paths = find_pkg_energy_paths()
    if not energy_paths:
        raise RuntimeError(
            "Intel RAPL PKG energy paths could not be found or are not readable."
        )

    e0_list = []
    # Read initial energy values from all sockets
    try:
        t0 = time.perf_counter()
        for path in energy_paths:
            with open(path, "r") as f:
                e0_list.append(int(f.read().strip()))
    except Exception as e:
        raise RuntimeError(
            f"Failed to read initial energy: {e}"
        )

    # Let the system remain in idle state during the window
    time.sleep(duration_sec)

    e1_list = []
    # Read final energy values from all sockets
    try:
        t1 = time.perf_counter()
        for path in energy_paths:
            with open(path, "r") as f:
                e1_list.append(int(f.read().strip()))
    except Exception as e:
        raise RuntimeError(
            f"Failed to read final energy: {e}"
        )

    elapsed = t1 - t0
    if elapsed <= 0.0:
        return 0.0

    energy_total_j = 0.0
    for i in range(len(energy_paths)):
        diff = (e1_list[i] - e0_list[i]) / 1e6
        if diff < 0:
            raise RuntimeError(
                "RAPL energy counter rollover detected during idle measurement."
            )
        energy_total_j += diff

    return energy_total_j / elapsed


if __name__ == "__main__":
    print("Iniciando medición de consumo en reposo (idle)...")
    try:
        power_w = measure_idle_power(duration_sec=3.0)
        print(f"Consumo promedio en reposo (idle) medido: {power_w:.4f} W")
    except Exception as err:
        print(f"Error al medir el consumo en reposo: {err}")
