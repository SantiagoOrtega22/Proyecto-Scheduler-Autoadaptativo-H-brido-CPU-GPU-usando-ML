#!/usr/bin/env python3
"""Module for measuring idle power of the CPU PKG domain.

This module provides tools to read the Intel RAPL PKG energy counter from sysfs
and compute the average power consumed over a baseline idle window.
"""

import os
import time
from typing import Optional


def find_pkg_energy_path() -> Optional[str]:
    """Finds the path to the Intel RAPL PKG energy file in sysfs.

    Checks first for the standard socket 0 PKG domain directory and falls back
    to scanning the powercap directories to locate the CPU Package domain.

    Returns:
        Optional[str]: The absolute path to the energy_uj file if readable,
            otherwise None.
    """
    base_dir = "/sys/class/powercap"
    if not os.path.isdir(base_dir):
        return None

    # intel-rapl:0 is the first CPU socket (PKG domain) in most systems
    common_path = os.path.join(base_dir, "intel-rapl:0", "energy_uj")
    if os.path.isfile(common_path) and os.access(common_path, os.R_OK):
        return common_path

    # Dynamic scanning for a folder with 'package' in its 'name' file
    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("intel-rapl"):
                    continue

                name_path = os.path.join(base_dir, entry.name, "name")
                energy_path = os.path.join(base_dir, entry.name, "energy_uj")
                if (
                    os.path.isfile(name_path)
                    and os.path.isfile(energy_path)
                    and os.access(energy_path, os.R_OK)
                ):
                    with open(name_path, "r") as f:
                        name_val = f.read().strip().lower()
                    if "package" in name_val:
                        return energy_path
    except OSError as e:
        print(f"Error scanning for RAPL PKG domain: {e}")

    return None


def measure_idle_power(duration_sec: float = 2.0) -> float:
    """Measures the average idle power consumption of the CPU PKG domain.

    Reads the starting energy value from RAPL, sleeps for the specified
    duration, reads the final energy value, and calculates the average power
    in Watts.

    Args:
        duration_sec (float): Duration in seconds of the idle measurement
            window. Defaults to 2.0.

    Returns:
        float: Average idle power consumption in Watts (W).

    Raises:
        RuntimeError: If the RAPL path is not found, readable, or if a
            counter rollover occurs during measurement.
    """
    energy_path = find_pkg_energy_path()
    if not energy_path:
        raise RuntimeError(
            "Intel RAPL PKG energy path could not be found or is not readable."
        )

    # Read initial energy value
    try:
        t0 = time.perf_counter()
        with open(energy_path, "r") as f:
            e0 = int(f.read().strip())
    except Exception as e:
        raise RuntimeError(
            f"Failed to read initial energy from {energy_path}: {e}"
        )

    # Let the system remain in idle state during the window
    time.sleep(duration_sec)

    # Read final energy value
    try:
        t1 = time.perf_counter()
        with open(energy_path, "r") as f:
            e1 = int(f.read().strip())
    except Exception as e:
        raise RuntimeError(
            f"Failed to read final energy from {energy_path}: {e}"
        )

    elapsed = t1 - t0
    if elapsed <= 0.0:
        return 0.0

    # energy_uj is in microjoules. Convert to Joules: 1 microjoule = 1e-6 Joules
    energy_j = (e1 - e0) / 1e6

    # Handle counter rollover
    if energy_j < 0:
        raise RuntimeError(
            "RAPL energy counter rollover detected during idle measurement."
        )

    return energy_j / elapsed


if __name__ == "__main__":
    print("Iniciando medición de consumo en reposo (idle)...")
    try:
        power_w = measure_idle_power(duration_sec=3.0)
        print(f"Consumo promedio en reposo (idle) medido: {power_w:.4f} W")
    except Exception as err:
        print(f"Error al medir el consumo en reposo: {err}")
