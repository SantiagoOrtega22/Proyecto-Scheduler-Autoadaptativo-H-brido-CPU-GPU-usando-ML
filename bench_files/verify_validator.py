#!/usr/bin/env python3
"""
verify_validator.py
===================
Generates mock telemetry data to test the MeasurementValidator and prints
the validation report to confirm everything works properly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add root folder to path to import the validator
sys.path.append(str(Path(__file__).parent.parent))
from bench_files.measurement_validator import MeasurementValidator


def generate_mock_csv(output_path: str) -> None:
    """Generates a CSV file containing mock benchmark telemetry data.

    Creates three configurations of 30 iterations each:
    1. Config A: Stable, normal distribution, stable temperature. (Dato Fiable)
    2. Config B: High variance / noisy (CV > 5%). (Requiere Re-ejecución)
    3. Config C: Non-normal distribution with thermal drift. (Usar Mediana / Re-ejecución)
    """
    np.random.seed(12345)
    
    records = []

    # Config A: Stable, normal distribution
    # N=1024, CPU, sgemm, constant temp around 45C
    times_a = np.random.normal(loc=0.012, scale=0.0002, size=30)  # Low std dev -> CV ~ 1.6%
    temps_a = np.random.normal(loc=45.0, scale=0.3, size=30)
    for i, (t, temp) in enumerate(zip(times_a, temps_a)):
        records.append({
            "Device": "cpu",
            "N": 1024,
            "Precision": "S",
            "Iteration": i,
            "Time_sec": t,
            "Avg_Power_W": 45.2,
            "Energy_J": 45.2 * t,
            "EDP": 45.2 * t * t,
            "Temperature_C": temp
        })

    # Config B: High variance / noisy (CV > 5%)
    # N=2048, CPU, sgemm, high noise (e.g. background task ran)
    # Mean = 0.045, standard deviation = 0.004 -> CV ~ 8.8%
    times_b = np.random.normal(loc=0.045, scale=0.004, size=30)
    # Add one massive outlier
    times_b[15] = 0.085
    temps_b = np.random.normal(loc=52.0, scale=0.5, size=30)
    for i, (t, temp) in enumerate(zip(times_b, temps_b)):
        records.append({
            "Device": "cpu",
            "N": 2048,
            "Precision": "S",
            "Iteration": i,
            "Time_sec": t,
            "Avg_Power_W": 65.1,
            "Energy_J": 65.1 * t,
            "EDP": 65.1 * t * t,
            "Temperature_C": temp
        })

    # Config C: Non-normal distribution with thermal drift
    # N=4096, GPU, sgemm
    # Temperature increases steadily from 40C to 75C, causing clocks to throttle and time to increase
    temps_c = np.linspace(40.0, 75.0, 30)
    # Let time be heavily correlated with temperature
    times_c = 0.005 + 0.0002 * temps_c + np.random.normal(loc=0.0, scale=0.0003, size=30)
    for i, (t, temp) in enumerate(zip(times_c, temps_c)):
        records.append({
            "Device": "gpu",
            "N": 4096,
            "Precision": "S",
            "Iteration": i,
            "Time_sec": t,
            "Avg_Power_W": 120.0,
            "Energy_J": 120.0 * t,
            "EDP": 120.0 * t * t,
            "Temperature_C": temp
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Mock telemetry generated at: {output_path}")


def main() -> None:
    csv_file = "bench_files/mock_telemetry.csv"
    generate_mock_csv(csv_file)
    
    print("\n--- RUNNING MEASUREMENT VALIDATOR ON MOCK DATA ---\n")
    validator = MeasurementValidator(
        csv_path=csv_file,
        metric_col="Time_sec",
        temp_col="Temperature_C"
    )
    validator.print_validation_report()


if __name__ == "__main__":
    main()
