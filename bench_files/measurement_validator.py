#!/usr/bin/env python3
"""
measurement_validator.py
========================
Statistical validator for CPU/GPU benchmarks in heterogenous environments.
Evaluates metrics reliability, thermal stability, environment setup, and normality
in collected data to ensure suitability for training Reinforcement Learning agents.

Design standards:
- SRP: Exclusively handles metrics validation, statistical checks, and telemetry sanity.
- Rigor: Validates warm-up effects, thermal delta limits, and frequency scaling modes.
- OOP: Uses clear class hierarchy, Google-style docstrings, and strict type hinting.
"""

import os
import re
import sys
import glob
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import matplotlib
    # Use Agg backend if running in a headless environment (e.g. cluster node)
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class MeasurementValidator:
    """Validator class for benchmark measurement reliability.

    This class parses benchmark CSV output, groups iterations into bursts,
    and applies rigorous statistical tests (CV, Shapiro-Wilk, thermal delta checks,
    and IQR outlier removal) to verify if data is clean enough for RL training.
    """

    def __init__(
        self,
        csv_path: str,
        metric_col: str = "Time_sec",
        temp_col: Optional[str] = None,
        power_col: str = "Avg_Power_W",
        edp_col: str = "EDP",
        energy_col: str = "Energy_J",
    ) -> None:
        """Initializes the MeasurementValidator.

        Args:
            csv_path (str): Path to the CSV file containing measurement results.
            metric_col (str): Target performance metric column name (default: "Time_sec").
            temp_col (Optional[str]): Temperature metric column name. If None, auto-detected.
            power_col (str): Power consumption column name (default: "Avg_Power_W").
            edp_col (str): Energy-Delay Product column name (default: "EDP").

        Raises:
            FileNotFoundError: If the CSV path does not exist.
            ValueError: If the CSV is empty or invalid.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.metric_col = metric_col
        self.power_col = power_col
        self.edp_col = edp_col
        self.energy_col = energy_col
        self.temp_col = temp_col
        self.df: pd.DataFrame = pd.DataFrame()
        self._load_data()

    def _load_data(self) -> None:
        """Loads data from the CSV file and performs basic integrity checks.

        Raises:
            ValueError: If the CSV has no rows or is missing necessary columns.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {self.csv_path}: {e}")

        if self.df.empty:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        # Standardize column casing to match expected names
        col_mapping = {col.lower(): col for col in self.df.columns}
        
        # Verify performance metric column exists
        if self.metric_col not in self.df.columns:
            lower_metric = self.metric_col.lower()
            if lower_metric in col_mapping:
                self.metric_col = col_mapping[lower_metric]
            else:
                raise ValueError(
                    f"Performance metric column '{self.metric_col}' not found in CSV. "
                    f"Available columns: {list(self.df.columns)}"
                )

        # Handle power and edp columns
        if self.power_col not in self.df.columns and self.power_col.lower() in col_mapping:
            self.power_col = col_mapping[self.power_col.lower()]
        if self.edp_col not in self.df.columns and self.edp_col.lower() in col_mapping:
            self.edp_col = col_mapping[self.edp_col.lower()]
        if self.energy_col not in self.df.columns:
            if self.energy_col.lower() in col_mapping:
                self.energy_col = col_mapping[self.energy_col.lower()]
            elif "energy_j" in col_mapping:
                self.energy_col = col_mapping["energy_j"]
            elif "energy" in col_mapping:
                self.energy_col = col_mapping["energy"]

        # Try auto-detecting temperature column if not provided
        if self.temp_col is None:
            temp_candidates = ["temperature_c", "temp_c", "start_temp_c", "temp", "temperature"]
            for cand in temp_candidates:
                if cand in col_mapping:
                    self.temp_col = col_mapping[cand]
                    break
                # Case-insensitive substring match
                for col in self.df.columns:
                    if cand in col.lower():
                        self.temp_col = col
                        break
                if self.temp_col:
                    break
        elif self.temp_col not in self.df.columns:
            # Fallback if specific temp col requested is missing
            self.temp_col = None

    def _detect_config_cols(self) -> List[str]:
        """Detects columns that define the experimental configuration.

        Exclude performance and energy metrics so grouping isolates config bursts.

        Returns:
            List[str]: Column names that identify the hardware/algorithm parameters.
        """
        excluded_patterns = [
            r"time", r"sec", r"ms", r"gflops", r"power", r"watt",
            r"energy", r"joule", r"edp", r"temp", r"sample", r"elapsed",
            r"iter", r"run", r"index"
        ]
        config_cols = []
        for col in self.df.columns:
            is_excluded = False
            for pat in excluded_patterns:
                if re.search(pat, col.lower()):
                    is_excluded = True
                    break
            if not is_excluded:
                config_cols.append(col)
        
        # If no config columns found, default to standard ones if present
        if not config_cols:
            for fallback in ["Device", "M", "N", "K", "Precision", "OpA", "OpB", "Nx", "Ny", "Nz", "Batch", "Domain"]:
                if fallback in self.df.columns:
                    config_cols.append(fallback)

        return config_cols

    def check_thermal_stability(self, temp_series: pd.Series) -> Tuple[bool, float, List[float]]:
        """Checks if initial temperatures follow the stationary thermal state protocol.

        Protocol requires that temperatures within a burst remain stable:
        1. All successive differences between adjacent iterations remain within +-2 C.
        2. The total span (max - min) within the burst remains within +-2 C.

        Args:
            temp_series (pd.Series): Sequence of temperature readings.

        Returns:
            Tuple[bool, float, List[float]]:
                - bool: True if all stability constraints are satisfied.
                - float: Maximum absolute temperature delta (from consecutive steps or overall span).
                - List[float]: List of consecutive temperature deltas.
        """
        if temp_series.empty:
            return True, 0.0, []

        temp_vals = temp_series.to_numpy()
        if len(temp_vals) < 2:
            return True, 0.0, []

        deltas = np.abs(np.diff(temp_vals))
        max_consecutive_delta = float(np.max(deltas))
        total_span = float(np.max(temp_vals) - np.min(temp_vals))
        
        # Thermal stability is violated if consecutive difference > 2C or overall span > 2C
        stable = (max_consecutive_delta <= 2.0) and (total_span <= 2.0)
        max_delta = max(max_consecutive_delta, total_span)
        
        return stable, max_delta, list(deltas)

    def check_normality(self, data: pd.Series) -> Tuple[float, float, bool]:
        """Performs the Shapiro-Wilk test to evaluate the normality of the samples.

        Args:
            data (pd.Series): Measurements dataset.

        Returns:
            Tuple[float, float, bool]:
                - float: W-statistic.
                - float: p-value.
                - bool: True if normal (p-value >= 0.05), False otherwise.
        """
        if len(data) < 3:
            # Shapiro-Wilk requires at least 3 samples
            return 1.0, 1.0, True

        w_stat, p_val = stats.shapiro(data)
        is_normal = p_val >= 0.05
        return float(w_stat), float(p_val), is_normal

    def check_outliers(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, float, float]:
        """Filters outliers using the Interquartile Range (IQR) method.

        Args:
            data (pd.Series): Metric values series.

        Returns:
            Tuple[pd.Series, pd.Series, float, float]:
                - pd.Series: Cleaned data containing no outliers.
                - pd.Series: Values identified as outliers.
                - float: Lower limit boundary.
                - float: Upper limit boundary.
        """
        if len(data) < 4:
            # Insufficient data to calculate meaningful quartiles
            return data, pd.Series(dtype=data.dtype), -float("inf"), float("inf")

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        cleaned = data[(data >= lower_bound) & (data <= upper_bound)]
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return cleaned, outliers, float(lower_bound), float(upper_bound)

    def diagnose_environment(self) -> Dict[str, Any]:
        """Diagnoses CPU scaling governors and nodes isolation status on the system.

        Queries Linux sysfs interface safely.

        Returns:
            Dict[str, Any]: Environmental metrics including:
                - governors: List of active CPU governors.
                - performance_mode: True if all active cores are set to 'performance'.
                - cores_count: Number of CPUs detected.
                - isolated_cores: Info about isolated CPUs (isolcpus).
                - error_log: List of execution logs or errors.
        """
        diagnosis = {
            "governors": [],
            "performance_mode": False,
            "cores_count": 0,
            "isolated_cores": "None",
            "warnings": []
        }

        # Query CPU Governors
        gov_path = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        try:
            gov_files = glob.glob(gov_path)
            governors = []
            for path in gov_files:
                with open(path, "r") as f:
                    governors.append(f.read().strip())
            
            if governors:
                diagnosis["governors"] = list(set(governors))
                diagnosis["cores_count"] = len(governors)
                diagnosis["performance_mode"] = all(gov == "performance" for gov in governors)
                if not diagnosis["performance_mode"]:
                    diagnosis["warnings"].append(
                        f"Active CPU governors found: {diagnosis['governors']}. "
                        "HPC protocol requires all cores in 'performance' mode."
                    )
            else:
                diagnosis["warnings"].append("No CPU cpufreq scaling governors detected in sysfs.")
        except Exception as e:
            diagnosis["warnings"].append(f"Could not read CPU scaling governors: {e}")

        # Check Isolated Cores (isolcpus) in boot command line
        cmdline_path = "/proc/cmdline"
        try:
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "r") as f:
                    cmdline = f.read()
                    match = re.search(r"isolcpus=(\S+)", cmdline)
                    if match:
                        diagnosis["isolated_cores"] = match.group(1)
                    else:
                        diagnosis["warnings"].append(
                            "No CPU core isolation (isolcpus) detected in kernel command line. "
                            "Benchmarking on shared nodes might suffer from scheduler context switches."
                        )
        except Exception as e:
            diagnosis["warnings"].append(f"Could not read kernel cmdline: {e}")

        return diagnosis

    def analyze_bursts(self) -> List[Dict[str, Any]]:
        """Processes the CSV data and evaluates each constant configuration burst.

        Returns:
            List[Dict[str, Any]]: Analysis results containing statistical checks
                for each detected burst.
        """
        config_cols = self._detect_config_cols()
        if not config_cols:
            # Fallback: Treat entire dataset as a single group if no configs are isolated
            groups = [((), self.df)]
        else:
            groups = self.df.groupby(config_cols)

        results = []
        for gp_val, gp_df in groups:
            # Reconstruct config dict
            config = {}
            if len(config_cols) == 1:
                config[config_cols[0]] = gp_val
            else:
                for i, col in enumerate(config_cols):
                    config[col] = gp_val[i]

            m_size = len(gp_df)
            metric_data = gp_df[self.metric_col]

            # 1. Calculate base stats
            mean_raw = float(metric_data.mean())
            std_raw = float(metric_data.std()) if m_size > 1 else 0.0
            cv_raw = (std_raw / mean_raw * 100.0) if mean_raw > 0 else 0.0

            # 2. Outlier filtering
            cleaned_metric, outliers, l_lim, u_lim = self.check_outliers(metric_data)
            mean_clean = float(cleaned_metric.mean())
            median_clean = float(cleaned_metric.median())
            std_clean = float(cleaned_metric.std()) if len(cleaned_metric) > 1 else 0.0
            cv_clean = (std_clean / mean_clean * 100.0) if mean_clean > 0 else 0.0

            # 3. Shapiro-Wilk Test for Normality
            w_stat, p_val, is_normal = self.check_normality(metric_data)

            # 4. Thermal Stability delta check
            thermal_stable = True
            max_t_delta = 0.0
            temp_list = []
            thermal_drift_corr = 0.0
            
            if self.temp_col and self.temp_col in gp_df.columns:
                temp_series = gp_df[self.temp_col]
                temp_list = temp_series.tolist()
                thermal_stable, max_t_delta, _ = self.check_thermal_stability(temp_series)

                # Compute Pearson correlation between Metric and Temperature if normal test fails
                if len(metric_data) > 2 and temp_series.std() > 0 and metric_data.std() > 0:
                    corr, _ = stats.pearsonr(temp_series, metric_data)
                    thermal_drift_corr = float(corr)

            # 5. Determine Veredicto de Confianza
            if cv_clean > 5.0:
                verdict = "Requiere Re-ejecución"
                justification = f"El Coeficiente de Variación depurado ({cv_clean:.2f}%) supera el límite del 5%, indicando alto ruido de fondo."
            elif not is_normal:
                verdict = "Usar Mediana por No-Normalidad"
                justification = f"Falla prueba de Shapiro-Wilk (p={p_val:.4f}). Los residuos no son normales (ruido sistemático). Se aconseja usar la mediana en lugar de la media para mitigar sesgos."
            else:
                verdict = "Dato Fiable"
                justification = f"Mediciones estables (CV={cv_clean:.2f}%) y distribución normal (Shapiro W={w_stat:.4f}, p={p_val:.4f})."

            if not thermal_stable:
                verdict = "Requiere Re-ejecución"
                justification = f"Violación del Protocolo de Estado Estacionario Térmico: Delta de temperatura máximo de {max_t_delta:.1f}°C supera el límite de ±2°C."

            results.append({
                "config": config,
                "burst_size": m_size,
                "metric_col": self.metric_col,
                "mean_raw": mean_raw,
                "cv_raw": cv_raw,
                "mean_clean": mean_clean,
                "median_clean": median_clean,
                "cv_clean": cv_clean,
                "outliers_count": len(outliers),
                "outliers_values": outliers.tolist(),
                "shapiro_w": w_stat,
                "shapiro_p": p_val,
                "is_normal": is_normal,
                "thermal_stable": thermal_stable,
                "max_temp_delta": max_t_delta,
                "thermal_correlation": thermal_drift_corr,
                "verdict": verdict,
                "justification": justification,
                "temperatures": temp_list
            })

        return results

    def print_validation_report(self) -> None:
        """Prints a highly detailed validation report formatted as requested."""
        env = self.diagnose_environment()
        bursts = self.analyze_bursts()

        print("=" * 80)
        print("                 REPORTE DE VALIDACIÓN DE TELEMETRÍA (HPC)")
        print("=" * 80)
        
        # 1. Diagnóstico de Entorno
        print("\n=== Diagnóstico de Entorno ===")
        print(f"Número de Cores CPU Detectados: {env['cores_count']}")
        print(f"Gobernadores de Frecuencia CPU Activos: {env['governors']}")
        status_gov = "Frecuencia Fija (PERFORMANCE)" if env["performance_mode"] else "Modo de ahorro de energía o Variable (¡ALERTA!)"
        print(f"Estado de Frecuencias: {status_gov}")
        print(f"Aislamiento de Cores (isolcpus): {env['isolated_cores']}")
        if env["warnings"]:
            print("Alertas del Entorno:")
            for w in env["warnings"]:
                print(f"  [!] {w}")

        # 2. Burst analysis
        print("\n=== Análisis Estadístico por Configuración ===")
        for i, b in enumerate(bursts):
            cfg_desc = ", ".join(f"{k}={v}" for k, v in b["config"].items())
            print("-" * 60)
            print(f"Configuración [{i+1}]: {cfg_desc}")
            print(f"  Tamaño de Ráfaga (M): {b['burst_size']}")
            print(f"  Métrica Analizada: {b['metric_col']}")
            print(f"  CV Raw / Clean: {b['cv_raw']:.2f}% / {b['cv_clean']:.2f}%")
            print(f"  Shapiro-Wilk W / p-val: {b['shapiro_w']:.4f} / {b['shapiro_p']:.4f} (Normal: {b['is_normal']})")
            if not b["is_normal"] and self.temp_col:
                print(f"  Correlación Temperatura-Métrica: {b['thermal_correlation']:.4f}")
            print(f"  Anomalías Descartadas (IQR): {b['outliers_count']}")
            if b["temperatures"]:
                print(f"  Estabilidad Térmica (Delta Max): {b['max_temp_delta']:.1f}°C (Estable: {b['thermal_stable']})")
            
            print(f"\n  [VERDICTO DE CONFIANZA]: {b['verdict']}")
            print(f"  [Justificación Técnica]: {b['justification']}")
        print("=" * 80)

    def plot_power_vs_n(self, output_path: str = "power_vs_n.png", x_scale: str = "linear") -> None:
        """Generates a plot of power consumption vs problem size N.

        Groups the data by Device (CPU/GPU) and Precision (S, D, C, Z) if available,
        plotting individual runs as scatter points and trend lines of the average power.

        Args:
            output_path (str): File path to save the generated plot.
            x_scale (str): Scale/layout style of the X-axis. Choices:
                - "linear": Continuous linear scaling.
                - "log": Continuous logarithmic scaling.
                - "categorical": Spreads unique sizes evenly as discrete categories.
        """
        if plt is None:
            print("[!] Matplotlib no está instalado. No se puede generar la gráfica de potencia vs N.", file=sys.stderr)
            return

        # Auto-detect size column
        size_col = None
        for col in ["N", "Nx"]:
            if col in self.df.columns:
                size_col = col
                break

        if not size_col:
            print("[!] No se encontró columna de tamaño ('N' o 'Nx') en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        if self.power_col not in self.df.columns:
            print(f"[!] Columna de potencia '{self.power_col}' no encontrada en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        # Ensure power and size columns are numeric, and clean the data
        temp_df = self.df.copy()
        temp_df[self.power_col] = pd.to_numeric(temp_df[self.power_col], errors='coerce')
        temp_df[size_col] = pd.to_numeric(temp_df[size_col], errors='coerce')

        # Drop rows with NaN values in size or power columns
        valid_df = temp_df[temp_df[self.power_col].notna() & temp_df[size_col].notna()]
        
        if valid_df.empty:
            print("[!] No hay mediciones válidas para graficar.", file=sys.stderr)
            return

        # Use slightly wider figure for categorical scale to display tick labels clearly
        fig_width = 12 if x_scale == "categorical" else 10
        plt.figure(figsize=(fig_width, 6))

        # Check grouping columns
        has_device = "Device" in valid_df.columns
        has_precision = "Precision" in valid_df.columns

        devices = valid_df["Device"].unique() if has_device else ["default"]
        
        # Consistent color palette for devices (CPU = Red/Pinkish, GPU = Blue/Cyan)
        colors = {
            "gpu": "#0984e3",  # Blue
            "cpu": "#d63031",  # Red
            "default": "#2d3436"  # Dark gray
        }

        # Style options for precisions to make the plot clean and readable
        linestyles = {
            "S": "-",
            "D": "--",
            "C": "-.",
            "Z": ":"
        }
        markers = {
            "S": "o",  # Circle
            "D": "s",  # Square
            "C": "^",  # Triangle up
            "Z": "D"   # Diamond
        }

        # Compute mapping for categorical scale
        unique_sizes = sorted(valid_df[size_col].unique())
        size_to_idx = {size: idx for idx, size in enumerate(unique_sizes)}

        for dev in sorted(devices):
            dev_df = valid_df[valid_df["Device"] == dev] if has_device else valid_df
            if dev_df.empty:
                continue

            color = colors.get(str(dev).lower(), "#6c5ce7")
            dev_label = str(dev).upper() if has_device else "Total"

            # Check if there are different precisions inside this device group
            precisions = dev_df["Precision"].unique() if has_precision else ["default"]

            for prec in sorted(precisions):
                prec_df = dev_df[dev_df["Precision"] == prec] if has_precision else dev_df
                if prec_df.empty:
                    continue

                prec_label = f" (Precisión {prec})" if has_precision else ""
                linestyle = linestyles.get(str(prec).upper(), "-")
                marker = markers.get(str(prec).upper(), "o")

                if x_scale == "categorical":
                    x = prec_df[size_col].map(size_to_idx)
                else:
                    x = prec_df[size_col]
                y = prec_df[self.power_col]

                # 1. Scatter plot for individual iterations (capturing noise / variance)
                plt.scatter(
                    x, y,
                    color=color,
                    alpha=0.35,
                    edgecolors="none",
                    s=40,
                    marker=marker,
                    label=f"{dev_label}{prec_label} - Mediciones"
                )

                # 2. Line plot for trend (mean of power per size)
                grouped = prec_df.groupby(size_col)[self.power_col].mean().reset_index()
                grouped = grouped.sort_values(by=size_col)

                if x_scale == "categorical":
                    grouped_x = grouped[size_col].map(size_to_idx)
                else:
                    grouped_x = grouped[size_col]

                plt.plot(
                    grouped_x, grouped[self.power_col],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    marker=marker,
                    markersize=6,
                    label=f"{dev_label}{prec_label} - Promedio"
                )

        title_suffix = ""
        if x_scale == "categorical":
            title_suffix = " (Escala Categórica)"
        elif x_scale == "log":
            title_suffix = " (Escala Logarítmica)"

        plt.title(f"Consumo de Potencia Promedio vs Tamaño del Problema (N){title_suffix}", fontsize=14, fontweight="bold", pad=15)
        
        if x_scale == "categorical":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Índice Categórico]", fontsize=12)
            plt.xticks(
                ticks=range(len(unique_sizes)),
                labels=[str(int(s)) for s in unique_sizes],
                rotation=45,
                ha='right'
            )
        elif x_scale == "log":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Logarítmica]", fontsize=12)
            # Use base 2 logarithmic scale if sizes are powers of 2
            is_pow2 = all((int(s) & (int(s) - 1) == 0) and s > 0 for s in unique_sizes if float(s).is_integer())
            if is_pow2:
                plt.xscale("log", base=2)
            else:
                plt.xscale("log")
        else:
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Lineal]", fontsize=12)

        plt.ylabel("Potencia (W)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Deduplicate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, facecolor="white", edgecolor="#dfe6e9")

        # Save to file
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[*] Gráfico de potencia vs N guardado como '{output_path}'")
        except Exception as e:
            print(f"[!] No se pudo guardar la imagen del gráfico de potencia: {e}", file=sys.stderr)

        # Interactive display
        try:
            if matplotlib.get_backend() != "Agg":
                plt.show()
        except Exception as e:
            pass

    def plot_energy_vs_n(self, output_path: str = "energy_vs_n.png", x_scale: str = "linear") -> None:
        """Generates a plot of energy consumption vs problem size N.

        Groups the data by Device (CPU/GPU) and Precision (S, D, C, Z) if available,
        plotting individual runs as scatter points and trend lines of the average energy.

        Args:
            output_path (str): File path to save the generated plot.
            x_scale (str): Scale/layout style of the X-axis. Choices:
                - "linear": Continuous linear scaling.
                - "log": Continuous logarithmic scaling.
                - "categorical": Spreads unique sizes evenly as discrete categories.
        """
        if plt is None:
            print("[!] Matplotlib no está instalado. No se puede generar la gráfica de energía vs N.", file=sys.stderr)
            return

        # Auto-detect size column
        size_col = None
        for col in ["N", "Nx"]:
            if col in self.df.columns:
                size_col = col
                break

        if not size_col:
            print("[!] No se encontró columna de tamaño ('N' o 'Nx') en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        if self.energy_col not in self.df.columns:
            print(f"[!] Columna de energía '{self.energy_col}' no encontrada en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        # Ensure energy and size columns are numeric, and clean the data
        temp_df = self.df.copy()
        temp_df[self.energy_col] = pd.to_numeric(temp_df[self.energy_col], errors='coerce')
        temp_df[size_col] = pd.to_numeric(temp_df[size_col], errors='coerce')

        # Drop rows with NaN values in size or energy columns
        valid_df = temp_df[temp_df[self.energy_col].notna() & temp_df[size_col].notna()]
        
        if valid_df.empty:
            print("[!] No hay mediciones válidas para graficar.", file=sys.stderr)
            return

        # Use slightly wider figure for categorical scale to display tick labels clearly
        fig_width = 12 if x_scale == "categorical" else 10
        plt.figure(figsize=(fig_width, 6))

        # Check grouping columns
        has_device = "Device" in valid_df.columns
        has_precision = "Precision" in valid_df.columns

        devices = valid_df["Device"].unique() if has_device else ["default"]
        
        # Consistent color palette for devices (CPU = Red/Pinkish, GPU = Blue/Cyan)
        colors = {
            "gpu": "#0984e3",  # Blue
            "cpu": "#d63031",  # Red
            "default": "#2d3436"  # Dark gray
        }

        # Style options for precisions to make the plot clean and readable
        linestyles = {
            "S": "-",
            "D": "--",
            "C": "-.",
            "Z": ":"
        }
        markers = {
            "S": "o",  # Circle
            "D": "s",  # Square
            "C": "^",  # Triangle up
            "Z": "D"   # Diamond
        }

        # Compute mapping for categorical scale
        unique_sizes = sorted(valid_df[size_col].unique())
        size_to_idx = {size: idx for idx, size in enumerate(unique_sizes)}

        for dev in sorted(devices):
            dev_df = valid_df[valid_df["Device"] == dev] if has_device else valid_df
            if dev_df.empty:
                continue

            color = colors.get(str(dev).lower(), "#6c5ce7")
            dev_label = str(dev).upper() if has_device else "Total"

            # Check if there are different precisions inside this device group
            precisions = dev_df["Precision"].unique() if has_precision else ["default"]

            for prec in sorted(precisions):
                prec_df = dev_df[dev_df["Precision"] == prec] if has_precision else dev_df
                if prec_df.empty:
                    continue

                prec_label = f" (Precisión {prec})" if has_precision else ""
                linestyle = linestyles.get(str(prec).upper(), "-")
                marker = markers.get(str(prec).upper(), "o")

                if x_scale == "categorical":
                    x = prec_df[size_col].map(size_to_idx)
                else:
                    x = prec_df[size_col]
                y = prec_df[self.energy_col]

                # 1. Scatter plot for individual iterations (capturing noise / variance)
                plt.scatter(
                    x, y,
                    color=color,
                    alpha=0.35,
                    edgecolors="none",
                    s=40,
                    marker=marker,
                    label=f"{dev_label}{prec_label} - Mediciones"
                )

                # 2. Line plot for trend (mean of energy per size)
                grouped = prec_df.groupby(size_col)[self.energy_col].mean().reset_index()
                grouped = grouped.sort_values(by=size_col)

                if x_scale == "categorical":
                    grouped_x = grouped[size_col].map(size_to_idx)
                else:
                    grouped_x = grouped[size_col]

                plt.plot(
                    grouped_x, grouped[self.energy_col],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    marker=marker,
                    markersize=6,
                    label=f"{dev_label}{prec_label} - Promedio"
                )

        title_suffix = ""
        if x_scale == "categorical":
            title_suffix = " (Escala Categórica)"
        elif x_scale == "log":
            title_suffix = " (Escala Logarítmica)"

        plt.title(f"Consumo de Energía Promedio vs Tamaño del Problema (N){title_suffix}", fontsize=14, fontweight="bold", pad=15)
        
        if x_scale == "categorical":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Índice Categórico]", fontsize=12)
            plt.xticks(
                ticks=range(len(unique_sizes)),
                labels=[str(int(s)) for s in unique_sizes],
                rotation=45,
                ha='right'
            )
        elif x_scale == "log":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Logarítmica]", fontsize=12)
            is_pow2 = all((int(s) & (int(s) - 1) == 0) and s > 0 for s in unique_sizes if float(s).is_integer())
            if is_pow2:
                plt.xscale("log", base=2)
            else:
                plt.xscale("log")
        else:
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Lineal]", fontsize=12)

        plt.ylabel("Energía (J)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Deduplicate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, facecolor="white", edgecolor="#dfe6e9")

        # Save to file
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[*] Gráfico de energía vs N guardado como '{output_path}'")
        except Exception as e:
            print(f"[!] No se pudo guardar la imagen del gráfico de energía: {e}", file=sys.stderr)

        # Interactive display
        try:
            if matplotlib.get_backend() != "Agg":
                plt.show()
        except Exception as e:
            pass

    def plot_time_vs_n(self, output_path: str = "time_vs_n.png", x_scale: str = "linear") -> None:
        """Generates a plot of execution time vs problem size N.

        Groups the data by Device (CPU/GPU) and Precision (S, D, C, Z) if available,
        plotting individual runs as scatter points and trend lines of the average execution time.

        Args:
            output_path (str): File path to save the generated plot.
            x_scale (str): Scale/layout style of the X-axis. Choices:
                - "linear": Continuous linear scaling.
                - "log": Continuous logarithmic scaling.
                - "categorical": Spreads unique sizes evenly as discrete categories.
        """
        if plt is None:
            print(f"[!] Matplotlib no está instalado. No se puede generar la gráfica de {self.metric_col} vs N.", file=sys.stderr)
            return

        # Auto-detect size column
        size_col = None
        for col in ["N", "Nx"]:
            if col in self.df.columns:
                size_col = col
                break

        if not size_col:
            print("[!] No se encontró columna de tamaño ('N' o 'Nx') en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        if self.metric_col not in self.df.columns:
            print(f"[!] Columna métrica '{self.metric_col}' no encontrada en los datos. Cancelando gráfica.", file=sys.stderr)
            return

        # Ensure metric and size columns are numeric, and clean the data
        temp_df = self.df.copy()
        temp_df[self.metric_col] = pd.to_numeric(temp_df[self.metric_col], errors='coerce')
        temp_df[size_col] = pd.to_numeric(temp_df[size_col], errors='coerce')

        # Drop rows with NaN values in size or metric columns
        valid_df = temp_df[temp_df[self.metric_col].notna() & temp_df[size_col].notna()]
        
        if valid_df.empty:
            print("[!] No hay mediciones válidas para graficar.", file=sys.stderr)
            return

        # Use slightly wider figure for categorical scale to display tick labels clearly
        fig_width = 12 if x_scale == "categorical" else 10
        plt.figure(figsize=(fig_width, 6))

        # Check grouping columns
        has_device = "Device" in valid_df.columns
        has_precision = "Precision" in valid_df.columns

        devices = valid_df["Device"].unique() if has_device else ["default"]
        
        # Consistent color palette for devices (CPU = Red/Pinkish, GPU = Blue/Cyan)
        colors = {
            "gpu": "#0984e3",  # Blue
            "cpu": "#d63031",  # Red
            "default": "#2d3436"  # Dark gray
        }

        # Style options for precisions to make the plot clean and readable
        linestyles = {
            "S": "-",
            "D": "--",
            "C": "-.",
            "Z": ":"
        }
        markers = {
            "S": "o",  # Circle
            "D": "s",  # Square
            "C": "^",  # Triangle up
            "Z": "D"   # Diamond
        }

        # Compute mapping for categorical scale
        unique_sizes = sorted(valid_df[size_col].unique())
        size_to_idx = {size: idx for idx, size in enumerate(unique_sizes)}

        for dev in sorted(devices):
            dev_df = valid_df[valid_df["Device"] == dev] if has_device else valid_df
            if dev_df.empty:
                continue

            color = colors.get(str(dev).lower(), "#6c5ce7")
            dev_label = str(dev).upper() if has_device else "Total"

            # Check if there are different precisions inside this device group
            precisions = dev_df["Precision"].unique() if has_precision else ["default"]

            for prec in sorted(precisions):
                prec_df = dev_df[dev_df["Precision"] == prec] if has_precision else dev_df
                if prec_df.empty:
                    continue

                prec_label = f" (Precisión {prec})" if has_precision else ""
                linestyle = linestyles.get(str(prec).upper(), "-")
                marker = markers.get(str(prec).upper(), "o")

                if x_scale == "categorical":
                    x = prec_df[size_col].map(size_to_idx)
                else:
                    x = prec_df[size_col]
                y = prec_df[self.metric_col]

                # 1. Scatter plot for individual iterations (capturing noise / variance)
                plt.scatter(
                    x, y,
                    color=color,
                    alpha=0.35,
                    edgecolors="none",
                    s=40,
                    marker=marker,
                    label=f"{dev_label}{prec_label} - Mediciones"
                )

                # 2. Line plot for trend (mean of metric per size)
                grouped = prec_df.groupby(size_col)[self.metric_col].mean().reset_index()
                grouped = grouped.sort_values(by=size_col)

                if x_scale == "categorical":
                    grouped_x = grouped[size_col].map(size_to_idx)
                else:
                    grouped_x = grouped[size_col]

                plt.plot(
                    grouped_x, grouped[self.metric_col],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    marker=marker,
                    markersize=6,
                    label=f"{dev_label}{prec_label} - Promedio"
                )

        title_suffix = ""
        if x_scale == "categorical":
            title_suffix = " (Escala Categórica)"
        elif x_scale == "log":
            title_suffix = " (Escala Logarítmica)"

        plt.title(f"Tiempo de Ejecución Promedio ({self.metric_col}) vs Tamaño del Problema (N){title_suffix}", fontsize=14, fontweight="bold", pad=15)
        
        if x_scale == "categorical":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Índice Categórico]", fontsize=12)
            plt.xticks(
                ticks=range(len(unique_sizes)),
                labels=[str(int(s)) for s in unique_sizes],
                rotation=45,
                ha='right'
            )
        elif x_scale == "log":
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Logarítmica]", fontsize=12)
            is_pow2 = all((int(s) & (int(s) - 1) == 0) and s > 0 for s in unique_sizes if float(s).is_integer())
            if is_pow2:
                plt.xscale("log", base=2)
            else:
                plt.xscale("log")
        else:
            plt.xlabel(f"Tamaño de Problema ({size_col}) [Escala Lineal]", fontsize=12)

        plt.ylabel(f"Tiempo ({self.metric_col})", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Deduplicate legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, facecolor="white", edgecolor="#dfe6e9")

        # Save to file
        try:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[*] Gráfico de tiempo vs N guardado como '{output_path}'")
        except Exception as e:
            print(f"[!] No se pudo guardar la imagen del gráfico de tiempo: {e}", file=sys.stderr)

        # Interactive display
        try:
            if matplotlib.get_backend() != "Agg":
                plt.show()
        except Exception as e:
            pass


def plot_qq_validation(residuos: np.ndarray) -> None:
    """Genera el gráfico Q-Q comparando contra la normal.

    Args:
        residuos (np.ndarray): Vector de residuos de las mediciones.
    """
    if plt is None:
        print("[!] Matplotlib no está instalado. No se puede generar la gráfica Q-Q.", file=sys.stderr)
        return

    plt.figure(figsize=(8, 6))
    
    # Genera el gráfico Q-Q comparando contra la normal
    stats.probplot(residuos, dist="norm", plot=plt)
    
    plt.title('Q-Q Plot: Validación de Residuos (FFT en GPU)')
    plt.xlabel('Cuantiles Teóricos (Distribución Normal)')
    plt.ylabel('Cuantiles de los Datos Observados')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save to file in case of headless terminal
    try:
        plt.savefig("qq_validation.png", dpi=300, bbox_inches="tight")
        print("[*] Gráfico Q-Q guardado como 'qq_validation.png'")
    except Exception as e:
        print(f"[!] No se pudo guardar la imagen del gráfico: {e}", file=sys.stderr)

    try:
        if matplotlib.get_backend() != "Agg":
            plt.show()
    except Exception as e:
        print(f"[!] No se pudo mostrar el gráfico de forma interactiva: {e}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MeasurementValidator CLI Utility")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con mediciones de telemetría")
    parser.add_argument("--metric", default="Time_sec", help="Métrica de rendimiento principal a evaluar (default: Time_sec)")
    parser.add_argument("--temp-col", default=None, help="Columna que contiene la temperatura (opcional)")
    parser.add_argument("--plot", action="store_true", help="Generar gráfico Q-Q de los residuos")
    parser.add_argument("--plot-power", action="store_true", help="Generar gráfico de potencia consumida vs N")
    parser.add_argument("--plot-energy", action="store_true", help="Generar gráfico de energía consumida vs N")
    parser.add_argument("--plot-time", action="store_true", help="Generar gráfico de tiempo de ejecución vs N")
    parser.add_argument("--x-scale", choices=["linear", "log", "categorical"], default="categorical", help="Escala/diseño del eje X para el gráfico de potencia (linear, log, o categorical)")
    
    args = parser.parse_args()

    try:
        validator = MeasurementValidator(
            csv_path=args.csv,
            metric_col=args.metric,
            temp_col=args.temp_col
        )
        validator.print_validation_report()

        if args.plot_power:
            validator.plot_power_vs_n(x_scale=args.x_scale)

        if args.plot_energy:
            validator.plot_energy_vs_n(x_scale=args.x_scale)

        if args.plot_time:
            validator.plot_time_vs_n(x_scale=args.x_scale)

        if args.plot:
            all_residuals = []
            config_cols = validator._detect_config_cols()
            if not config_cols:
                groups = [((), validator.df)]
            else:
                groups = validator.df.groupby(config_cols)

            for _, gp_df in groups:
                vals = gp_df[validator.metric_col]
                if len(vals) > 0:
                    residuals = vals - vals.mean()
                    all_residuals.extend(residuals.tolist())

            if all_residuals:
                plot_qq_validation(np.array(all_residuals))
            else:
                print("[!] No hay suficientes datos para calcular residuos.", file=sys.stderr)

    except Exception as exc:
        print(f"Error en validación: {exc}", file=sys.stderr)
        sys.exit(1)
