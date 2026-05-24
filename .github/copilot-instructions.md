# Copilot Cloud Agent Onboarding Instructions
# Software Design Description: Benchmark for Reinforcement Learning Data Collection

---

## Section 1 - Product Vision
The objective is to create a benchmark capable of collecting CPU and GPU measurements resulting from the execution of **GEMM** and **FFT** algorithms. This tool is part of a degree project where a reinforcement learning agent will be trained using the collected data.

---

## Section 2 - Users and Use Cases
The benchmark module is used by students for the data collection process required to subsequently train a reinforcement learning agent.

---

## Section 3 - Functionalities

* **Data Sourcing**: The benchmark must retrieve the necessary matrix or array from the data bank for test execution.
* **GEMM Variations**: GEMM codes must include their different variations (**sgemm, dgemm, cgemm, zgemm**) and vary their internal parameters, such as transposes.
* **FFT Variations**: FFT codes must include different variations (**1D, 2D, 3D**) with the possibility of varying internal parameters like **R2C** and **C2C**.
* **Metric Measurement**:
    * **GPU**: Execution time must be measured via **NVML**, taking into account transfer times.
    * **CPU**: Execution time must be measured via **chronos**.
* **Execution Modes**:
    * The benchmark must execute the respective GEMM and FFT binaries on each device (CPU and GPU) according to the requested execution mode.
    * The benchmark will vary the size **N** (NxN matrix for GEMM and array size for FFT).
    * For cases involving both devices, the benchmark must alternate execution between one and the other for each **N**.
* **Rigorous Measurement Protocol**:
    * **Warm-up**: 4 warm-up executions must be performed before each measurement.
    * **Metric Isolation**: To avoid measurement errors, a first execution will be performed without energy tools active to measure only time and calculate **GFLOPS**.
    * **Power Monitoring**: A second execution of the exact same experiment (same size N) will follow with continuous power monitoring active (using **RAPL** or `nvmlDeviceGetPowerUsage`).
* **Output and Reporting**:
    * **Required Data**: The benchmark must provide: Device, N, Time (ms), Average Power (W), Energy (J), EDP (J*s), and GFLOPS.
    * **Formats**: Data must be printed in real-time to the console in table format and subsequently saved to a **.csv** file.
    * **Metadata**: The .csv must store the execution parameters (e.g., for GEMM: sgemm, transA: N, transB: T, etc.).

---

## Section 4 - User Flows

Students may define different execution modes:
* **Full Execution (Default)**: A complete sweep executing all algorithm types on each device, varying all parameters.
* **Quick Test**: Uses small sizes of **N** without varying parameters.
* **Only CPU**: All tests are executed on the CPU.
* **Only GPU**: All tests are executed on the GPU.
* **Only GEMM**: Only GEMM-related tests are executed.
* **Only FFT**: Only FFT-related tests are executed.

---

## Section 5 - Architecture

The system consists of a data bank containing the matrices/arrays for each size and data type. Specific codes exist for each GEMM or FFT function. The benchmark orchestrates the execution of these binaries for the respective device and algorithm, collects measurements, and saves them.

**Technology Stack**:
* **Benchmark Orchestrator**: Python.
* **GEMM CPU**: MKL / OpenBLAS.
* **GEMM GPU**: cuBLAS.
* **FFT CPU**: FFTw.
* **FFT GPU**: cuFFT.
* **Measurement Tools**: RAPL (CPU), NVML (GPU), chronos.

---

## Section 6 - Non-Functional Requirements

* **Reliability**: All measurements taken must be reliable and free from measurement risks or errors.
