# Copilot Cloud Agent Onboarding Instructions

## Repository purpose
- Hybrid CPU/GPU benchmarking project for GEMM and FFT with power/energy metrics and CSV output.
- Main orchestrator: `benchmark_runner.py`.
- Native binaries live in `algoritmos/` (CPU and CUDA implementations).

## Repository layout
- `benchmark_runner.py`: orchestrates benchmark sweeps, monitors power (NVML on GPU, RAPL on CPU), writes CSV.
- `algoritmos/gemm_gpu.cu`, `algoritmos/fft_gpu.cu`: CUDA binaries.
- `algoritmos/gemm_cpu.c`, `algoritmos/fft_cpu.c`: CPU binaries (OpenBLAS/FFTW).
- `bench_files/gen_benchmark_bank.py`: generates HDF5 benchmark bank and text manifest.
- `bench_files/benchmark_bank_manifest.txt`: reference manifest.

## Fast working workflow for agents
1. Read `benchmark_runner.py` first (CLI flags, defaults, expected binary paths).
2. Compile binaries before running sweeps:
   - GPU:
     - `nvcc -O3 -o algoritmos/gemm_gpu algoritmos/gemm_gpu.cu -lcublas`
     - `nvcc -O3 -o algoritmos/fft_gpu algoritmos/fft_gpu.cu -lcufft`
   - CPU:
     - `gcc -O3 -march=native -o algoritmos/gemm_cpu algoritmos/gemm_cpu.c -I/usr/include/openblas -lopenblas -lm`
     - `gcc -O3 -o algoritmos/fft_cpu algoritmos/fft_cpu.c -lfftw3 -lfftw3f -lm`
3. Ensure Python deps are installed in the active environment:
   - `python3 -m pip install pynvml h5py numpy`
4. If benchmark bank is needed and missing, generate it from repo root:
   - `python3 bench_files/gen_benchmark_bank.py`
5. Run benchmark examples:
   - GEMM GPU: `python3 benchmark_runner.py --benchmark gemm --device gpu`
   - GEMM CPU: `python3 benchmark_runner.py --benchmark gemm --device cpu`
   - FFT both: `python3 benchmark_runner.py --benchmark fft --device both`

## Validation commands
- Python syntax check:
  - `python3 -m py_compile benchmark_runner.py bench_files/gen_benchmark_bank.py gpu_watts_en_consola.py`
- CLI sanity:
  - `python3 benchmark_runner.py --help`
- Build checks:
  - Re-run the compile commands above for CPU/GPU binaries.

## Important behavior and constraints
- GEMM supports `S,D,C,Z`; FFT supports `S,D` with domains `C2C,R2C,C2R`.
- Runner expects native binary output containing `Time_sec=...`; keep this output stable when editing C/CUDA code.
- `--binary` is deprecated alias for `--gemm-binary-gpu`; prefer explicit `--gemm-binary-*` and `--fft-binary-*`.
- Default benchmark bank path is `bench_files/benchmark_bank.h5`; it may not exist in fresh clones.
- CPU energy metrics depend on readable RAPL (`/sys/class/powercap/.../energy_uj`); if unavailable, runner continues with 0 power/energy values.

## Errors encountered in clean environment and workarounds
- `ModuleNotFoundError: No module named 'pynvml'` when running `python3 benchmark_runner.py --help`.
  - Workaround: `python3 -m pip install pynvml`.
- `fatal error: cblas.h: No such file or directory` when compiling `algoritmos/gemm_cpu.c`.
  - Workaround: install OpenBLAS development headers (for Debian/Ubuntu: `sudo apt-get install libopenblas-dev`) or adjust include path.
- `fatal error: fftw3.h: No such file or directory` when compiling `algoritmos/fft_cpu.c`.
  - Workaround: install FFTW development headers (for Debian/Ubuntu: `sudo apt-get install libfftw3-dev`).
- `nvcc: command not found` when compiling CUDA binaries.
  - Workaround: use a CUDA-enabled environment with Toolkit installed and `nvcc` on `PATH`.
