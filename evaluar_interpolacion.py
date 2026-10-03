"""
evaluar_interpolacion.py

Prueba de interpolación del agente DQN sobre tamaños de problema no observados
dentro de los rangos caracterizados (revisión P3/P4).

Procedimiento
-------------
1. Los tamaños de cada carga (N en GEMM y FFT 2D/3D, Nx en FFT 1D) se reparten en
   5 particiones (PlanificadorEnv.asignar_particiones_por_tamano): todas las
   variantes de un tamaño van a la misma partición, dentro de cada octava los
   tamaños se alternan entre particiones y el mínimo y el máximo de cada carga
   quedan siempre en el entrenamiento. Es una prueba de interpolación, no de
   extrapolación.
2. Para cada partición k y cada semilla s, el agente se entrena con train.py
   (mismos hiperparámetros del libro) sobre las otras particiones y se evalúa en
   modo determinista sobre la partición k: 5 particiones x 5 semillas = 25 corridas.
3. En las mismas particiones se evalúan referencias que no requieren RL:
   Solo GPU, regla de umbral de tamaño por carga y precisión, vecino más cercano
   (la tabla de consulta extendida a tamaños no medidos), y, si scikit-learn está
   instalado, un árbol de decisión y una regresión logística (línea base supervisada).
4. Fase 4 sobre tamaños no observados: para cada partición se arma una cola de
   200 tareas con semilla BENCH_SEED a partir de las tareas reservadas y se calcula
   el EDP de sistema de cada política, con la misma lógica de evaluar_agente.py.

Salidas (en resultados_interpolacion/)
--------------------------------------
- corridas.csv: una fila por (método, partición, semilla) con las métricas.
- fase4_colas.csv: EDP de sistema por política y partición.
- resumen.json: medias y desviaciones usadas en el libro.
- tabla_interpolacion.tex, tabla_fase4_interpolacion.tex: tablas LaTeX.

Uso
---
    python evaluar_interpolacion.py                       # 25 corridas (~30 min)
    python evaluar_interpolacion.py --solo-referencias    # sin entrenar (segundos)
    python evaluar_interpolacion.py --semillas 42         # 5 corridas

No modifica ni sobrescribe el modelo del libro: los modelos de esta prueba se
guardan en modelos_interpolacion/ y sus registros en logs_entrenamiento/interpolacion/.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Callable

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entorno.gym import PlanificadorEnv  # noqa: E402

BENCH_SEED = 42  # Igual que train.BENCH_SEED; se repite para no importar SB3 en --solo-referencias.
N_PARTICIONES = 5
SEMILLAS: list[int] = [42, 43, 44, 45, 46]  # Mismas semillas que barrido_semillas.py
TAMANO_COLA = 200
CARGAS = ("GEMM", "FFT1D", "FFT2D", "FFT3D")

# Índices del vector de observación (ver Tabla de features del libro).
IDX_TAMANO = (2, 3, 4)
IDX_PRECISION = slice(6, 10)
IDX_RADIX = 22

# Un entrenador recibe (ruta_dataset, particion, semilla, timesteps) y devuelve una
# función que mapea una matriz de observaciones (n x 23) a acciones (n,).
Entrenador = Callable[[str, int, int, int], Callable[[np.ndarray], np.ndarray]]


# =============================================================================
# Datos
# =============================================================================

def cargar_arreglos(csv_path: str) -> dict:
    """Carga el dataset con el mismo lector del entorno y lo pasa a arreglos.

    Args:
        csv_path: CSV codificado (salida de codificador_csv.py).

    Returns:
        dict: X (n x 23), edp/t/e (n x 2, columna 0 = CPU, 1 = GPU), oraculo (n,),
            carga (n,), tamano (n,) con Dim_1_log2, particion (n,) y tareas (lista
            original de diccionarios del entorno).
    """
    tareas = PlanificadorEnv(csv_path=csv_path).dataset_tareas
    X = np.stack([t["obs"] for t in tareas]).astype(np.float64)
    edp = np.array([[t["metricas"][0]["edp"], t["metricas"][1]["edp"]] for t in tareas])
    tiempo = np.array([[t["metricas"][0]["tiempo"], t["metricas"][1]["tiempo"]] for t in tareas])
    energia = np.array([[t["metricas"][0]["energia"], t["metricas"][1]["energia"]] for t in tareas])
    claves = [PlanificadorEnv.clave_tamano(t["obs"]) for t in tareas]
    return {
        "X": X,
        "edp": edp,
        "t": tiempo,
        "e": energia,
        # Mismo criterio que gym.py: empate -> CPU.
        "oraculo": np.where(edp[:, 0] <= edp[:, 1], 0, 1),
        "carga": np.array([c[0] for c in claves]),
        "tamano": np.array([c[1] for c in claves]),
        "particion": np.array(PlanificadorEnv.asignar_particiones_por_tamano(tareas, N_PARTICIONES)),
        "tareas": tareas,
    }


def clave_configuracion(obs: np.ndarray, carga: str) -> tuple:
    """Configuración de la tarea sin el tamaño: carga + precisión, transposición,
    dominio, dirección y layout (índices 5..21, que no dependen de N)."""
    return (carga,) + tuple(np.round(obs[5:22], 6))


# =============================================================================
# Referencias sin aprendizaje por refuerzo
# =============================================================================

def politica_solo_gpu(d: dict, entr: np.ndarray, prueba: np.ndarray) -> np.ndarray:
    """Envía todas las tareas a la GPU."""
    return np.ones(prueba.sum(), dtype=int)


def politica_umbral(d: dict, entr: np.ndarray, prueba: np.ndarray) -> np.ndarray:
    """Regla de umbral de tamaño calibrada solo con los tamaños de entrenamiento.

    Para cada combinación (carga, precisión) se elige el umbral u que minimiza los
    errores de entrenamiento de la regla «GPU si tamaño >= u, CPU en otro caso».
    Generaliza el punto de equilibrio N* de la Tabla de GEMM a las cuatro cargas.
    """
    X, y, tam, carga = d["X"], d["oraculo"], d["tamano"], d["carga"]
    prec = np.argmax(X[:, IDX_PRECISION], axis=1)
    acciones = np.ones(len(X), dtype=int)
    for c in CARGAS:
        for p in range(4):
            g_entr = entr & (carga == c) & (prec == p)
            g_prueba = prueba & (carga == c) & (prec == p)
            if not g_prueba.any():
                continue
            if not g_entr.any():
                continue
            s, lab = tam[g_entr], y[g_entr]
            valores = np.unique(s)
            # Candidatos: debajo de todo (todo GPU), entre valores consecutivos y encima (todo CPU).
            candidatos = np.concatenate(([valores[0] - 1.0], (valores[:-1] + valores[1:]) / 2, [valores[-1] + 1.0]))
            errores = [np.sum((s >= u) != (lab == 1)) for u in candidatos]
            u = candidatos[int(np.argmin(errores))]
            acciones[g_prueba] = (tam[g_prueba] >= u).astype(int)
    return acciones[prueba]


def politica_vecino(d: dict, entr: np.ndarray, prueba: np.ndarray) -> np.ndarray:
    """Vecino más cercano: tabla de consulta extendida a tamaños no medidos.

    Para cada tarea reservada toma el dispositivo óptimo del tamaño de entrenamiento
    más cercano (en log2 N) con la misma configuración. En empate de distancia se
    toma el tamaño menor.
    """
    X, y, tam, carga = d["X"], d["oraculo"], d["tamano"], d["carga"]
    tabla: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    indices_entr = np.flatnonzero(entr)
    agrupado: dict[tuple, list[int]] = {}
    for i in indices_entr:
        agrupado.setdefault(clave_configuracion(X[i], carga[i]), []).append(i)
    for k, idx in agrupado.items():
        idx = np.array(idx)
        orden = np.argsort(tam[idx])
        tabla[k] = (tam[idx][orden], y[idx][orden])
    salida = []
    for i in np.flatnonzero(prueba):
        tams, labs = tabla[clave_configuracion(X[i], carga[i])]
        j = int(np.searchsorted(tams, tam[i]))
        if j == 0:
            salida.append(labs[0])
        elif j == len(tams):
            salida.append(labs[-1])
        else:
            izq, der = tam[i] - tams[j - 1], tams[j] - tam[i]
            salida.append(labs[j - 1] if izq <= der else labs[j])
    return np.array(salida, dtype=int)


def _politicas_sklearn() -> dict[str, Callable]:
    """Árbol de decisión y regresión logística, si scikit-learn está disponible."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print("[!] scikit-learn no está instalado: se omiten árbol y regresión logística.")
        return {}

    profundidades = [2, 4, 6, 8, 10, 12, 15, 20, None]

    def politica_arbol(d: dict, entr: np.ndarray, prueba: np.ndarray) -> np.ndarray:
        """Árbol de decisión; profundidad elegida por validación cruzada interna
        dejando fuera, una a la vez, cada partición de entrenamiento."""
        X, y, part = d["X"], d["oraculo"], d["particion"]
        internas = sorted(set(part[entr].tolist()) - {-1})
        mejor, mejor_prec = None, -1.0
        for prof in profundidades:
            precs = []
            for q in internas:
                tr = entr & (part != q)
                va = entr & (part == q)
                m = DecisionTreeClassifier(max_depth=prof, random_state=BENCH_SEED).fit(X[tr], y[tr])
                precs.append(np.mean(m.predict(X[va]) == y[va]))
            if np.mean(precs) > mejor_prec:
                mejor, mejor_prec = prof, float(np.mean(precs))
        m = DecisionTreeClassifier(max_depth=mejor, random_state=BENCH_SEED).fit(X[entr], y[entr])
        politica_arbol.profundidades.append(mejor)
        return m.predict(X[prueba]).astype(int)

    politica_arbol.profundidades = []

    def politica_logistica(d: dict, entr: np.ndarray, prueba: np.ndarray) -> np.ndarray:
        """Regresión logística sobre los mismos 23 rasgos."""
        m = LogisticRegression(max_iter=5000).fit(d["X"][entr], d["oraculo"][entr])
        return m.predict(d["X"][prueba]).astype(int)

    return {"Árbol de decisión": politica_arbol, "Regresión logística": politica_logistica}


# =============================================================================
# Métricas
# =============================================================================

def metricas(acciones: np.ndarray, d: dict, mascara: np.ndarray) -> dict[str, float]:
    """Precisión frente al oráculo, % a GPU, exceso de suma de EDP y pérdida de recompensa.

    Args:
        acciones: Acciones (0 CPU, 1 GPU) para las tareas de la máscara, en orden.
        d: Arreglos del dataset.
        mascara: Tareas evaluadas.

    Returns:
        dict: precision, pct_gpu, exceso_edp (sum EDP / sum EDP oráculo - 1, en %;
            dominado por las tareas grandes, casi siempre ~0), perdida_recompensa
            (media de R(a*) - R(a), el objetivo de entrenamiento), sobrecosto_geom
            (media geométrica de EDP(a)/EDP(a*) - 1, en %; cada tarea pesa igual) y
            precisión por carga.
    """
    edp = d["edp"][mascara]
    opt = d["oraculo"][mascara]
    n = len(acciones)
    edp_pol = edp[np.arange(n), acciones]
    edp_opt = edp[np.arange(n), opt]
    peor = edp.max(axis=1)
    peor_seguro = np.where(peor > 0, peor, 1.0)
    r_pol = np.where(peor > 0, (peor - edp_pol) / peor_seguro, 1.0)
    r_opt = np.where(peor > 0, (peor - edp_opt) / peor_seguro, 1.0)
    salida = {
        "n": n,
        "precision": 100.0 * float(np.mean(acciones == opt)),
        "pct_gpu": 100.0 * float(np.mean(acciones == 1)),
        "exceso_edp": 100.0 * float(edp_pol.sum() / edp_opt.sum() - 1.0),
        "perdida_recompensa": float(np.mean(r_opt - r_pol)),
        "sobrecosto_geom": 100.0 * float(np.exp(np.mean(np.log(edp_pol / edp_opt))) - 1.0),
    }
    carga = d["carga"][mascara]
    for c in CARGAS:
        sel = carga == c
        salida[f"precision_{c}"] = 100.0 * float(np.mean(acciones[sel] == opt[sel])) if sel.any() else float("nan")
    return salida


# =============================================================================
# Fase 4: EDP de sistema (misma lógica y desempates que evaluar_agente.py)
# =============================================================================

def sistema_por_acciones(acciones: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[float, float, float]:
    """Tiempo (makespan de colas paralelas), energía y EDP de sistema."""
    idx = np.arange(len(acciones))
    t_cpu = float(t[idx, 0][acciones == 0].sum())
    t_gpu = float(t[idx, 1][acciones == 1].sum())
    energia = float(e[idx, acciones].sum())
    tiempo = max(t_cpu, t_gpu)
    return tiempo, energia, energia * tiempo


def acciones_met(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """MET: menor tiempo de ejecución por tarea (empate -> CPU)."""
    return np.where(t[:, 0] <= t[:, 1], 0, 1)


def acciones_mct(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """MCT: menor tiempo de finalización considerando las colas (empate -> CPU)."""
    libre = [0.0, 0.0]
    acc = []
    for tc, tg in t:
        a = 0 if libre[0] + tc <= libre[1] + tg else 1
        libre[a] += tc if a == 0 else tg
        acc.append(a)
    return np.array(acc)


def _acciones_lote(t: np.ndarray, maximo: bool) -> np.ndarray:
    """Min-Min (maximo=False) o Max-Min (maximo=True), con los desempates de evaluar_agente.py."""
    libre = [0.0, 0.0]
    pendientes = list(range(len(t)))
    acc = np.zeros(len(t), dtype=int)
    while pendientes:
        mejor_pos, mejor_dev, mejor_val = -1, -1, (-1.0 if maximo else float("inf"))
        for pos, i in enumerate(pendientes):
            ct_cpu, ct_gpu = libre[0] + t[i, 0], libre[1] + t[i, 1]
            local = min(ct_cpu, ct_gpu)
            if (local > mejor_val) if maximo else (local < mejor_val):
                mejor_val, mejor_pos, mejor_dev = local, pos, (0 if ct_cpu <= ct_gpu else 1)
        i = pendientes.pop(mejor_pos)
        acc[i] = mejor_dev
        libre[mejor_dev] += t[i, mejor_dev]
    return acc


def construir_cola(csv_path: str, particion: int) -> list[dict]:
    """Cola de TAMANO_COLA tareas reservadas, muestreada como en evaluar_agente.py."""
    env = PlanificadorEnv(
        csv_path=csv_path, tamano_lote=TAMANO_COLA, shuffle=True,
        modo_particion="tamano", particion=particion, n_particiones=N_PARTICIONES, split="holdout",
    )
    env.reset(seed=BENCH_SEED)
    return list(env.cola_tareas)


def evaluar_cola(cola: list[dict], predictores: dict[str, Callable[[np.ndarray], np.ndarray]]) -> dict[str, tuple]:
    """EDP de sistema de cada política sobre una cola."""
    X = np.stack([c["obs"] for c in cola]).astype(np.float64)
    t = np.array([[c["metricas"][0]["tiempo"], c["metricas"][1]["tiempo"]] for c in cola])
    e = np.array([[c["metricas"][0]["energia"], c["metricas"][1]["energia"]] for c in cola])
    edp = np.array([[c["metricas"][0]["edp"], c["metricas"][1]["edp"]] for c in cola])
    acciones = {
        "Solo CPU": np.zeros(len(cola), dtype=int),
        "Solo GPU": np.ones(len(cola), dtype=int),
        "MET": acciones_met(t, e),
        "MCT": acciones_mct(t, e),
        "Min-Min": _acciones_lote(t, maximo=False),
        "Max-Min": _acciones_lote(t, maximo=True),
        "Oráculo por tarea": np.where(edp[:, 0] <= edp[:, 1], 0, 1),
    }
    for nombre, f in predictores.items():
        acciones[nombre] = np.asarray(f(X), dtype=int)
    return {k: sistema_por_acciones(a, t, e) + (int(a.sum()),) for k, a in acciones.items()}


# =============================================================================
# Agente (Stable-Baselines3, el mismo train.py del libro)
# =============================================================================

def entrenador_sb3(csv_path: str, particion: int, semilla: int, timesteps: int) -> Callable[[np.ndarray], np.ndarray]:
    """Entrena con train.entrenar_agente sobre las particiones de entrenamiento."""
    from stable_baselines3 import DQN
    from train import entrenar_agente

    ruta = entrenar_agente(
        split="train",
        modo_particion="tamano",
        particion=particion,
        n_particiones=N_PARTICIONES,
        modelo_nombre=os.path.join("modelos_interpolacion", f"dqn_p{particion}_s{semilla}"),
        log_subdir=os.path.join("interpolacion", f"p{particion}_s{semilla}"),
        seed=semilla,
        timesteps=timesteps,
        dataset=csv_path,
    )
    modelo = DQN.load(ruta)

    def predecir(X: np.ndarray) -> np.ndarray:
        acciones, _ = modelo.predict(X.astype(np.float32), deterministic=True)
        return np.asarray(acciones, dtype=int).reshape(-1)

    return predecir


# =============================================================================
# Orquestación
# =============================================================================

def _media_de(filas: list[dict], metodo: str, clave: str) -> tuple[float, float]:
    vals = [f[clave] for f in filas if f["metodo"] == metodo]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def ejecutar_prueba(
    csv_path: str,
    salida_dir: str,
    semillas: list[int],
    timesteps: int = 100_000,
    entrenador: Entrenador | None = entrenador_sb3,
) -> dict:
    """Corre la prueba completa y escribe los resultados.

    Args:
        csv_path: Dataset codificado (el mismo con el que se entrenó el modelo del libro).
        salida_dir: Carpeta de resultados.
        semillas: Semillas del DQN.
        timesteps: Pasos de entrenamiento por corrida (100 000 en el libro).
        entrenador: Función de entrenamiento del agente; None omite el agente.

    Returns:
        dict: Resumen (también se guarda en resumen.json).
    """
    os.makedirs(salida_dir, exist_ok=True)
    d = cargar_arreglos(csv_path)
    part = d["particion"]
    print(f"Tareas: {len(part)} | por partición: "
          + ", ".join(f"{k}={int(np.sum(part == k))}" for k in range(N_PARTICIONES))
          + f" | extremos siempre en entrenamiento: {int(np.sum(part == -1))}")

    referencias = {"Solo GPU": politica_solo_gpu, "Umbral de tamaño": politica_umbral,
                   "Vecino más cercano": politica_vecino}
    referencias.update(_politicas_sklearn())

    filas: list[dict] = []
    filas_fase4: list[dict] = []
    t_inicio = time.time()
    for k in range(N_PARTICIONES):
        entr, prueba = part != k, part == k
        for nombre, f in referencias.items():
            m = metricas(f(d, entr, prueba), d, prueba)
            filas.append({"metodo": nombre, "particion": k, "semilla": "", **m})
            print(f"[p{k}] {nombre:22s} precisión {m['precision']:6.2f}%  sobrecosto EDP {m['sobrecosto_geom']:6.2f}%")

        predictores: dict[str, Callable] = {}
        if entrenador is not None:
            for s in semillas:
                t0 = time.time()
                pred = entrenador(csv_path, k, s, timesteps)
                m = metricas(pred(d["X"][prueba]), d, prueba)
                m_entr = metricas(pred(d["X"][entr]), d, entr)
                filas.append({"metodo": "Agente DQN", "particion": k, "semilla": s, **m,
                              "precision_entrenamiento": m_entr["precision"],
                              "segundos": round(time.time() - t0, 1)})
                predictores[f"Agente DQN (s{s})"] = pred
                print(f"[p{k} s{s}] Agente DQN           precisión {m['precision']:6.2f}%  "
                      f"(entrenamiento {m_entr['precision']:.2f}%)  sobrecosto EDP {m['sobrecosto_geom']:6.2f}%  "
                      f"[{time.time() - t0:.0f}s, total {(time.time() - t_inicio) / 60:.1f} min]")

        cola = construir_cola(csv_path, k)
        for pol, (T, E, EDP, n_gpu) in evaluar_cola(cola, predictores).items():
            filas_fase4.append({"particion": k, "politica": pol, "T_sis": T, "E_sis": E, "EDP_sis": EDP, "tareas_gpu": n_gpu})

    # ---------------- Archivos de salida
    columnas = sorted({c for f in filas for c in f}, key=lambda c: (c not in ("metodo", "particion", "semilla"), c))
    with open(os.path.join(salida_dir, "corridas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(salida_dir, "fase4_colas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_fase4[0].keys()))
        w.writeheader()
        w.writerows(filas_fase4)

    metodos = list(dict.fromkeys(f["metodo"] for f in filas))
    resumen: dict = {"dataset": os.path.abspath(csv_path), "semillas": semillas, "timesteps": timesteps,
                     "tareas_por_particion": {k: int(np.sum(part == k)) for k in range(N_PARTICIONES)},
                     "metodos": {}}
    for mtd in metodos:
        r = {}
        for clave in ["precision", "pct_gpu", "exceso_edp", "perdida_recompensa", "sobrecosto_geom"] + [
            f"precision_{c}" for c in CARGAS
        ]:
            media, sd = _media_de(filas, mtd, clave)
            r[clave] = {"media": media, "sd": sd}
        if mtd == "Agente DQN":
            r["precision_entrenamiento"] = dict(zip(("media", "sd"), _media_de(filas, mtd, "precision_entrenamiento")))
            # Desviación entre semillas promediada sobre particiones (variabilidad del aprendizaje).
            sd_sem = [np.std([f["precision"] for f in filas if f["metodo"] == mtd and f["particion"] == k], ddof=1)
                      for k in range(N_PARTICIONES)] if len(semillas) > 1 else [0.0]
            r["sd_entre_semillas_precision"] = float(np.mean(sd_sem))
        resumen["metodos"][mtd] = r
    arbol = referencias.get("Árbol de decisión")
    if arbol is not None:
        resumen["profundidad_arbol_por_particion"] = [str(p) for p in arbol.profundidades]

    # Fase 4: el agente se resume como la media sobre semillas en cada partición.
    f4: dict[str, list[float]] = {}
    for k in range(N_PARTICIONES):
        filas_k = [f for f in filas_fase4 if f["particion"] == k]
        agentes = [f["EDP_sis"] for f in filas_k if f["politica"].startswith("Agente DQN")]
        if agentes:
            f4.setdefault("Agente DQN", []).append(float(np.mean(agentes)))
        for f in filas_k:
            if not f["politica"].startswith("Agente DQN"):
                f4.setdefault(f["politica"], []).append(f["EDP_sis"])
    base = "Agente DQN" if "Agente DQN" in f4 else "Solo GPU"
    resumen["fase4"] = {"referencia": base, "politicas": {}}
    for pol, vals in f4.items():
        dif = [100.0 * (v / b - 1.0) for v, b in zip(vals, f4[base])]
        resumen["fase4"]["politicas"][pol] = {"EDP_sis_por_particion": vals,
                                               "dif_vs_referencia_media": float(np.mean(dif)),
                                               "dif_vs_referencia_sd": float(np.std(dif, ddof=1)),
                                               "dif_min": float(np.min(dif)), "dif_max": float(np.max(dif))}

    with open(os.path.join(salida_dir, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    escribir_tablas_latex(resumen, salida_dir)
    print(f"\nResultados en {salida_dir} ({(time.time() - t_inicio) / 60:.1f} min).")
    return resumen


def escribir_tablas_latex(resumen: dict, salida_dir: str) -> None:
    """Escribe las dos tablas del libro a partir del resumen."""
    def pm(x: dict, dec: int = 2) -> str:
        return f"{x['media']:.{dec}f}" if x["sd"] == 0 else f"{x['media']:.{dec}f} $\\pm$ {x['sd']:.{dec}f}"

    orden = ["Agente DQN", "Vecino más cercano", "Árbol de decisión", "Umbral de tamaño",
             "Regresión logística", "Solo GPU"]
    lineas = [
        "\\begin{table}[H]", "\\centering", "\\small", "\\renewcommand{\\arraystretch}{1.3}",
        "\\caption{Desempeño sobre tamaños no observados dentro de los rangos caracterizados "
        "(5 particiones por tamaño; el agente, además, con 5 semillas). Media $\\pm$ desviación estándar.}",
        "\\label{tab:interpolacion}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{|l|c|c|c|c|c|c|c|}", "\\hline",
        "\\textbf{Política} & \\textbf{Precisión (\\%)} & \\textbf{GEMM (\\%)} & \\textbf{FFT 1D (\\%)} & "
        "\\textbf{FFT 2D (\\%)} & \\textbf{FFT 3D (\\%)} & \\textbf{Tareas a GPU (\\%)} & "
        "\\textbf{Sobrecosto de EDP (\\%)} \\\\ \\hline",
    ]
    for mtd in orden:
        r = resumen["metodos"].get(mtd)
        if r is None:
            continue
        nombre = f"\\textbf{{{mtd}}}" if mtd == "Agente DQN" else mtd
        lineas.append(" & ".join([nombre, pm(r["precision"]), pm(r["precision_GEMM"]), pm(r["precision_FFT1D"]),
                                  pm(r["precision_FFT2D"]), pm(r["precision_FFT3D"]), pm(r["pct_gpu"], 1),
                                  pm(r["sobrecosto_geom"], 2)]) + " \\\\ \\hline")
    lineas += ["\\end{tabular}%", "}", "\\end{table}",
               "% FUENTE: evaluar_interpolacion.py -> resultados_interpolacion/resumen.json"]
    with open(os.path.join(salida_dir, "tabla_interpolacion.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")

    f4 = resumen["fase4"]["politicas"]
    ref = resumen["fase4"]["referencia"]
    lineas = [
        "\\begin{table}[H]", "\\centering", "\\renewcommand{\\arraystretch}{1.3}",
        f"\\caption{{EDP de sistema sobre colas de 200 tareas de tamaños no observados (una cola por partición): "
        f"diferencia relativa respecto {'del agente' if ref == 'Agente DQN' else 'de ' + ref}.}}",
        "\\label{tab:fase4_interpolacion}", "\\begin{tabular}{|l|c|c|}", "\\hline",
        "\\textbf{Política} & \\textbf{Diferencia media (\\%)} & \\textbf{Rango [mín, máx] (\\%)} \\\\ \\hline",
    ]
    for pol in ["Solo CPU", "Solo GPU", "MET", "MCT", "Min-Min", "Max-Min", "Oráculo por tarea", "Agente DQN"]:
        if pol in f4:
            v = f4[pol]
            lineas.append(f"{pol} & {v['dif_vs_referencia_media']:+.2f} & [{v['dif_min']:+.2f}, {v['dif_max']:+.2f}] \\\\ \\hline")
    lineas += ["\\end{tabular}", "\\end{table}"]
    with open(os.path.join(salida_dir, "tabla_fase4_interpolacion.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Prueba de interpolación por tamaño (revisión P3).")
    parser.add_argument("--dataset", default=os.path.join(base_dir, "Entorno", "dataset_rl.csv"),
                        help="CSV codificado; debe ser el mismo del modelo del libro.")
    parser.add_argument("--salida", default=os.path.join(base_dir, "resultados_interpolacion"))
    parser.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--solo-referencias", action="store_true", help="No entrena el agente.")
    args = parser.parse_args()
    ejecutar_prueba(args.dataset, args.salida, args.semillas, args.timesteps,
                    entrenador=None if args.solo_referencias else entrenador_sb3)
