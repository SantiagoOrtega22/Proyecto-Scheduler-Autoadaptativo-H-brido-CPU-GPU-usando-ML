"""
estilo_graficas.py

Estilo visual compartido para las graficas del Agente RL (matplotlib).

Centraliza la paleta de colores y el formato para que `graficar_frontera.py`
y `evaluar_agente.py` luzcan consistentes entre si. La paleta pastel no es
un simple aclarado esteticista: cada tono es una version suavizada (mezclada
con blanco o oscurecida levemente) de una paleta categorica validada contra
confusion por daltonismo (deuteranopia/protanopia/tritanopia) y separacion
minima de contraste para vision normal. Antes de tocar los valores hex,
revalidar con el validador de paletas de la skill `dataviz`
(`node scripts/validate_palette.js "<hex,hex,...>" --mode light`).
"""

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer

# Fondo y tinta: superficie casi blanca (no blanco puro, evita el brillo duro
# del fondo por defecto) con textos en gris carbon en vez de negro puro.
COLOR_SUPERFICIE = "#fcfcfb"
COLOR_TEXTO_PRIMARIO = "#2b2b2b"
COLOR_TEXTO_SECUNDARIO = "#595959"
COLOR_GRID = "#dedcd4"
COLOR_EJE = "#b8b6ac"

# Paleta categorica pastel (7 series), en orden fijo. Cada color mantiene
# Delta-E >= 8 (CVD) y >= 15 (vision normal) frente a sus vecinos adyacentes,
# suficiente para barras/lineas ordenadas (no para dispersion todos-contra-todos).
PALETA_CATEGORICA: list[str] = [
    "#4488db",  # azul pastel
    "#ed7a4c",  # naranja pastel
    "#36b98a",  # verde azulado (aqua) pastel
    "#d59100",  # mostaza (mas oscura para no perder croma en amarillo pastel)
    "#eb8baf",  # rosa/magenta pastel
    "#1f921f",  # verde pastel
    "#6052b2",  # violeta pastel
]

# Semantica CPU/GPU consistente entre graficas (rojo=CPU, verde=GPU). Par
# pastel independiente, mas suave que PALETA_CATEGORICA porque solo compite
# consigo mismo (2 series) y ya va acompañado de posicion en el eje Y como
# codificacion redundante en graficar_frontera.py; aun asi valida por encima
# de los pisos minimos (Delta-E 25.9 vision normal, 7.4 CVD).
COLOR_CPU = "#eb807f"
COLOR_GPU = "#4da84d"

# Color de acento para resaltar la barra del Agente RL (la propuesta de la
# tesis) sin romper la paleta: mismo violeta de PALETA_CATEGORICA con un
# borde mas marcado, aplicado en `resaltar_barra`.
COLOR_ACENTO_RL = "#6052b2"


def configurar_estilo_pastel() -> None:
    """Aplica el estilo visual pastel a matplotlib via rcParams.

    Debe llamarse una sola vez, antes de crear cualquier figura. Ajusta
    fondo, tipografia, grid y bordes (spines) de forma global para que todas
    las graficas del modulo compartan la misma identidad visual.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_SUPERFICIE,
            "axes.facecolor": COLOR_SUPERFICIE,
            "savefig.facecolor": COLOR_SUPERFICIE,
            "axes.edgecolor": COLOR_EJE,
            "axes.labelcolor": COLOR_TEXTO_PRIMARIO,
            "text.color": COLOR_TEXTO_PRIMARIO,
            "xtick.color": COLOR_TEXTO_SECUNDARIO,
            "ytick.color": COLOR_TEXTO_SECUNDARIO,
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def limpiar_bordes(ax: plt.Axes) -> None:
    """Quita los bordes superior y derecho, y suaviza los restantes.

    Args:
        ax: Eje de matplotlib a formatear.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_EJE)
    ax.spines["bottom"].set_color(COLOR_EJE)


def etiquetar_barras(
    ax: plt.Axes,
    barras: BarContainer,
    formato: str = "{:.1f}",
    fontsize: int = 9,
    offset_puntos: int = 4,
) -> None:
    """Escribe el valor de cada barra sobre su extremo (etiquetado directo).

    Sirve tambien como mitigacion de accesibilidad: varios tonos de la
    paleta pastel quedan por debajo de 3:1 de contraste contra el fondo
    (ok para relleno de barra, no para texto), asi que el valor exacto no
    depende de distinguir el color a simple vista.

    Args:
        ax: Eje sobre el que estan dibujadas las barras.
        barras: Contenedor devuelto por `ax.bar(...)`.
        formato: Formato de texto para el valor (`str.format`).
        fontsize: Tamaño de fuente de la etiqueta (subir para figuras de poster).
        offset_puntos: Separacion vertical entre la barra y la etiqueta, en puntos.
    """
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(
            formato.format(altura),
            xy=(barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, offset_puntos),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLOR_TEXTO_SECUNDARIO,
        )


def resaltar_barra(barras: BarContainer, indice: int, color_borde: str = COLOR_TEXTO_PRIMARIO) -> None:
    """Marca una barra especifica con un borde mas grueso para destacarla.

    Uso previsto: resaltar la barra del Agente RL en las comparativas, ya
    que es el resultado central de la tesis frente a las heuristicas clasicas.

    Args:
        barras: Contenedor devuelto por `ax.bar(...)`.
        indice: Posicion de la barra a resaltar dentro del contenedor.
        color_borde: Color del borde de realce.
    """
    barra = barras.patches[indice]
    barra.set_edgecolor(color_borde)
    barra.set_linewidth(2.2)
    barra.set_zorder(3)


def paleta_para(n: int) -> Iterable[str]:
    """Devuelve las primeras `n` entradas de la paleta categorica pastel.

    Args:
        n: Cantidad de series a colorear (<= len(PALETA_CATEGORICA)).

    Returns:
        Iterable[str]: Colores en formato hex, en el orden fijo de la paleta.

    Raises:
        ValueError: Si se piden mas series que colores disponibles en la
            paleta validada.
    """
    if n > len(PALETA_CATEGORICA):
        raise ValueError(
            f"Se pidieron {n} colores pero la paleta validada solo tiene {len(PALETA_CATEGORICA)}."
        )
    return PALETA_CATEGORICA[:n]
