"""Poster-scale diagram of the scheduler architecture (Task Queue -> DQN
Policy -> CPU/GPU dispatch, closed by the RAPL/NVML-measured EDP reward).
Recreated at larger scale (no legend/subtitle, bigger titles) to read well
on a printed poster."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = "scheduler_architecture.png"

NAVY_FILL = "#dce9f7"
NAVY_EDGE = "#1d4e77"
NAVY_TEXT = "#173553"

DQN_FILL = "#173553"
DQN_TEXT = "#ffffff"

GREEN_FILL = "#e1f3e1"
GREEN_EDGE = "#2e7d32"
GREEN_TEXT = "#2e7d32"

PURPLE_FILL = "#f3e6fa"
PURPLE_EDGE = "#7b2fa0"
PURPLE_TEXT = "#7b2fa0"

GRAY_FILL = "#eeeeee"
GRAY_EDGE = "#555555"
GRAY_TEXT = "#333333"

ORANGE_FILL = "#fbe3c4"
ORANGE_EDGE = "#c5670e"
ORANGE_TEXT = "#9a4c08"

TITLE_COLOR = "#173553"

fig, ax = plt.subplots(figsize=(24, 11), dpi=300)
ax.set_xlim(0, 24)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, text, fill, edge, textcolor, fontsize=22, lw=2.6):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             fontweight="bold", color=textcolor, zorder=4, linespacing=1.3)
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def arrow(p_from, p_to, color, lw=3.2, style="-", connectionstyle="arc3,rad=0.0",
          shrink=2):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=28,
        color=color, linewidth=lw, linestyle=style,
        connectionstyle=connectionstyle, shrinkA=shrink, shrinkB=shrink, zorder=2,
    )
    ax.add_patch(a)


# --- top flow row ---
y_top = 8.0
h_top = 2.6

tq = box(2.2, y_top, 3.6, h_top, "Task\nQueue", NAVY_FILL, NAVY_EDGE, NAVY_TEXT)
enc = box(6.8, y_top, 4.2, h_top, "23-Feature\nState Encoder", NAVY_FILL, NAVY_EDGE, NAVY_TEXT)
dqn = box(11.8, y_top, 4.2, h_top + 0.4, "DQN\nPolicy", DQN_FILL, DQN_FILL, DQN_TEXT, fontsize=24)

cpu_d = box(16.6, y_top + 1.05, 3.4, 1.5, "CPU Dispatch", GREEN_FILL, GREEN_EDGE, GREEN_TEXT, fontsize=19)
gpu_d = box(16.6, y_top - 1.05, 3.4, 1.5, "GPU Dispatch", PURPLE_FILL, PURPLE_EDGE, PURPLE_TEXT, fontsize=19)

lut = box(21.2, y_top, 3.6, 2.6, "Benchmark\nLookup Table\n(offline)", GRAY_FILL, GRAY_EDGE, GRAY_TEXT, fontsize=19)

arrow((tq[1], y_top), (enc[0], y_top), NAVY_EDGE)
arrow((enc[1], y_top), (dqn[0], y_top), NAVY_EDGE)
arrow((dqn[1], y_top + 1.05), (cpu_d[0], y_top + 1.05), NAVY_EDGE)
arrow((dqn[1], y_top - 1.05), (gpu_d[0], y_top - 1.05), NAVY_EDGE)
arrow((cpu_d[1], y_top + 1.05), (lut[0], y_top + 0.35), NAVY_EDGE)
arrow((gpu_d[1], y_top - 1.05), (lut[0], y_top - 0.35), NAVY_EDGE)

# --- bottom energy-loop row ---
y_bot = 3.1

edp = box(4.4, y_bot, 8.0, 2.4,
          "EDP Reward\n$R = (EDP_{worse} - EDP)\\ /\\ EDP_{worse}$",
          ORANGE_FILL, ORANGE_EDGE, ORANGE_TEXT, fontsize=21)
rapl = box(13.2, y_bot, 5.0, 2.0, "RAPL-measured\nCPU energy", ORANGE_FILL, ORANGE_EDGE, ORANGE_TEXT, fontsize=19)
nvml = box(19.4, y_bot, 5.0, 2.0, "NVML-measured\nGPU energy", ORANGE_FILL, ORANGE_EDGE, ORANGE_TEXT, fontsize=19)

# Lookup table feeds the two energy sources
arrow((lut[0] + 0.6, lut[2]), (rapl[1] - 0.6, rapl[3] + 0.05), ORANGE_EDGE,
      connectionstyle="arc3,rad=-0.15")
arrow((lut[1] - 0.2, lut[2]), (nvml[1] - 0.2, nvml[3]), ORANGE_EDGE,
      connectionstyle="arc3,rad=-0.05")

# RAPL feeds the reward directly; NVML dips below RAPL's box to reach the
# reward too, so both energy sources visibly land on the same target.
arrow((rapl[0], y_bot), (edp[1], y_bot), ORANGE_EDGE)
arrow((nvml[0], y_bot - 1.0), (edp[1] + 0.3, y_bot - 1.1), ORANGE_EDGE,
      connectionstyle="arc3,rad=-0.35")

# Feedback loop: reward -> DQN policy (dashed). Departs from the top of the
# EDP box and arrives vertically (angleB=90) at the bottom-center of the DQN
# box, so the line approaches head-on instead of grazing along the box's
# bottom edge (which hid it behind the box in an earlier version).
arrow((edp[0] + 2.5, edp[3]), (11.8, dqn[2]), ORANGE_EDGE, lw=3.0,
      style=(0, (7, 5)),
      connectionstyle="arc,angleA=90,angleB=90,armA=40,armB=60,rad=20")

# --- title ---
fig.suptitle("Scheduler Architecture", fontsize=46, fontweight="bold",
             color=TITLE_COLOR, y=0.99)

fig.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.98)
fig.savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight")
print("saved", OUT_PATH)
