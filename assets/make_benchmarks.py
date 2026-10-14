#!/usr/bin/env python3
"""Generate the LLMZero benchmark results figure for the OSS README.

Two public tasks (GRPO on SSMR-Bench, SFT on GovReport), three conditions each
(Base model / verl-default no-search / LLMZero MCTS search). Because the two
metrics live on different scales they get one panel apiece.

matplotlib-only, no network, no LaTeX. Data is taken verbatim from
docs/benchmarks.md.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for future tweaks)

# Write next to this script (assets/benchmarks.png), regardless of CWD.
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.png")

# ---- Data (exact, from docs/benchmarks.md) --------------------------------
CONDITIONS = ["Base", "verl default", "LLMZero"]

# Panel A: GRPO / SSMR-Bench: validation accuracy, converted to percent.
GRPO_VALUES = [50.4, 73.6, 84.0]  # from 0.504 / 0.736 / 0.840

# Panel B: SFT / GovReport: held-out test scrolls_score (0-100 scale).
SFT_VALUES = [22.86, 31.74, 34.66]

# ---- Palette (muted, consistent across panels) ----------------------------
BASE_FILL = "#9AA0A6"   # muted gray
VERL_FILL = "#4878CF"   # muted blue
LLM_FILL = "#2E8B57"    # sea green (winner)
LLM_EDGE = "#B8860B"    # subtle gold edge to emphasize the winner

FILLS = [BASE_FILL, VERL_FILL, LLM_FILL]
EDGES = ["none", "none", LLM_EDGE]
EDGE_WIDTHS = [0.0, 0.0, 1.6]

INK = "#202124"          # primary text
MUTED_INK = "#5F6368"    # secondary text
GRID = "#DADCE0"         # recessive gridlines

# ---- Typography / global style --------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "text.color": INK,
        "axes.edgecolor": MUTED_INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": MUTED_INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

BAR_WIDTH = 0.62


def draw_panel(ax, values, title, ylabel, ylim, fmt, headroom_label):
    x = range(len(CONDITIONS))
    bars = ax.bar(
        x,
        values,
        width=BAR_WIDTH,
        color=FILLS,
        edgecolor=EDGES,
        linewidth=EDGE_WIDTHS,
        zorder=3,
    )

    # Value labels above each bar.
    span = ylim[1] - ylim[0]
    for rect, val, cond in zip(bars, values, CONDITIONS):
        is_winner = cond == "LLMZero"
        ax.annotate(
            fmt.format(val),
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11.5 if is_winner else 10.5,
            fontweight="bold" if is_winner else "normal",
            color=INK if is_winner else MUTED_INK,
            zorder=4,
        )

    # Axes cosmetics.
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(list(x))
    ax.set_xticklabels(CONDITIONS)

    # Emphasize the winning tick label.
    for lbl, cond in zip(ax.get_xticklabels(), CONDITIONS):
        if cond == "LLMZero":
            lbl.set_fontweight("bold")
            lbl.set_color(LLM_FILL)

    # Light horizontal-only gridlines, recessive.
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Remove top/right spines; soften remaining.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED_INK)
    ax.tick_params(axis="both", length=0)
    ax.margins(x=0.08)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    draw_panel(
        axA,
        GRPO_VALUES,
        "GRPO: SSMR-Bench",
        "Validation accuracy (%)",
        ylim=(0, 100),
        fmt="{:.1f}",
        headroom_label=4,
    )
    draw_panel(
        axB,
        SFT_VALUES,
        "SFT: GovReport",
        "Test scrolls_score",
        ylim=(0, 42),
        fmt="{:.2f}",
        headroom_label=4,
    )

    fig.suptitle(
        "LLMZero vs. base and verl-default (zero human intervention)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
