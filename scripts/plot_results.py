"""Draw the README figures from the result values.

The values are copied from the run manifests (README "詳しい値" and ADR-0008),
because the raw runs are not in the repository. Edit them here if a value changes.

    python scripts/plot_results.py        # writes docs/figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Categorical slots in a fixed order (validated: adjacent CVD dE >= 9.1 on #fcfcfb).
# A config keeps its color in every figure.
COLOR = {
    "vLLM AWQ": "#2a78d6",
    "vLLM bf16": "#eb6834",
    "llama.cpp bf16": "#1baf7a",
    "transformers bf16": "#eda100",
    "llama.cpp Q4_K_M": "#e87ba4",
}
MARKER = {
    "vLLM AWQ": "o",
    "vLLM bf16": "s",
    "llama.cpp bf16": "D",
    "transformers bf16": "^",
    "llama.cpp Q4_K_M": "v",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# (concurrency, TTFT p99 ms, TPOT p99 ms, meets the conversational limits)
LATENCY = {
    "vLLM AWQ": [(1, 783, 22.6, True), (2, 1372, 53.3, True), (4, 1466, 97.9, True),
                 (8, 2619, 185.1, False), (16, 3923, 364.1, False)],
    "vLLM bf16": [(1, 838, 61.5, True), (2, 1344, 88.2, True), (4, 1503, 130.4, False),
                  (8, 2995, 207.3, False)],
    "llama.cpp bf16": [(1, 1333, 61.2, True), (2, 2455, 131.5, False), (4, 2876, 217.8, False),
                       (8, 5597, 406.3, False), (16, 9357, 803.4, False)],
    "transformers bf16": [(1, 904, 69.8, True)],
    "llama.cpp Q4_K_M": [(1, 1187, 23.2, True), (2, 2162, 83.1, False), (4, 2370, 157.6, False),
                         (8, 4780, 316.7, False), (16, 7440, 613.7, False),
                         (32, 19436, 1098.5, False)],
}
LIMIT = {"ttft": 2000, "tpot": 100}
TICKS = {"ttft": [500, 1000, 2000, 5000, 10000, 20000], "tpot": [20, 50, 100, 200, 500, 1000]}

# (selection difference 4bit - bf16 in pt, one-sided 99.5% lower bound in pt, verdict)
NONINFERIORITY = [
    ("vLLM AWQ", -0.6, -2.48, "非劣化を示せた"),
    ("llama.cpp Q4_K_M", -2.0, -4.05, "示せなかった"),
]
MARGIN = -2.6

# (selection mean, 95% CI low, high) at concurrency 1
ACCURACY = [
    ("vLLM bf16", 0.937, 0.912, 0.961),
    ("llama.cpp bf16", 0.939, 0.914, 0.963),
    ("transformers bf16", 0.935, 0.911, 0.959),
    ("vLLM AWQ", 0.930, 0.902, 0.957),
]


def fmt(x, digits):
    """Signed number with a real minus sign, to match the axis labels."""
    return f"{x:+.{digits}f}".replace("-", "−")


def style():
    plt.rcParams.update({
        "font.family": ["Yu Gothic", "Meiryo", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
    })


def latency():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [(axes[0], 1, "ttft", "TTFT p99（ms）"), (axes[1], 2, "tpot", "TPOT p99（ms）")]
    for ax, col, key, label in panels:
        for name, rows in LATENCY.items():
            xs = [r[0] for r in rows]
            ys = [r[col] for r in rows]
            c = COLOR[name]
            ax.plot(xs, ys, color=c, linewidth=2, zorder=2)
            for x, y, ok in zip(xs, ys, [r[3] for r in rows]):
                ax.scatter(x, y, s=48, marker=MARKER[name], zorder=3, linewidths=1.6,
                           facecolors=c if ok else SURFACE, edgecolors=c)
        ax.axhline(LIMIT[key], color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(36, LIMIT[key] * 0.88, f"会話用の基準 {LIMIT[key]:,}ms", color=INK, fontsize=9,
                ha="right", va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = TICKS[key]
        ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
        ax.minorticks_off()
        ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_xlim(0.8, 40)
        ax.set_xlabel("並行数")
        ax.set_title(label, loc="left", fontsize=11)
    handles = [plt.Line2D([], [], color=COLOR[n], marker=MARKER[n], linewidth=2, markersize=7, label=n)
               for n in LATENCY]
    handles.append(plt.Line2D([], [], color=INK_2, marker="o", linestyle="", markersize=7,
                              markerfacecolor=SURFACE, label="白抜き＝基準を満たさない"))
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "latency.png", dpi=150)
    plt.close(fig)


def noninferiority():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ys = list(range(len(NONINFERIORITY)))[::-1]
    right = 1.5
    for y, (name, diff, low, verdict) in zip(ys, NONINFERIORITY):
        c = COLOR[name]
        ax.annotate("", xy=(right, y), xytext=(low, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, linewidth=2, shrinkA=0, shrinkB=0))
        ax.plot([low, low], [y - 0.12, y + 0.12], color=c, linewidth=2)
        ax.scatter(diff, y, s=60, marker=MARKER[name], color=c, zorder=3)
        ax.text(low + 0.06, y + 0.2, f"下限 {fmt(low, 2)}pt", ha="left", fontsize=9, color=INK_2)
        ax.text(diff, y - 0.32, f"差 {fmt(diff, 1)}pt", ha="center", fontsize=9, color=INK_2)
        ax.text(right + 0.1, y, verdict, va="center", fontsize=10, color=INK)
    ax.axvline(MARGIN, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(MARGIN - 0.08, len(ys) - 0.45, f"許す差 {fmt(MARGIN, 1)}pt", ha="right", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(ys, [n for n, *_ in NONINFERIORITY])
    ax.set_ylim(-0.6, len(ys) - 0.3)
    ax.set_xlim(-5, right)
    ax.set_xlabel("selectionの差（4bit − bf16、pt）。線は片側99.5%区間（下限から右）")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "noninferiority.png", dpi=150)
    plt.close(fig)


def accuracy():
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ys = list(range(len(ACCURACY)))[::-1]
    for y, (name, mean, low, high) in zip(ys, ACCURACY):
        c = COLOR[name]
        ax.plot([low, high], [y, y], color=c, linewidth=2)
        ax.scatter(mean, y, s=60, marker=MARKER[name], color=c, zorder=3)
        ax.text(high + 0.002, y, f"{mean:.3f} [{low:.3f}, {high:.3f}]", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(ys, [n for n, *_ in ACCURACY])
    ax.set_ylim(-0.6, len(ys) - 0.4)
    ax.set_xlim(0.89, 0.995)
    ax.set_xlabel("selection（並行数1、200件×5試行）。線はケース単位のブートストラップ95% CI")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    latency()
    noninferiority()
    accuracy()


if __name__ == "__main__":
    main()
