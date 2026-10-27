"""Draw the README figures from the result values.

The values are copied from the run manifests and scripts/noninferiority.py
(README "詳しい値" and ADR-0008), because the raw runs are not in the
repository. Edit them here if a value changes.

    pip install matplotlib
    python scripts/plot_results.py        # writes docs/figures/*.png

Figures and the outside figure each one follows:
  selection.png       the conclusion, as a single bar per config
  throughput.png, ttft.png, tpot.png
                      one metric against concurrency per figure, as in Red Hat's
                      'Ollama vs. vLLM: A deep dive into performance benchmarking'
  noninferiority.png  confidence interval against the margin, as recommended by
                      the CONSORT extension for non-inferiority trials
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Categorical slots in a fixed order (validated: adjacent CVD dE >= 9.1 on #fcfcfb).
# A config keeps its color and marker in every figure except selection.png,
# which highlights the chosen config and greys the rest.
CONFIGS = ["vLLM AWQ", "vLLM bf16", "llama.cpp bf16", "transformers bf16", "llama.cpp Q4_K_M"]
COLOR = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(CONFIGS, ["o", "s", "D", "^", "v"]))
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"

# Max concurrency per GPU that meets the conversational limits (main run 1;
# Q4_K_M from main run 3), and the accuracy gate.
SELECTION = [
    ("vLLM AWQ", 4, "精度：bf16と比べて非劣化を示せた", True),
    ("vLLM bf16", 2, "精度：基準（bf16）", True),
    ("llama.cpp bf16", 1, "精度：基準（bf16）", True),
    ("transformers bf16", 1, "精度：基準（bf16）", True),
    ("llama.cpp Q4_K_M", 1, "精度：非劣化を示せず → 除外", False),
]

# (concurrency, TTFT p99 ms, TPOT p99 ms, throughput req/s, meets the conversational limits)
# Throughput is the mean of the measured windows (measured_stability).
LATENCY = {
    "vLLM AWQ": [(1, 783, 22.6, 0.557, True), (2, 1372, 53.3, 0.780, True), (4, 1466, 97.9, 1.019, True),
                 (8, 2619, 185.1, 1.180, False), (16, 3923, 364.1, 1.264, False)],
    "vLLM bf16": [(1, 838, 61.5, 0.244, True), (2, 1344, 88.2, 0.403, True), (4, 1503, 130.4, 0.640, False),
                  (8, 2995, 207.3, 0.896, False)],
    "llama.cpp bf16": [(1, 1333, 61.2, 0.224, True), (2, 2455, 131.5, 0.338, False),
                       (4, 2876, 217.8, 0.444, False), (8, 5597, 406.3, 0.534, False),
                       (16, 9357, 803.4, 0.589, False)],
    "transformers bf16": [(1, 904, 69.8, 0.215, True)],
    "llama.cpp Q4_K_M": [(1, 1187, 23.2, 0.459, True), (2, 2162, 83.1, 0.581, False),
                         (4, 2370, 157.6, 0.657, False), (8, 4780, 316.7, 0.688, False),
                         (16, 7440, 613.7, 0.751, False), (32, 19436, 1098.5, 0.760, False)],
}
LIMIT = {"ttft": 2000, "tpot": 100}

# (selection difference 4bit - bf16 in pt, two-sided 99% t interval in pt, verdict).
# The lower end equals the one-sided 99.5% bound the verdict uses.
NONINFERIORITY = [
    ("vLLM AWQ", -0.6, (-2.48, 1.28), "非劣化を示せた"),
    ("llama.cpp Q4_K_M", -2.0, (-4.05, 0.05), "示せなかった"),
]
MARGIN = -2.6


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


def selection():
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ys = list(range(len(SELECTION)))[::-1]
    for y, (name, maxc, gate, ok) in zip(ys, SELECTION):
        chosen = name == "vLLM AWQ"
        if ok:
            ax.barh(y, maxc, height=0.6, color=COLOR[name] if chosen else MUTED, zorder=2)
        else:
            ax.barh(y, maxc, height=0.6, color=SURFACE, edgecolor="#8f8e88", hatch="///",
                    linewidth=1.2, zorder=2)
        ax.text(maxc + 0.08, y, f"{maxc}並行", va="center", fontsize=10, color=INK,
                fontweight="bold" if chosen else "normal")
        ax.text(5.6, y, gate, va="center", fontsize=9, color=COLOR[name] if chosen else INK_2)
    ax.set_yticks(ys, [r[0] for r in SELECTION], color=INK)
    ax.set_xlim(0, 9.2)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("L4 1台で会話用の基準（TTFT 2秒・TPOT 100ms）を満たす最大の並行数", loc="left")
    fig.suptitle("vLLM AWQは精度を保ったまま、1台でbf16の2倍の並行数をさばける", x=0.02, ha="left",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.875, "必要な台数＝目標の並行数÷この値（例：16並行なら AWQ 4台、vLLM bf16 8台、ほか16台）",
             fontsize=9, color=INK_2)
    fig.text(0.02, 0.01, "AWQの4並行は本番1回目のインスタンスでの値。別のインスタンスでは2並行だった（README「限界と範囲外」）。",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    fig.savefig(OUT / "selection.png", dpi=150)
    plt.close(fig)


def per_concurrency(key, col, title, ylabel, better, filename, limit=None, ticks=None):
    """One metric against concurrency, one line per config (as in Red Hat's
    'Ollama vs. vLLM' benchmarking article: one figure per metric)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, rows in LATENCY.items():
        xs = [r[0] for r in rows]
        ys = [r[col] for r in rows]
        ax.plot(xs, ys, color=COLOR[name], marker=MARKER[name], markersize=7, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1, label=name, zorder=2)
    if limit is not None:
        ax.axhline(limit, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(32, limit * 1.06, f"会話用の基準 {limit:,}ms", ha="right", va="bottom", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.8, 40)
    if ticks:
        ax.set_yscale("log")
        ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("並行数（同時に送るリクエストの数）")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.97, better, transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.legend(loc="upper left", fontsize=9, title="構成", title_fontsize=9)
    fig.text(0.5, 0.01, "Qwen3-8B | NVIDIA L4 1台（g6.xlarge）| 日本語のツール呼び出し200件（入力 約2,300トークン）",
             ha="center", fontsize=8.5, color=INK_2, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def latency():
    per_concurrency("rps", 3, "スループット と 並行数", "1台のスループット（件/秒）", "上ほど良い",
                    "throughput.png")
    per_concurrency("ttft", 1, "P99 TTFT と 並行数", "P99 TTFT（ms、対数）", "下ほど良い",
                    "ttft.png", limit=LIMIT["ttft"], ticks=[500, 1000, 2000, 5000, 10000, 20000])
    per_concurrency("tpot", 2, "P99 TPOT と 並行数", "P99 TPOT（ms、対数）", "下ほど良い",
                    "tpot.png", limit=LIMIT["tpot"], ticks=[20, 50, 100, 200, 500, 1000])


def noninferiority():
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    ys = list(range(len(NONINFERIORITY)))[::-1]
    for y, (name, diff, (low, high), verdict) in zip(ys, NONINFERIORITY):
        c = COLOR[name]
        ax.plot([low, high], [y, y], color=c, linewidth=2, solid_capstyle="butt")
        for x in (low, high):
            ax.plot([x, x], [y - 0.1, y + 0.1], color=c, linewidth=2)
        ax.scatter(diff, y, s=60, marker=MARKER[name], color=c, zorder=3)
        ax.text(low + 0.06, y - 0.3, fmt(low, 2), ha="left", fontsize=8.5, color=INK_2)
        ax.text(diff, y + 0.2, f"差 {fmt(diff, 1)}", ha="center", fontsize=8.5, color=INK_2)
        ax.text(-6.3, y, f"{name}\n{verdict}", va="center", ha="left", fontsize=9.5, color=INK)
    ax.axvline(0, color=INK, linewidth=1.4)
    ax.axvline(MARGIN, color=INK, linewidth=1.2, linestyle=(0, (2, 2)))
    ax.text(MARGIN - 0.08, len(ys) - 0.35, f"−Δ（許す差）{fmt(MARGIN, 1)}pt", ha="right", fontsize=9, color=INK)
    ax.set_xlim(-6.4, 2.5)
    ax.set_ylim(-0.7, len(ys) - 0.2)
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("selectionの差（4bit − bf16、pt）。横線は両側99%区間（左端＝判定に使う片側99.5%の下限）")
    ax.text(-0.15, -0.62, "← 4bitが劣る", ha="right", fontsize=9, color=INK_2)
    ax.text(0.15, -0.62, "4bitが優れる →", ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "noninferiority.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    selection()
    latency()
    noninferiority()


if __name__ == "__main__":
    main()
