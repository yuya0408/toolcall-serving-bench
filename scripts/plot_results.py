"""Draw the README figures from the result values.

The values are copied from the run manifests and scripts/noninferiority.py
(README "詳しい値", ADR-0008 and docs/gpu-main-batching-2026-10-30.md),
because the raw runs are not in the repository. Edit them here if a value changes.

    pip install matplotlib
    python scripts/plot_results.py        # writes docs/figures/*.png

Figures (phase 1 without a suffix, phase 2 with _batching; the two are drawn
apart because the instance and the accuracy gate differ):
  selection*.png        the conclusion: gate (groups) -> max concurrency (bars)
                        -> instances for 16 concurrent (right column)
  latency*.png          P99 TTFT and TPOT side by side against concurrency, with
                        the per-cell verdict as the marker fill, as in Red Hat's
                        'Ollama vs. vLLM: A deep dive into performance benchmarking'
  throughput*.png       reference only, not used by any verdict
  noninferiority.png, gate_batching.png
                        confidence interval against the margin, as recommended by
                        the CONSORT extension for non-inferiority trials

One look across every figure: color is the engine, marker and line style the
quantization. The chosen config is in color, the others that passed the gate
in grey, and the excluded ones hatched (bars) or thin grey (lines).
"""

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Categorical slots in a fixed order (validated: adjacent CVD dE >= 9.1 on #fcfcfb).
ENGINE_COLOR = {"vLLM": "#2a78d6", "TRT-LLM": "#eb6834", "SGLang": "#1baf7a",
                "transformers": "#eda100", "llama.cpp": "#e87ba4"}
QUANT_LOOK = {"bf16": ("s", "-"), "AWQ": ("o", (0, (5, 2))), "Q4_K_M": ("v", (0, (5, 2))),
              "FP8": ("^", (0, (1.5, 1.5)))}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"
LIMIT = {"ttft": 2000, "tpot": 100}
MARGIN = -2.6
TARGET = 16   # the target concurrency the selection figures count instances for


def look(name):
    engine, quant = name.rsplit(" ", 1)
    marker, line = QUANT_LOOK[quant]
    return ENGINE_COLOR[engine], marker, line


class Phase:
    def __init__(self, label, chosen, selection, latency, gate, footnote, xmax):
        self.label = label          # "第1段階"
        self.chosen = chosen
        self.selection = selection  # [(config, max concurrency, passed the gate)]
        self.latency = latency      # {config: [(concurrency, TTFT p99, TPOT p99, req/s, verdict)]}
        self.gate = gate            # [(config, diff pt, lower bound pt, group)]
        self.footnote = footnote
        self.xmax = xmax

    def passed(self, name):
        return next(ok for n, _, ok in self.selection if n == name)


# Verdicts: "m" meets, "f" fails, "i" indeterminate (ADR-0003).
# Phase 1: main run 1 (Q4_K_M from main run 3); gate from ADR-0008 (one-sided 99.5%).
PHASE1 = Phase(
    "第1段階", "vLLM AWQ",
    [("vLLM AWQ", 4, True), ("vLLM bf16", 2, True), ("llama.cpp bf16", 1, True),
     ("transformers bf16", 1, True), ("llama.cpp Q4_K_M", 1, False)],
    {
        "vLLM AWQ": [(1, 783, 22.6, 0.557, "m"), (2, 1372, 53.3, 0.780, "m"), (4, 1466, 97.9, 1.019, "m"),
                     (8, 2619, 185.1, 1.180, "f"), (16, 3923, 364.1, 1.264, "f")],
        "vLLM bf16": [(1, 838, 61.5, 0.244, "m"), (2, 1344, 88.2, 0.403, "m"), (4, 1503, 130.4, 0.640, "f"),
                      (8, 2995, 207.3, 0.896, "f")],
        "llama.cpp bf16": [(1, 1333, 61.2, 0.224, "m"), (2, 2455, 131.5, 0.338, "f"),
                           (4, 2876, 217.8, 0.444, "f"), (8, 5597, 406.3, 0.534, "f"),
                           (16, 9357, 803.4, 0.589, "f")],
        "transformers bf16": [(1, 904, 69.8, 0.215, "m")],
        "llama.cpp Q4_K_M": [(1, 1187, 23.2, 0.459, "m"), (2, 2162, 83.1, 0.581, "f"),
                             (4, 2370, 157.6, 0.657, "f"), (8, 4780, 316.7, 0.688, "f"),
                             (16, 7440, 613.7, 0.751, "f"), (32, 19436, 1098.5, 0.760, "f")],
    },
    [("vLLM AWQ", -0.6, -2.48, None), ("llama.cpp Q4_K_M", -2.0, -4.05, None)],
    "第1段階のインスタンス。transformersは並行数1だけを測った。vLLM AWQの並行数4は、別のインスタンスでは満たさなかった（README「限界と範囲外」）。",
    32,
)

# Phase 2: one instance (docs/gpu-main-batching-2026-10-30.md); gate from ADR-0009 (one-sided 99.9%).
PHASE2 = Phase(
    "第2段階", "TRT-LLM FP8",
    [("TRT-LLM FP8", 4, True), ("vLLM bf16", 2, True), ("vLLM FP8", 2, True), ("TRT-LLM bf16", 1, True),
     ("vLLM AWQ", 2, False), ("SGLang AWQ", 2, False), ("SGLang FP8", 2, False), ("TRT-LLM AWQ", 2, False),
     ("SGLang bf16", 1, False)],
    {
        "vLLM bf16": [(1, 860, 61.7, 0.245, "m"), (2, 1374, 89.1, 0.402, "m"), (4, 1833, 131.1, 0.631, "f"),
                      (8, 2683, 214.3, 0.881, "f")],
        "vLLM AWQ": [(1, 821, 22.7, 0.543, "m"), (2, 1444, 54.5, 0.769, "m"), (4, 1853, 101.9, 0.986, "f"),
                     (8, 2770, 194.4, 1.147, "f"), (16, 4115, 382.3, 1.220, "f")],
        "vLLM FP8": [(1, 1062, 35.4, 0.362, "m"), (2, 1722, 72.7, 0.534, "m"), (4, 2508, 133.0, 0.728, "f"),
                     (8, 4145, 239.9, 0.868, "f"), (16, 5917, 499.5, 0.960, "f")],
        "SGLang bf16": [(1, 908, 61.5, 0.243, "m"), (2, 1503, 100.9, 0.385, "i"), (4, 2223, 152.0, 0.584, "f"),
                        (8, 2747, 245.5, 0.815, "f")],
        "SGLang AWQ": [(1, 899, 22.2, 0.540, "m"), (2, 1633, 60.6, 0.725, "m"), (4, 2272, 119.6, 0.900, "f"),
                       (8, 3515, 217.7, 1.025, "f"), (16, 4482, 433.3, 1.101, "f")],
        "SGLang FP8": [(1, 639, 40.3, 0.373, "m"), (2, 1130, 64.8, 0.610, "m"), (4, 1686, 107.9, 0.892, "f"),
                       (8, 2115, 175.9, 1.154, "f"), (16, 3089, 356.0, 1.355, "f")],
        "TRT-LLM bf16": [(1, 846, 62.4, 0.242, "m"), (2, 1412, 94.0, 0.393, "i"), (4, 1546, 137.6, 0.614, "f"),
                         (8, 2823, 221.4, 0.855, "f")],
        "TRT-LLM AWQ": [(1, 823, 25.4, 0.472, "m"), (2, 1417, 58.0, 0.698, "m"), (4, 1599, 101.4, 0.935, "f"),
                        (8, 2783, 184.3, 1.099, "f"), (16, 4185, 373.2, 1.211, "f")],
        "TRT-LLM FP8": [(1, 502, 35.5, 0.432, "m"), (2, 844, 54.3, 0.716, "m"), (4, 924, 81.8, 1.091, "m"),
                        (8, 1627, 130.8, 1.452, "f"), (16, 2491, 267.9, 1.732, "f")],
    },
    [("vLLM bf16", -0.10, -1.52, "bf16"), ("TRT-LLM bf16", 0.35, -1.10, "bf16"), ("SGLang bf16", -0.25, -2.83, "bf16"),
     ("vLLM FP8", -0.85, -2.08, "quant"), ("TRT-LLM FP8", -0.15, -1.43, "quant"), ("vLLM AWQ", -0.45, -2.62, "quant"),
     ("SGLang FP8", -0.75, -3.37, "quant"), ("SGLang AWQ", -0.95, -3.67, "quant"),
     ("TRT-LLM AWQ", -2.25, -5.11, "quant")],
    "第2段階のインスタンス（第1段階とは別）。9構成をすべて同じ1台で測った。",
    16,
)

WORKLOAD = "Qwen3-8B | NVIDIA L4 1台（g6.xlarge）| 日本語のツール呼び出し200件（入力 約2,300トークン）"


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


def title(fig, text, sub=None):
    """Title and subtitle at fixed distances from the top, whatever the figure height."""
    h = fig.get_figheight()
    fig.suptitle(text, x=0.02, y=1 - 0.12 / h, ha="left", va="top", fontsize=12.5, color=INK, fontweight="bold")
    if sub:
        fig.text(0.02, 1 - 0.5 / h, sub, ha="left", va="top", fontsize=9, color=INK_2)
    return 1 - (0.85 if sub else 0.55) / h     # top of the plotting area


def footer(fig, *lines):
    for i, line in enumerate(reversed(lines)):
        fig.text(0.02, 0.01 + i * 0.16 / fig.get_figheight(), line, fontsize=8, color=INK_2)
    return (0.05 + 0.16 * len(lines)) / fig.get_figheight()


def selection(phase, title_text, filename):
    """Gate as two groups, max concurrency as bars, instances for TARGET as the right column."""
    passed = [r for r in phase.selection if r[2]]
    excluded = [r for r in phase.selection if not r[2]]
    rows = len(passed) + len(excluded) + 2           # two group headers
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.42 * rows))
    y = rows - 1
    ticks, labels = [], []
    for header, group in (("精度の足切りを通った", passed), ("精度で除外（非劣化を示せなかった）", excluded)):
        ax.text(-0.02, y, f"{header}（{len(group)}構成）", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=9.5, color=INK, fontweight="bold")
        y -= 1
        for name, maxc, ok in group:
            chosen = name == phase.chosen
            if ok:
                ax.barh(y, maxc, height=0.62, color=look(name)[0] if chosen else MUTED, zorder=2)
                n = ceil(TARGET / maxc)
                ax.text(6.4, y, f"{n}台", va="center", ha="right", fontsize=10, color=INK,
                        fontweight="bold" if chosen else "normal")
            else:
                ax.barh(y, maxc, height=0.62, color=SURFACE, edgecolor="#8f8e88", hatch="///",
                        linewidth=1.0, zorder=2)
                ax.text(6.4, y, "—", va="center", ha="right", fontsize=10, color=INK_2)
            ax.text(maxc + 0.08, y, f"{maxc}並行", va="center", fontsize=10, color=INK if ok else INK_2,
                    fontweight="bold" if chosen else "normal")
            ticks.append(y)
            labels.append(name)
            y -= 1
        if group is passed:
            ax.axhline(y + 0.5, color=INK_2, linewidth=0.8, xmin=0, xmax=1)
    ax.text(6.4, rows - 1, f"目標{TARGET}並行の台数", va="center", ha="right", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(ticks, labels, color=INK)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_xlim(0, 6.6)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("1台で会話用の基準（TTFT 2秒・TPOT 100ms）を満たす最大の並行数", loc="left")
    top = title(fig, title_text, f"台数＝ceil({TARGET} ÷ 最大の並行数)。読む順：足切り（上下のグループ）→ 最大の並行数（棒）→ 台数（右）")
    bottom = footer(fig, phase.footnote)
    fig.tight_layout(rect=(0.1, bottom, 1, top))
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def _lines(ax, phase, col, with_verdict):
    """Configs that passed the gate in color (the chosen one thicker), the
    excluded ones thin grey. The marker fill is the verdict of that cell."""
    for name, rows in phase.latency.items():
        xs = [r[0] for r in rows]
        ys = [r[col] for r in rows]
        color, marker, line = look(name)
        if not phase.passed(name):
            ax.plot(xs, ys, color=MUTED, linewidth=1, linestyle=line, zorder=1, label=f"{name}（除外）")
            continue
        width = 3 if name == phase.chosen else 2
        ax.plot(xs, ys, color=color, linewidth=width, linestyle=line, zorder=2, label=name,
                marker=marker, markersize=7, markerfacecolor=color, markeredgecolor=SURFACE)
        if with_verdict:
            for x, yv, v in zip(xs, ys, [r[4] for r in rows]):
                if v == "m":
                    continue
                face, edge = (SURFACE, color) if v == "f" else (MUTED, INK)
                ax.plot([x], [yv], marker=marker, markersize=8, markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=1.6, linestyle="none", zorder=3)


def _concurrency_axis(ax, phase):
    xt = [c for c in (1, 2, 4, 8, 16, 32) if c <= phase.xmax]
    ax.set_xscale("log", base=2)
    ax.set_xticks(xt, [str(c) for c in xt])
    ax.set_xlim(0.8, phase.xmax * 1.25)
    ax.minorticks_off()
    ax.set_xlabel("並行数（同時に送るリクエストの数）")


def latency(phase, title_text, filename):
    """TTFT and TPOT side by side: the verdict needs both."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, key, col, ticks in ((axes[0], "ttft", 1, [500, 1000, 2000, 5000, 10000, 20000]),
                                (axes[1], "tpot", 2, [20, 50, 100, 200, 500, 1000])):
        _lines(ax, phase, col, with_verdict=True)
        limit = LIMIT[key]
        ax.axhline(limit, color=INK_2, linewidth=1, zorder=1)
        ax.text(phase.xmax * 1.2, limit * 0.93, f"会話用の基準 {limit:,}ms", ha="right", va="top", fontsize=9,
                color=INK_2)
        _concurrency_axis(ax, phase)
        ax.set_yscale("log")
        top = max(r[col] for rows in phase.latency.values() for r in rows)
        ax.set_yticks([t for t in ticks if t <= top * 1.3], [f"{t:,}" for t in ticks if t <= top * 1.3])
        ax.minorticks_off()
        name = "TTFT" if key == "ttft" else "TPOT"
        ax.set_title(f"P99 {name}（ms、対数）　下ほど良い", loc="left", fontsize=10.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    marks = [plt.Line2D([], [], marker="o", color=INK_2, markerfacecolor=INK_2, linestyle="none"),
             plt.Line2D([], [], marker="o", color=INK_2, markerfacecolor=SURFACE, markeredgewidth=1.6, linestyle="none"),
             plt.Line2D([], [], marker="o", color=INK, markerfacecolor=MUTED, markeredgewidth=1.6, linestyle="none")]
    fig.legend(handles + marks, labels + ["色の塗り：満たす", "白抜き：満たさない", "灰の塗り：判定不能"],
               loc="upper left", bbox_to_anchor=(0.785, 0.84), fontsize=9, handlelength=3,
               title="構成と、超過件数による判定", title_fontsize=9)
    top = title(fig, title_text, "判定は超過件数（1,000件中、超過が1%以下と言えるか）で行う。P99が基準線より下でも「判定不能」になり得る")
    bottom = footer(fig, WORKLOAD, phase.footnote)
    fig.tight_layout(rect=(0, bottom, 0.78, top))
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def throughput(phase, title_text, filename):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    _lines(ax, phase, 3, with_verdict=False)
    _concurrency_axis(ax, phase)
    ax.set_ylim(0, None)
    ax.set_ylabel("1台のスループット（件/秒）")
    ax.set_title("上ほど良い", loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, handlelength=3, title="構成", title_fontsize=9)
    top = title(fig, title_text)
    bottom = footer(fig, WORKLOAD, phase.footnote)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def interval(phase, title_text, filename, interval_label, groups=None, left=0.4):
    """Difference with the verdict's interval. The t interval is symmetric, so
    the two-sided interval's left end is the one-sided lower bound used."""
    rows = list(phase.gate)
    n = len(rows) + (len(groups) if groups else 0)
    fig, ax = plt.subplots(figsize=(10.5, 2.2 + 0.5 * n))
    lo = min(r[2] for r in rows) - 0.8
    ax.axvspan(lo - 10, MARGIN, color=GRID, alpha=0.6, zorder=0)
    y = n - 1
    for group, header in (groups or [(None, None)]):
        if header:
            ax.text(-0.01, y, header, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
                    color=INK, fontweight="bold")
            y -= 1
        for name, diff, low, g in rows:
            if g != group:
                continue
            color, marker, _ = look(name)
            ok = low > MARGIN
            high = 2 * diff - low
            ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="butt", zorder=2)
            for x in (low, high):
                ax.plot([x, x], [y - 0.12, y + 0.12], color=color, linewidth=2, zorder=2)
            ax.scatter(diff, y, s=55, marker=marker, color=color, zorder=3)
            ax.text(high + 0.12, y, f"差 {fmt(diff, 2)}", ha="left", va="center", fontsize=8.5, color=INK_2)
            ax.text(-0.01, y, f"{name}   下限 {fmt(low, 2)}   {'示せた' if ok else '示せなかった'}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5,
                    color=INK if ok else INK_2, fontweight="bold" if ok else "normal")
            y -= 1
    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)
    ax.axvline(MARGIN, color=INK, linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)
    ax.text(MARGIN - 0.08, n - 0.35, f"許す差 −Δ ＝ {fmt(MARGIN, 1)}pt", ha="right", va="top", fontsize=9, color=INK)
    ax.set_xlim(lo, 3.6)
    ax.set_ylim(-0.8, n - 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("selectionの差（pt）")
    ax.text(-0.1, -0.55, "← 劣る", ha="right", va="center", fontsize=9, color=INK_2)
    ax.text(0.1, -0.55, "優れる →", ha="left", va="center", fontsize=9, color=INK_2)
    top = title(fig, title_text, f"横線は判定の区間で、左端が下限（{interval_label}）。左端が網掛け（許す差より左）に入ると「示せなかった」")
    fig.tight_layout(rect=(left, 0, 1, top))  # left: room for the row labels
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    selection(PHASE1, "第1段階：精度を保った構成のうち、vLLM AWQが1台で最も多い並行数をさばく", "selection.png")
    latency(PHASE1, "第1段階：TTFT・TPOTとも基準を満たす並行数は、vLLM AWQの4が最大", "latency.png")
    throughput(PHASE1, "第1段階：1台のスループット（参考。判定には使わない）", "throughput.png")
    interval(PHASE1, "第1段階：4bitで自分のbf16に対する非劣化を示せたのはvLLM AWQだけ", "noninferiority.png",
             "片側99.5%の下限 ＝ 両側99%区間の左端。4bit − 自分のbf16", left=0.31)
    selection(PHASE2, "第2段階：足切りを通った4構成のうち、TRT-LLM FP8が1台で最も多い並行数をさばく",
              "selection_batching.png")
    latency(PHASE2, "第2段階：並行数4でTTFT・TPOTとも基準を満たしたのは、TRT-LLM FP8だけ", "latency_batching.png")
    throughput(PHASE2, "第2段階：1台のスループット（参考。判定には使わない）", "throughput_batching.png")
    interval(PHASE2, "第2段階：精度の足切りを通ったのは9構成中4つ", "gate_batching.png",
             "片側99.9%の下限 ＝ 両側99.8%区間の左端",
             groups=[("bf16", "bf16（基準：他の2エンジンのbf16の平均）"),
                     ("quant", "量子化（基準：通ったvLLM・TRT-LLM bf16の平均）")])


if __name__ == "__main__":
    main()
