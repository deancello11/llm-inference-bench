"""Turns the raw results into a summary table and all the charts.

    python -m analysis.make_charts --results results --out results/charts

Runs on a laptop from the saved result files (needs pandas and matplotlib only).
Energy is the GPU board's energy from the NVML counter. Add --energy idle_sub to subtract the
idle power first (energy - idle_power * time). Each chart label says which one it shows.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bench.io_utils import read_jsonl  # noqa: E402
from bench.stats import pareto_frontier  # noqa: E402

# One color for each serving stack.
STACK_COLOR = {"hf": "#2a78d6", "vllm": "#eb6834", "llamacpp": "#1baf7a"}
STACK_LABEL = {"hf": "Hugging Face", "vllm": "vLLM", "llamacpp": "llama.cpp"}
# The marker shape shows the precision, so the charts don't depend on color alone.
MARKER = {"fp16": "o", "gguf-f16": "o", "int8-bnb": "s", "gguf-q8_0": "s",
          "nf4-bnb": "^", "awq-int4": "^", "gptq-int4": "D", "gguf-q4_k_m": "^"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
OFFSETS = [(7, 6, "left"), (7, -13, "left"), (-7, 6, "right"), (-7, -13, "right")]  # label positions to rotate through
# Where to put each config's label on the Pareto chart: (x shift, y shift, alignment).
# Picked by hand so labels of nearby points (awq and gptq, q8 and vllm-fp16) don't overlap.
# A config that isn't listed here just rotates through OFFSETS.
LABEL_POS = {
    "hf-fp16": (7, 6, "left"), "vllm-fp16": (-7, 8, "right"), "llamacpp-f16": (7, -13, "left"),
    "llamacpp-q8": (-9, 6, "right"), "llamacpp-q4km": (7, 6, "left"),
    "vllm-awq": (7, 6, "left"), "vllm-gptq": (-7, -13, "right"),
    "hf-int8-bnb": (7, -13, "left"), "hf-nf4-bnb": (-7, 6, "right"),
}
BANDWIDTH_GBS = 1008.0  # RTX 4090 peak memory bandwidth


def style(ax, xlabel, ylabel, title):
    """Give every chart the same look: light grid, no top/right border, left-aligned title."""
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def new_fig(w=7.5, h=4.8, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), facecolor=SURFACE)
    return fig, axes


def stack_legend(ax, loc="best"):
    """A small legend that maps each color to a serving stack."""
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="-", label=STACK_LABEL[s])
               for s, c in STACK_COLOR.items()]
    ax.legend(handles=handles, frameon=False, labelcolor=INK2, loc=loc, fontsize=9)


# ------------------------------------------------------------------ loading data ---
def load(results: pathlib.Path, energy: str):
    """Read the result files and return (every run, one row per cell with the median, quality table)."""
    runs = pd.DataFrame(read_jsonl(results / "runs.jsonl"))
    loads = pd.DataFrame(read_jsonl(results / "loads.jsonl"))
    qual = pd.DataFrame(read_jsonl(results / "quality.jsonl"))
    if runs.empty:
        raise SystemExit(f"no runs in {results/'runs.jsonl'}")
    if energy == "idle_sub":
        runs["energy_per_token_mj"] = 1000 * runs["energy_idle_subtracted_j"] / runs["total_tokens"]
    runs["is_throttled"] = runs["thermal_throttled"].astype(bool)
    keys = ["config_id", "stack", "precision", "workload", "input_len", "concurrency"]
    num = ["ttft_mean_s", "ttft_p95_s", "decode_tps_per_request", "itl_mean_ms", "throughput_tps",
           "energy_per_token_mj", "mem_delta_peak_mib", "power_mean_w", "sm_clock_mean_mhz", "temp_max_c"]
    agg = runs.groupby(keys).agg({**{c: "median" for c in num}, "is_throttled": "sum", "repeat": "count"}) \
              .rename(columns={"repeat": "n_windows", "is_throttled": "n_throttled"}).reset_index()
    # how much the repeats of the same cell differ from each other
    spread = runs.groupby(keys)["decode_tps_per_request"].agg(lambda s: (s.max() - s.min()) / s.median()).rename("tps_spread")
    agg = agg.merge(spread.reset_index(), on=keys)

    if not loads.empty:
        l = loads.groupby("config_id").agg(weights_bytes=("weights_bytes", "first"),
                                           kv_cache_tokens=("kv_cache_tokens", "first") if "kv_cache_tokens" in loads else ("weights_bytes", "size"),
                                           idle_power_w=("idle_power_w", "median")).reset_index()
        agg = agg.merge(l, on="config_id", how="left")

    q = pd.DataFrame()
    if not qual.empty:
        q = qual.pivot_table(index="config_id", columns="task", values=["acc", "ppl"], aggfunc="last")
        cols = {}
        for task in ("arc", "mmlu", "hellaswag"):
            if ("acc", task) in q:
                cols[f"acc_{task}"] = q[("acc", task)]
        if ("ppl", "ppl") in q:
            cols["ppl"] = q[("ppl", "ppl")]
        q = pd.DataFrame(cols)
        acc_cols = [c for c in q.columns if c.startswith("acc_")]
        if acc_cols:
            q["acc_mean"] = q[acc_cols].mean(axis=1)
        q = q.reset_index()
    return runs, agg, q


def pick(agg, wl, conc):
    """The rows for one workload and one concurrency level."""
    return agg[(agg.workload == wl) & (agg.concurrency == conc)].copy()


# ------------------------------------------------------------------ charts ---------
def chart_pareto(agg, q, out, energy_label, wl="short", concs=(1, 32)):
    """Accuracy against energy per token, one dot per config. The dashed line joins the best trade-offs."""
    if q.empty or "acc_mean" not in q:
        print("skip pareto: no quality results yet")
        return
    fig, axes = new_fig(6.4, 4.8, ncols=len(concs))
    axes = [axes] if len(concs) == 1 else list(axes)
    for ax, conc in zip(axes, concs):
        d = pick(agg, wl, conc).merge(q, on="config_id")
        if d.empty:
            continue
        pts = list(zip(d.energy_per_token_mj, d.acc_mean * 100))
        front = pareto_frontier(pts)
        fx = [pts[i][0] for i in front]
        fy = [pts[i][1] for i in front]
        ax.step(fx, fy, where="post", color=INK2, linewidth=1.2, linestyle="--", zorder=1)
        for i, r in enumerate(d.itertuples()):
            ax.scatter(r.energy_per_token_mj, r.acc_mean * 100, s=70, color=STACK_COLOR[r.stack],
                       marker=MARKER.get(r.precision, "o"), edgecolor=SURFACE, linewidth=1.5, zorder=3)
            dx, dy, ha = LABEL_POS.get(r.config_id, OFFSETS[i % len(OFFSETS)])
            ax.annotate(r.config_id, (r.energy_per_token_mj, r.acc_mean * 100), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=8, color=INK2)
        style(ax, f"mJ per generated token, {energy_label} (lower is better)",
              "Mean accuracy % (ARC-C, MMLU, HellaSwag)",
              f"Concurrency {conc}, {wl} workload")
        stack_legend(ax, "lower right")
    fig.suptitle("Quality vs energy: dashed line = Pareto frontier", color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=170)
    plt.close(fig)


def plain_log_yticks(ax):
    """Label a log-scale y-axis at 1, 2, 5, 10, 20, 50... as plain numbers (matplotlib shows only 10, 100, 1000)."""
    lo, hi = ax.get_ylim()
    ticks = [m * 10 ** k for k in range(-1, 8) for m in (1, 2, 5) if lo <= m * 10 ** k <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def label_line_ends(fig, ax, items, min_gap_pt=11.0):
    """Write each line's name at its right end, nudging labels apart so they don't overlap."""
    fig.canvas.draw()  # draw once so the axis limits are final before we convert positions
    # (name, x, y, vertical pixel position), sorted from bottom to top
    px = [(name, x, y, ax.transData.transform((x, y))[1]) for name, x, y in items]
    px.sort(key=lambda t: t[3])
    gap = min_gap_pt * fig.dpi / 72.0
    pos = []
    for _, _, _, py in px:
        pos.append(py if not pos else max(py, pos[-1] + gap))
    shift = (sum(p - t[3] for p, t in zip(pos, px)) / len(px))  # center the group so it isn't pushed upward
    for (name, x, y, py), p in zip(px, pos):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, (p - shift - py) * 72.0 / fig.dpi),
                    fontsize=8, color=INK2, va="center")


def _lines(agg, out, name, ycol, ylabel, title, wl="short", logy=False, x="concurrency"):
    """One line per config showing how a metric changes as concurrency goes up."""
    fig, ax = new_fig()
    d = agg[agg.workload == wl]
    ends = []
    for cid, g in d.groupby("config_id"):
        g = g.sort_values(x)
        r = g.iloc[0]
        ax.plot(g[x], g[ycol], color=STACK_COLOR[r["stack"]], marker=MARKER.get(r.precision, "o"),
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ends.append((cid, g[x].iloc[-1], g[ycol].iloc[-1]))
    if x == "concurrency":
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.concurrency.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.margins(x=0.25)
    style(ax, "Concurrent requests" if x == "concurrency" else x, ylabel, title)
    stack_legend(ax, "upper left")
    fig.tight_layout()
    if logy:
        fig.canvas.draw()
        plain_log_yticks(ax)
    label_line_ends(fig, ax, ends)
    fig.savefig(out / f"{name}.png", dpi=170)
    plt.close(fig)


def chart_ttft_vs_input(agg, out):
    """Time to first token for short and long prompts. This is the cost of reading the prompt."""
    fig, ax = new_fig()
    d = agg[agg.concurrency == 1]
    ends = []
    for cid, g in d.groupby("config_id"):
        g = g.sort_values("input_len")
        r = g.iloc[0]
        ax.plot(g.input_len, g.ttft_mean_s * 1000, color=STACK_COLOR[r["stack"]], marker=MARKER.get(r.precision, "o"),
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ends.append((cid, g.input_len.iloc[-1], g.ttft_mean_s.iloc[-1] * 1000))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(d.input_len.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.margins(x=0.25)
    style(ax, "Input length (tokens)", "Time to first token, ms (log)", "Prefill cost: TTFT vs input length (concurrency 1)")
    stack_legend(ax, "upper left")
    fig.tight_layout()
    fig.canvas.draw()
    plain_log_yticks(ax)
    label_line_ends(fig, ax, ends)
    fig.savefig(out / "ttft_vs_input.png", dpi=170)
    plt.close(fig)


def chart_memory(agg, out):
    """Bar chart of the model's weight size for each config."""
    if "weights_bytes" not in agg:
        return
    d = agg[(agg.workload == "short") & (agg.concurrency == 1)].sort_values("weights_bytes")
    fig, ax = new_fig()
    y = range(len(d))
    ax.barh(list(y), d.weights_bytes / 1e9, color=[STACK_COLOR[s] for s in d["stack"]], height=0.55)
    for i, r in zip(y, d.itertuples()):
        ax.annotate(f"{r.weights_bytes/1e9:.2f} GB", (r.weights_bytes / 1e9, i), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.config_id, color=INK2)
    ax.margins(x=0.15)
    style(ax, "Model weight size on GPU, GB (what every decoded token must read)", "",
          "Memory footprint: weights only (vLLM total VRAM is a preallocation setting, see README)")
    stack_legend(ax, "lower right")
    fig.tight_layout()
    fig.savefig(out / "memory_weights.png", dpi=170)
    plt.close(fig)


def chart_roofline(agg, out):
    """Measured speed against the speed limit set by memory bandwidth.

    The limit is the GPU's bandwidth divided by the model size, because for every token
    the GPU has to read the whole model from memory once.
    """
    if "weights_bytes" not in agg:
        return
    d = agg[(agg.workload == "short") & (agg.concurrency == 1)].sort_values("weights_bytes", ascending=False)
    d = d.assign(ceiling=BANDWIDTH_GBS * 1e9 / d.weights_bytes)
    fig, ax = new_fig()
    y = list(range(len(d)))
    ax.barh(y, d.decode_tps_per_request, color=[STACK_COLOR[s] for s in d["stack"]], height=0.55)
    ax.scatter(d.ceiling, y, marker="|", s=420, color=INK, zorder=3, linewidths=2)
    for i, r in zip(y, d.itertuples()):
        ax.annotate(f"{r.decode_tps_per_request:.0f} tok/s = {100*r.decode_tps_per_request/r.ceiling:.0f}% of ceiling",
                    (r.ceiling, i), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.config_id, color=INK2)
    ax.margins(x=0.45)
    style(ax, f"Decode tokens/s at concurrency 1 (| = bandwidth ceiling, {BANDWIDTH_GBS:.0f} GB/s / weight bytes)", "",
          "Roofline: batch-1 decode vs memory-bandwidth ceiling")
    fig.tight_layout()
    fig.savefig(out / "roofline.png", dpi=170)
    plt.close(fig)


def write_summary(agg, q, out):
    """Save every cell as summary.csv, plus a small markdown table for the README."""
    d = agg.copy()
    if not q.empty:
        d = d.merge(q, on="config_id", how="left")
    d.to_csv(out / "summary.csv", index=False)
    s = d[(d.workload == "short") & (d.concurrency == 1)].copy()
    cols = [("config_id", "config", "{}"), ("decode_tps_per_request", "decode tok/s", "{:.0f}"),
            ("ttft_mean_s", "TTFT ms", None), ("energy_per_token_mj", "mJ/token", "{:.0f}"),
            ("weights_bytes", "weights GB", None), ("acc_mean", "acc %", None), ("ppl", "wt2 ppl", "{:.2f}")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in s.itertuples():
        cells = []
        for key, _, fmt in cols:
            v = getattr(r, key, float("nan"))
            if pd.isna(v):
                cells.append("-")
            elif key == "ttft_mean_s":
                cells.append(f"{v*1000:.0f}")
            elif key == "weights_bytes":
                cells.append(f"{v/1e9:.2f}")
            elif key == "acc_mean":
                cells.append(f"{v*100:.1f}")
            else:
                cells.append(fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    (out / "summary_short_c1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/charts")
    ap.add_argument("--energy", choices=["raw", "idle_sub"], default="raw")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, agg, q = load(pathlib.Path(args.results), args.energy)
    label = "raw GPU energy" if args.energy == "raw" else "idle-subtracted GPU energy"

    n_thr = int(runs.is_throttled.sum())
    if n_thr:
        print(f"WARNING: {n_thr}/{len(runs)} windows hit thermal/HW throttling; inspect before trusting them.")
    write_summary(agg, q, out)
    chart_pareto(agg, q, out, label)
    _lines(agg, out, "throughput_vs_concurrency", "throughput_tps", "Aggregate throughput, tokens/s",
           "Throughput under load (short workload)")
    _lines(agg, out, "energy_vs_concurrency", "energy_per_token_mj", f"Energy per token, mJ ({label})",
           "Batching amortizes weight reads: energy/token vs concurrency", logy=True)
    chart_ttft_vs_input(agg, out)
    chart_memory(agg, out)
    chart_roofline(agg, out)
    print(f"wrote charts + summary to {out}")


if __name__ == "__main__":
    main()
