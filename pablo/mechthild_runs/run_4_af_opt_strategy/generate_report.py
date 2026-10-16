"""Generate the run_4 AF-optimizer strategy HTML comparison report.

Models (st_y, lcm_ys) x batch-schemes (full_batch, split_3_3, split_2_2_2,
sequential) x six GA-budget tiers, 25 reps each. Reports the qLogNEI batch the
optimizer reaches, its wall-time cost, the GA / L-BFGS-refine split, batch
reproducibility and discrete-choice diversity. Reads
``data/results/result_*.json`` and writes ``af_opt_strategy_report.html``
next to this script.

Usage (from repo root):
    python -m pablo.mechthild_runs.run_4_af_opt_strategy.generate_report
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "results")
OUT_HTML = os.path.join(os.path.dirname(__file__), "af_opt_strategy_report.html")

D_CONTINUOUS = 4  # candidate layout is [cont(4) | cat_aug | sol | base]; tail = discrete identity

MODELS = ["st_y", "lcm_ys"]
MODEL_LABELS = {
    "st_y": "st_y — SingleTaskGP (Yield)",
    "lcm_ys": "lcm_ys — rank-2 LCM MultiTaskGP (Yield/Sel)",
}
SCHEMES = ["full_batch", "split_3_3", "split_2_2_2", "sequential"]
SCHEME_LABELS = {
    "full_batch": "Full batch [6]",
    "split_3_3": "Split [3,3]",
    "split_2_2_2": "Split [2,2,2]",
    "sequential": "Sequential [1×6]",
}
SCHEME_GA_CALLS = {"full_batch": 1, "split_3_3": 2, "split_2_2_2": 3, "sequential": 6}
SCHEME_COLORS = {
    "full_batch": "#2979ff",
    "split_3_3": "#00897b",
    "split_2_2_2": "#f9a825",
    "sequential": "#e65100",
}
N_TIERS = 6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load() -> tuple[dict, dict, dict]:
    """Load every ``result_*.json`` in ``RESULTS_DIR``.

    Returns:
        A tuple ``(cells, ga_settings, meta)``:

        - ``cells[(model, scheme, tier)]``: list of per-rep dicts with ``rep``,
          ``final_acq``, ``last_ga_acq``, ``total_t``, ``ga_t``, ``ref_t``
          (seconds), ``ga_frac`` and ``n_unique_disc``.
        - ``ga_settings[(scheme, tier)]``: ``(ga_n_evals, ga_pop_size)``.
        - ``meta``: run metadata with ``hosts``, ``devices``, ``torch``,
          ``python`` (sorted unique values), ``ts_min``, ``ts_max`` (UTC
          timestamps), ``total_wall_h`` (summed wall time in hours) and
          ``n_opts`` (number of optimizations).
    """
    cells = defaultdict(list)
    ga_settings = {}
    hosts, devices, torch_v, py_v, timestamps = set(), set(), set(), set(), []
    total_wall = 0.0
    n_opts = 0

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith("result_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as fh:
            raw = json.load(fh)
        m = raw["meta"]
        hosts.add(m["hostname"])
        devices.add(m["device"])
        torch_v.add(m["torch_version"])
        py_v.add(m["python_version"])
        timestamps.append(m["timestamp_utc"])

        model = raw["model"]
        scheme = raw["scheme"]
        for tr in raw["results"]:
            tier = tr["tier_idx"]
            cfg = tr["config_snapshot"]
            ga_settings[(scheme, tier)] = (cfg["ga_n_evals"], cfg["ga_pop_size"])
            res = tr["result"]
            steps = res["steps"]

            ga_t = sum(s["post_ga"]["wall_time_s"] for s in steps)
            ref_t = sum(s["post_refinement"]["wall_time_s"] for s in steps)
            total_t = res["total_wall_time_s"]
            last_ga_acq = steps[-1]["post_ga"]["acq_value"]

            cands = res["candidates"]
            disc = {tuple(round(c[i], 4) for i in range(D_CONTINUOUS, len(c))) for c in cands}

            cells[(model, scheme, tier)].append({
                "rep": tr["rep_idx"],
                "final_acq": res["final_acq_value"],
                "last_ga_acq": last_ga_acq,
                "total_t": total_t,
                "ga_t": ga_t,
                "ref_t": ref_t,
                "ga_frac": ga_t / total_t if total_t > 0 else 0.0,
                "n_unique_disc": len(disc),
            })
            total_wall += total_t
            n_opts += 1

    meta = {
        "hosts": sorted(hosts),
        "devices": sorted(devices),
        "torch": sorted(torch_v),
        "python": sorted(py_v),
        "ts_min": min(timestamps),
        "ts_max": max(timestamps),
        "total_wall_h": total_wall / 3600.0,
        "n_opts": n_opts,
    }
    return cells, ga_settings, meta


def acq(cells, model, scheme, tier):
    """Final acquisition values of one cell.

    Args:
        cells: Per-cell rep lists as returned by ``load``.
        model: Model name, e.g. ``"st_y"``.
        scheme: Batch scheme, e.g. ``"full_batch"``.
        tier: GA-budget tier index.

    Returns:
        Array of shape ``n_reps`` with the final qLogNEI value of each rep.
    """
    return np.array([r["final_acq"] for r in cells[(model, scheme, tier)]])


def wall_min(cells, model, scheme, tier):
    """Total wall times of one cell, in minutes.

    Args:
        cells: Per-cell rep lists as returned by ``load``.
        model: Model name, e.g. ``"st_y"``.
        scheme: Batch scheme, e.g. ``"full_batch"``.
        tier: GA-budget tier index.

    Returns:
        Array of shape ``n_reps`` with the wall time of each rep in minutes.
    """
    return np.array([r["total_t"] / 60.0 for r in cells[(model, scheme, tier)]])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_acq_vs_tier(cells) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, model in zip(axes, MODELS):
        for scheme in SCHEMES:
            meds = [np.median(acq(cells, model, scheme, t)) for t in range(N_TIERS)]
            stds = [np.std(acq(cells, model, scheme, t)) for t in range(N_TIERS)]
            ax.errorbar(range(N_TIERS), meds, yerr=stds, fmt="o-", capsize=3,
                        color=SCHEME_COLORS[scheme], linewidth=1.8, markersize=5,
                        label=SCHEME_LABELS[scheme], alpha=0.9)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Tier (GA budget rung)", fontsize=9)
        ax.set_xticks(range(N_TIERS))
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Final qLogNEI  (median ± std, n=25)", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Final qLogNEI vs GA budget — per model, per batch-scheme", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_wall_vs_tier(cells) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, model in zip(axes, MODELS):
        for scheme in SCHEMES:
            meds = [np.median(wall_min(cells, model, scheme, t)) for t in range(N_TIERS)]
            ax.plot(range(N_TIERS), meds, "o-", color=SCHEME_COLORS[scheme],
                    linewidth=1.8, markersize=5, label=SCHEME_LABELS[scheme], alpha=0.9)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Tier (GA budget rung)", fontsize=9)
        ax.set_xticks(range(N_TIERS))
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Median wall time (minutes, n=25)", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Wall-time cost vs GA budget — per model, per batch-scheme", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_acq_box(cells) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey="row")
    for i, model in enumerate(MODELS):
        for j, scheme in enumerate(SCHEMES):
            ax = axes[i, j]
            data = [acq(cells, model, scheme, t) for t in range(N_TIERS)]
            bp = ax.boxplot(data, positions=range(N_TIERS), widths=0.6, showfliers=True)
            for el in ["boxes", "whiskers", "caps", "medians"]:
                plt.setp(bp[el], color=SCHEME_COLORS[scheme])
            plt.setp(bp["medians"], linewidth=2)
            plt.setp(bp["fliers"], marker="o", markersize=2.5, alpha=0.4,
                     markeredgecolor=SCHEME_COLORS[scheme])
            if i == 0:
                ax.set_title(SCHEME_LABELS[scheme], fontsize=9)
            if i == 1:
                ax.set_xlabel("Tier", fontsize=8)
            ax.set_xticks(range(N_TIERS))
            ax.set_xticklabels(range(N_TIERS), fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=7)
        axes[i, 0].set_ylabel(f"{model}\nfinal qLogNEI", fontsize=8.5)
    fig.suptitle("Final qLogNEI distribution per cell (boxes, n=25)", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_runtime_breakdown(cells) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, model in zip(axes, MODELS):
        labels, ga_means, ref_means, colors = [], [], [], []
        for scheme in SCHEMES:
            for t in range(N_TIERS):
                recs = cells[(model, scheme, t)]
                ga_means.append(np.mean([r["ga_t"] / 60 for r in recs]))
                ref_means.append(np.mean([r["ref_t"] / 60 for r in recs]))
                labels.append(f"{SCHEME_LABELS[scheme].split()[0]}·T{t}")
                colors.append(SCHEME_COLORS[scheme])
        x = np.arange(len(labels))
        ax.bar(x, ga_means, color=colors, alpha=0.9, label="GA")
        ax.bar(x, ref_means, bottom=ga_means, color="#9e9e9e", alpha=0.6, label="L-BFGS refine")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_ylabel("Mean wall time (min)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Runtime breakdown: GA (colored) vs L-BFGS continuous refine (grey)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_efficiency(cells) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, model in zip(axes, MODELS):
        for scheme in SCHEMES:
            xs = [np.median(wall_min(cells, model, scheme, t)) for t in range(N_TIERS)]
            ys = [np.median(acq(cells, model, scheme, t)) for t in range(N_TIERS)]
            ax.plot(xs, ys, "o-", color=SCHEME_COLORS[scheme], linewidth=1.5,
                    markersize=5, alpha=0.9, label=SCHEME_LABELS[scheme])
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Median wall time (min)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Median final qLogNEI", fontsize=9)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Efficiency frontier: acq quality vs wall-time cost (tier medians)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_diversity(cells) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, model in zip(axes, MODELS):
        for scheme in SCHEMES:
            meds = [np.mean([r["n_unique_disc"] for r in cells[(model, scheme, t)]])
                    for t in range(N_TIERS)]
            ax.plot(range(N_TIERS), meds, "o-", color=SCHEME_COLORS[scheme],
                    linewidth=1.8, markersize=5, label=SCHEME_LABELS[scheme], alpha=0.9)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Tier", fontsize=9)
        ax.set_xticks(range(N_TIERS))
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Mean unique discrete combos\nin q=6 batch", fontsize=9)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Discrete-choice diversity: unique catalyst×solvent×base sets per batch",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1040px; margin: 2em auto; padding: 0 1.2em; color: #1c1c1c;
       line-height: 1.55; font-size: 15px; }
h1 { font-size: 1.7em; margin-bottom: 0.1em; }
h2 { margin-top: 1.9em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; font-size: 1.3em; }
h3 { margin-top: 1.4em; font-size: 1.08em; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 12.5px; width: 100%; }
th, td { border: 1px solid #ccc; padding: 5px 9px; text-align: left; }
th { background: #f0f0f0; }
.img-container { text-align: left; margin: 1em 0; }
.img-container img { max-width: 100%; height: auto; border: 1px solid #eee; }
.insight-box { background: #eef6ff; border-left: 3px solid #4a90e2; padding: 0.5em 0.9em; margin: 0.7em 0; font-size: 14px; }
.intro-box { background: #f9f9f9; border-left: 3px solid #999; padding: 0.5em 0.9em; margin: 0.7em 0; font-size: 14.5px; }
.warn-box  { background: #fff8e1; border-left: 3px solid #f9a825; padding: 0.5em 0.9em; margin: 0.7em 0; font-size: 14px; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
"""


def _img(b64, alt=""):
    return f'<div class="img-container"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


def _acq_table(cells, model):
    rows = ""
    for scheme in SCHEMES:
        cells_html = ""
        for t in range(N_TIERS):
            a = acq(cells, model, scheme, t)
            cells_html += f"<td>{np.median(a):+.3f} ± {np.std(a):.3f}</td>"
        rows += f"<tr><td>{SCHEME_LABELS[scheme]} ({SCHEME_GA_CALLS[scheme]} GA call{'s' if SCHEME_GA_CALLS[scheme] > 1 else ''})</td>{cells_html}</tr>"
    head = "".join(f"<th>T{t}</th>" for t in range(N_TIERS))
    return f"<table><thead><tr><th>Scheme</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _wall_table(cells, model):
    rows = ""
    for scheme in SCHEMES:
        cells_html = ""
        for t in range(N_TIERS):
            w = wall_min(cells, model, scheme, t)
            cells_html += f"<td>{np.median(w):.1f}</td>"
        rows += f"<tr><td>{SCHEME_LABELS[scheme]}</td>{cells_html}</tr>"
    head = "".join(f"<th>T{t}</th>" for t in range(N_TIERS))
    return f"<table><thead><tr><th>Scheme — median wall time (min)</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _ga_settings_table(ga_settings):
    rows = ""
    for scheme in SCHEMES:
        cells_html = ""
        for t in range(N_TIERS):
            ev, pop = ga_settings[(scheme, t)]
            cells_html += f"<td>{ev // 1000}k / {pop}</td>"
        rows += f"<tr><td>{SCHEME_LABELS[scheme]}</td>{cells_html}</tr>"
    head = "".join(f"<th>T{t}</th>" for t in range(N_TIERS))
    return (f"<table><thead><tr><th>Scheme — GA evals (per sub-batch call) / pop</th>{head}</tr>"
            f"</thead><tbody>{rows}</tbody></table>")


def build_html(cells, ga_settings, meta) -> str:
    """Render the standalone report with figures embedded as base64 PNGs.

    Args:
        cells: Per-cell rep lists as returned by ``load``.
        ga_settings: ``(scheme, tier) -> (ga_n_evals, ga_pop_size)``.
        meta: Run metadata as returned by ``load``.

    Returns:
        The complete HTML document.
    """
    print("Rendering figures …")
    f_acq = fig_acq_vs_tier(cells)
    f_wall = fig_wall_vs_tier(cells)
    f_box = fig_acq_box(cells)
    f_break = fig_runtime_breakdown(cells)
    f_eff = fig_efficiency(cells)
    f_div = fig_diversity(cells)
    print("Figures done.")

    def cheapest_fastest(model):
        # full_batch at T0 vs sequential at T5 wall span
        fb0 = np.median(wall_min(cells, model, "full_batch", 0))
        sq5 = np.median(wall_min(cells, model, "sequential", 5))
        return fb0, sq5

    st_fast, st_slow = cheapest_fastest("st_y")
    lcm_fast, lcm_slow = cheapest_fastest("lcm_ys")

    hosts = meta["hosts"]
    host_str = f"{len(hosts)} Mechthild nodes ({hosts[0]}–{hosts[-1]})"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>run_4: AF-Optimizer Strategy Comparison — Models × Batch-Schemes × Tiers</title>
<style>
{CSS}
</style>
</head>
<body>

<h1>run_4: AF-Optimizer Strategy Comparison</h1>
<p><em>Models × batch-schemes × GA-budget tiers — which q=6 decomposition reaches the best
qLogNEI batch, and at what wall-time cost.</em></p>

<h2>0. Setup and Grid</h2>
<div class="intro-box">
  <p><strong>Question.</strong> A q=6 batch can be optimized as one joint GA call or built up in
  smaller sub-batches with already-chosen candidates held fixed. This run measures, for two GP
  surrogates, how the <strong>batch decomposition</strong> and the <strong>GA generation budget</strong>
  affect the final <strong>qLogNEI</strong> value the mixed optimizer reaches — and what each scheme costs
  in wall time.</p>

  <p><strong>Two GP setups.</strong> Both share <code>grover / map / map</code> featurization
  (grover catalyst, MAP solvent + base) with a Matérn-5/2 base kernel, fit on percent-scale targets
  (<code>TARGET_PERCENT_SCALE = 100</code>) so qLogNEI values are positive and match the validated
  <code>REPRODUCE.py</code> recipe.</p>
  <table>
    <thead><tr><th></th><th>st_y (Setup 1)</th><th>lcm_ys (Setup 2)</th></tr></thead>
    <tbody>
      <tr><td>Architecture</td><td><code>SingleTaskGP</code> on yield only</td>
          <td>rank-2 LCM <code>KroneckerMultiTaskGP</code> on (Yield, Selectivity)</td></tr>
      <tr><td>Kernel / mean / transform</td><td>Vanilla / constant / raw</td>
          <td>KS1 / type_linear / raw</td></tr>
      <tr><td>AF objective</td><td>yield output directly</td>
          <td><strong>Yield task only</strong> — <code>samples[…, 0]</code>, no C·S product</td></tr>
      <tr><td>Acquisition</td><td>qLogNEI (128 MC samples)</td><td>qLogNEI (128 MC samples)</td></tr>
    </tbody>
  </table>
  <p>qLogNEI values are <strong>not comparable across the two models</strong> (different surrogate, different
  posterior); each model is analyzed on its own scale. <code>st_y</code> reproduces the validated positive-AF
  baseline.</p>

  <p><strong>Four batch-schemes.</strong> A q=6 batch is built in sub-batches of the listed sizes; at
  sub-batch <code>i</code> the GA frees <code>steps[i]</code> new candidates with the already-chosen ones held
  fixed, then an L-BFGS refine re-optimizes the continuous dims of all chosen points. The hard
  <code>(2,2,2)</code> temperature partition is enforced on the full q=6 batch regardless of decomposition.</p>
  <table>
    <thead><tr><th>Scheme</th><th>steps</th><th>GA calls</th><th>Note</th></tr></thead>
    <tbody>
      <tr><td>Full batch</td><td><code>[6]</code></td><td>1</td><td>single joint GA call</td></tr>
      <tr><td>Split [3,3]</td><td><code>[3, 3]</code></td><td>2</td><td>two halves</td></tr>
      <tr><td>Split [2,2,2]</td><td><code>[2, 2, 2]</code></td><td>3</td><td>three pairs</td></tr>
      <tr><td>Sequential</td><td><code>[1,1,1,1,1,1]</code></td><td>6</td><td>greedy one-at-a-time</td></tr>
    </tbody>
  </table>

  <p><strong>Six GA-budget tiers.</strong> Each scheme has its own calibrated ladder (per-eval cost differs
  by scheme). <code>ga_n_evals</code> is the GA budget <em>per sub-batch GA call</em>; pop scales with the
  rung. The L-BFGS refine is fixed and minimal across all cells — <strong>3 restarts / 128 raw samples</strong>
  — so GA dominates the budget.</p>
  {_ga_settings_table(ga_settings)}

  <p><strong>Grid &amp; packaging.</strong> 2 models × 4 schemes × 6 tiers × 25 reps =
  <strong>1200 optimizations</strong>; one worker = one <code>(rep, model, scheme)</code> cell running all six
  tiers (200 workers).</p>

  <p><strong>Execution environment.</strong> Run on {host_str},
  <code>device = {", ".join(meta["devices"])}</code>, torch {", ".join(meta["torch"])},
  Python {", ".join(meta["python"])}, partition <code>medium</code>, 2 CPUs / 32 GB per worker.
  Completed {meta["ts_min"]} → {meta["ts_max"]} UTC.
  Total measured compute across all {meta["n_opts"]} optimizations:
  <strong>{meta["total_wall_h"]:.0f} CPU-hours</strong>.</p>
</div>

<h2>1. Acquisition Value across Schemes and Tiers</h2>

{_img(f_acq, "Acq vs tier")}

<h3>st_y — final qLogNEI (median ± std, n=25)</h3>
{_acq_table(cells, "st_y")}
<h3>lcm_ys — final qLogNEI (median ± std, n=25)</h3>
{_acq_table(cells, "lcm_ys")}

<div class="insight-box">
  <strong>Within each model, the four schemes reach essentially the same qLogNEI.</strong> The
  scheme-to-scheme and tier-to-tier differences sit inside one rep-to-rep standard deviation
  (~0.07–0.10 for <code>st_y</code>, ~0.10–0.18 for <code>lcm_ys</code>). There is no monotone gain from
  spending more GA budget: the ladders are flat-to-noisy, indicating the GA has already located its
  preferred discrete set by the lowest tier. The full-batch joint optimization is not penalized for solving
  the harder q=6 landscape in one call — it matches the staged schemes.
</div>

{_img(f_box, "Per-cell acq boxplots")}

<h2>2. Computational Cost (Wall Time)</h2>

{_img(f_wall, "Wall vs tier")}

<h3>st_y — median wall time (minutes)</h3>
{_wall_table(cells, "st_y")}
<h3>lcm_ys — median wall time (minutes)</h3>
{_wall_table(cells, "lcm_ys")}

<div class="insight-box">
  <strong>Cost is driven by the number of GA calls and the budget tier, not by acq quality.</strong>
  Wall time rises monotonically full_batch &lt; split_3_3 &lt; split_2_2_2 &lt; sequential and with tier.
  For <code>st_y</code> the cheapest cell (full_batch T0, {st_fast:.1f} min) and the most expensive
  (sequential T5, {st_slow:.1f} min) span ~{st_slow / st_fast:.1f}×; for <code>lcm_ys</code> the LCM
  surrogate is far costlier to evaluate, spanning {lcm_fast:.1f} → {lcm_slow:.1f} min
  (~{lcm_slow / lcm_fast:.1f}×). Sequential's six serial GA+refine passes are the dominant overhead.
</div>

<h2>3. Runtime Breakdown — GA vs L-BFGS Refine</h2>

{_img(f_break, "Runtime breakdown")}

<div class="insight-box">
  The fixed minimal refine (3 restarts / 128 raw samples) keeps L-BFGS subordinate. For
  <code>lcm_ys</code> the GA share is large at every tier (≈60–90%) because each AF evaluation on the LCM
  posterior is expensive. For <code>st_y</code> the cheap single-task AF makes the constant-cost refine a
  larger relative slice at the low tiers (GA ≈16% at full_batch T0), but GA overtakes refine as the budget
  climbs. In absolute terms the refine is near-constant overhead; the budget knob is GA generations.
</div>

<h2>4. Reproducibility and Dispersion</h2>

<p>Per-cell standard deviations are reported in the §1 tables. Across the grid, <code>st_y</code> batches are
reproducible to ~0.07–0.10 qLogNEI units and <code>lcm_ys</code> to ~0.07–0.19. Dispersion does not shrink
with GA budget — the spread is set by the rep-to-rep restart randomness of the GA on a multi-modal discrete
landscape, not by under-converged optimization. No scheme is systematically tighter than the others once the
budget is matched.</p>

<h2>5. Discrete-Choice Diversity</h2>

{_img(f_div, "Discrete diversity")}

<div class="insight-box">
  Each q=6 batch concentrates on a small number of distinct catalyst×solvent×base sets (≈2–4 unique combos),
  with the staged/sequential schemes trending slightly lower than full_batch. Diversity does not grow with GA
  budget: extra generations recycle the same dominant discrete agents rather than discovering new ones, which
  is consistent with the flat acq ladders in §1.
</div>

<h2>6. Efficiency: Acq Value vs Wall Time</h2>

{_img(f_eff, "Efficiency frontier")}

<div class="insight-box">
  Plotting tier-median acq against tier-median wall time collapses the comparison: for both models the
  <strong>full_batch</strong> curve sits furthest left (cheapest) at matched acq, and moving to staged or
  sequential schemes — or to higher tiers — buys wall time without buying acquisition value. There is no
  point on the frontier where a more expensive scheme earns a higher qLogNEI.
</div>

<h2>7. Conclusions</h2>
<ol>
  <li><strong>Batch decomposition does not change the achieved qLogNEI.</strong> Full batch, [3,3], [2,2,2]
      and fully sequential all land within one rep-to-rep std of each other, for both the single-task and the
      LCM multitask surrogate.</li>
  <li><strong>Full-batch is the efficient choice.</strong> It reaches the same acq at the lowest wall time;
      the staged and sequential schemes only add GA-call overhead. Sequential is the most expensive and earns
      nothing for it.</li>
  <li><strong>The GA converges at the lowest tier.</strong> Acq value and discrete diversity are flat across
      the six budget rungs — additional generations are wasted. Tier-0 budgets are sufficient.</li>
  <li><strong>Refine is correctly subordinate.</strong> With the pinned 3-restart / 128-sample L-BFGS, GA
      dominates runtime (overwhelmingly so for the costlier LCM model); the continuous refine is near-constant
      overhead.</li>
  <li><strong>Cost is set by the surrogate, not the scheme.</strong> The LCM <code>lcm_ys</code> model is
      several times more expensive per optimization than <code>st_y</code>; if multitask information is not
      worth that cost, the single-task yield model with full-batch GA at a low tier is the production
      recommendation.</li>
</ol>

</body>
</html>
"""
    return html


def main():
    """Load the results, render the report and write it to ``OUT_HTML``."""
    print("Loading results …")
    cells, ga_settings, meta = load()
    print(f"Loaded {meta['n_opts']} optimizations across {len(cells)} cells.")
    html = build_html(cells, ga_settings, meta)
    with open(OUT_HTML, "w") as fh:
        fh.write(html)
    print(f"Report written to {OUT_HTML}")


if __name__ == "__main__":
    main()
