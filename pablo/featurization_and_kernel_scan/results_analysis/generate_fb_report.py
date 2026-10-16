"""Build the fully-Bayesian-vs-MAP comparison report (standalone HTML).

Reads ``results/fully_bayesian_eval/full_results.json``, a stored result file
whose producing script is not in the repository, and renders a self-contained
report with inline base64 charts. Covers the four models that completed cleanly — the
non-SAAS fully-Bayesian single-task model (``fb_st``) is omitted (it crashed
under NUTS; see the report note).

Run::

    python -m pablo.featurization_and_kernel_scan.results_analysis.generate_fb_report
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "fully_bayesian_eval"
JSON_PATH = RESULTS_DIR / "full_results.json"
HTML_PATH = RESULTS_DIR / "fully_bayesian_eval_report.html"

ORDER = ["map_st", "gem1", "saas_st", "saas_mt"]
LABELS = {
    "map_st": "MAP single-task (st_y)",
    "gem1": "MAP multitask — gem1 (lcm_cs)",
    "saas_st": "Fully-Bayesian SAAS single-task",
    "saas_mt": "Fully-Bayesian SAAS multitask (C·S)",
}
INFERENCE = {"map_st": "MAP", "gem1": "MAP", "saas_st": "NUTS", "saas_mt": "NUTS"}
COLORS = {"map_st": "#2c7fb8", "gem1": "#08306b", "saas_st": "#fd8d3c", "saas_mt": "#bd0026"}


def _png(fig) -> str:
    """Serialize a matplotlib figure to a base64 ``<img>`` data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bar(values: dict[str, float], title: str, ylabel: str, *, log: bool = False,
         zero_line: bool = False, fmt: str = "{:.3f}") -> str:
    """Single-series bar chart over the models, returned as a data URI."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = list(range(len(ORDER)))
    ax.bar(xs, [values[m] for m in ORDER], color=[COLORS[m] for m in ORDER], width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m].replace(" — ", "\n").replace(" (", "\n(") for m in ORDER],
                       fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, weight="bold")
    if log:
        ax.set_yscale("log")
    if zero_line:
        ax.axhline(0.0, color="#555", lw=1.0, ls="--")
    for x, m in zip(xs, ORDER):
        v = values[m]
        ax.annotate(fmt.format(v), (x, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8,
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.grid(axis="y", alpha=0.25)
    return _png(fig)


def _grouped_mae_rmse(summ: dict[str, dict]) -> str:
    """Grouped MAE / RMSE bars (percentage points; lower is better)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = list(range(len(ORDER)))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [summ[m]["mae_pts"] for m in ORDER], width=w,
           label="MAE", color="#4393c3")
    ax.bar([x + w / 2 for x in xs], [summ[m]["rmse_pts"] for m in ORDER], width=w,
           label="RMSE", color="#d6604d")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m].replace(" — ", "\n").replace(" (", "\n(") for m in ORDER],
                       fontsize=8)
    ax.set_ylabel("yield error (percentage points)")
    ax.set_title("Point error — MAE & RMSE (lower is better)", fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return _png(fig)


def _pred_vs_actual(models: dict[str, dict]) -> str:
    """Predicted vs. actual yield (%) for every held-out point, by model."""
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lo, hi = 0, 75
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1.0, ls="--", label="perfect")
    for m in ORDER:
        pts = models[m]["points"]
        ax.scatter([p["y_true"] * 100 for p in pts], [p["mu"] * 100 for p in pts],
                   s=55, color=COLORS[m], label=LABELS[m].split(" — ")[0], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("actual yield (%)")
    ax.set_ylabel("predicted yield (%)")
    ax.set_title("Predicted vs. actual on the top-5 points\n(all models sit below the line — "
                 "systematic under-prediction)", fontsize=10, weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    return _png(fig)


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def _summary_table(summ: dict[str, dict]) -> str:
    cols = [
        ("crpss", "CRPSS ↑", 3), ("mae_pts", "MAE (pts) ↓", 2),
        ("rmse_pts", "RMSE (pts) ↓", 2), ("signed_err_pts", "signed err (pts)", 2),
        ("msll", "MSLL ↓", 3), ("fit_time_mean_s", "fit/​fit (s)", 1),
        ("fit_time_total_s", "fit total (s)", 1), ("eval_time_total_s", "eval total (s)", 2),
    ]
    best = {
        "crpss": max(summ[m]["crpss"] for m in ORDER),
        "mae_pts": min(summ[m]["mae_pts"] for m in ORDER),
        "rmse_pts": min(summ[m]["rmse_pts"] for m in ORDER),
        "msll": min(summ[m]["msll"] for m in ORDER),
        "fit_time_mean_s": min(summ[m]["fit_time_mean_s"] for m in ORDER),
    }
    head = "".join(f"<th>{label}</th>" for _, label, _ in cols)
    rows = []
    for m in ORDER:
        cells = [f'<td class="model"><b>{LABELS[m]}</b><br><span class="tag {INFERENCE[m].lower()}">'
                 f'{INFERENCE[m]}</span></td>']
        for key, _, nd in cols:
            v = summ[m][key]
            mark = " win" if key in best and abs(v - best[key]) < 1e-9 else ""
            cells.append(f'<td class="num{mark}">{_fmt(v, nd)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="summary"><thead><tr><th>model</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def build_report() -> str:
    """Build the comparison report from ``JSON_PATH``.

    Returns:
        Complete HTML document with the summary table and inline base64 charts
        for the models in ``ORDER``.
    """
    data = json.loads(JSON_PATH.read_text())
    models = data["models"]
    summ = {m: models[m]["summary"] for m in ORDER}
    nuts = data["nuts"]

    gem = summ["gem1"]
    speed_vs_saas_mt = summ["saas_mt"]["fit_time_mean_s"] / gem["fit_time_mean_s"]
    speed_vs_saas_st = summ["saas_st"]["fit_time_mean_s"] / gem["fit_time_mean_s"]
    saas_mt_vs_map_st = summ["saas_mt"]["fit_time_mean_s"] / summ["map_st"]["fit_time_mean_s"]

    chart_runtime = _bar({m: summ[m]["fit_time_mean_s"] for m in ORDER},
                         "Mean model-fit time per fold (log scale)", "seconds / fit",
                         log=True, fmt="{:.0f}s")
    chart_crpss = _bar({m: summ[m]["crpss"] for m in ORDER},
                       "CRPSS — skill vs. train-mean baseline (higher is better)", "CRPSS",
                       fmt="{:.3f}")
    chart_msll = _bar({m: summ[m]["msll"] for m in ORDER},
                      "MSLL — calibration vs. baseline (lower is better; <0 beats baseline)",
                      "MSLL", zero_line=True, fmt="{:.2f}")
    chart_err = _grouped_mae_rmse(summ)
    chart_scatter = _pred_vs_actual(models)

    pts0 = models["gem1"]["points"]
    setup_pts = ", ".join(f"{p['yield_pct']:.0f}%" for p in sorted(pts0, key=lambda p: -p["yield_pct"]))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Fully-Bayesian vs. MAP — top-yield evaluation</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; color:#1a1a1a;
        max-width: 980px; margin: 32px auto; padding: 0 18px; line-height: 1.5; }}
 h1 {{ font-size: 1.7em; margin-bottom: 4px; }}
 h2 {{ margin-top: 34px; border-bottom: 2px solid #08306b; padding-bottom: 4px; }}
 .sub {{ color:#666; margin-top:0; }}
 table.summary {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.93em; }}
 table.summary th, table.summary td {{ border: 1px solid #ccc; padding: 6px 9px; text-align: right; }}
 table.summary th {{ background:#08306b; color:#fff; }}
 td.model {{ text-align: left; }}
 td.num.win {{ background:#d9f0d3; font-weight:700; }}
 .tag {{ font-size:0.72em; padding:1px 6px; border-radius:9px; color:#fff; }}
 .tag.map {{ background:#2c7fb8; }} .tag.nuts {{ background:#bd0026; }}
 .grid {{ display:flex; flex-wrap:wrap; gap:14px; }}
 .grid img {{ max-width: 100%; }}
 .card {{ flex: 1 1 440px; }}
 .key {{ background:#f3f6fb; border-left:4px solid #08306b; padding:10px 14px; margin:14px 0; }}
 .note {{ background:#fff7ec; border-left:4px solid #fd8d3c; padding:10px 14px; font-size:0.92em; }}
 img {{ border:1px solid #e3e3e3; border-radius:4px; }}
 code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; }}
</style></head><body>

<h1>Fully-Bayesian vs. MAP GP inference on the top-yield NHC points</h1>
<p class="sub">Leave-one-out over the 5 highest-yield non-replicate experiments
({setup_pts}), N=64 training rows, gem1 feature set
(<code>grover / physical / structured</code>). Fully-Bayesian models marginalize
hyperparameters with NUTS (warmup {nuts['warmup']}, {nuts['num_samples']} samples,
thinning {nuts['thinning']} → {nuts['num_samples'] // nuts['thinning']} draws);
MAP models use 5-restart L-BFGS. Yield for the multitask models is the
sample-based product <code>C·S</code>. 20 fits (4 models × 5 points).</p>

<div class="key">
<b>Bottom line.</b> The MAP multitask baseline <b>gem1 wins every accuracy metric</b>
(CRPSS {gem['crpss']:.3f}, MAE {gem['mae_pts']:.1f} pts, RMSE {gem['rmse_pts']:.1f} pts,
MSLL {gem['msll']:.2f}) — and is far cheaper than the fully-Bayesian models:
<b>{speed_vs_saas_st:.0f}× faster</b> than SAAS single-task and
<b>{speed_vs_saas_mt:.0f}× faster</b> than SAAS multitask per fit. NUTS marginalization
bought <i>no</i> accuracy here, at one to two orders of magnitude more compute.
</div>

<h2>Summary</h2>
{_summary_table(summ)}
<p class="sub">↑ higher is better, ↓ lower is better. Green = best in column. MAE/RMSE/signed
error in yield percentage points; MSLL and CRPSS are scored against the constant
<code>N(ȳ_train, σ²_train)</code> baseline.</p>

<h2>Runtime — the cost of going fully Bayesian</h2>
<div class="grid"><div class="card"><img src="{chart_runtime}"></div>
<div class="card">
<p>Mean wall-clock per single fit: <b>map_st {summ['map_st']['fit_time_mean_s']:.0f} s</b>,
<b>gem1 {summ['gem1']['fit_time_mean_s']:.0f} s</b>,
<b>saas_st {summ['saas_st']['fit_time_mean_s']:.0f} s</b>,
<b>saas_mt {summ['saas_mt']['fit_time_mean_s']:.0f} s</b>.
The NUTS models are {saas_mt_vs_map_st:.0f}× (multitask) and
{speed_vs_saas_st * summ['gem1']['fit_time_mean_s'] / summ['map_st']['fit_time_mean_s']:.0f}×
(single-task) the cost of the cheapest MAP fit. Evaluation/prediction time is
negligible for every model (≤0.25 s total over 5 points).</p></div></div>

<h2>Predictive accuracy</h2>
<div class="grid">
<div class="card"><img src="{chart_crpss}"></div>
<div class="card"><img src="{chart_msll}"></div>
</div>
<div class="grid">
<div class="card"><img src="{chart_err}"></div>
<div class="card"><img src="{chart_scatter}"></div>
</div>

<div class="key">
<b>Calibration is a multitask story, not a Bayesian one.</b> The two single-task models are
<i>worse than the trivial baseline</i> on MSLL (map_st {summ['map_st']['msll']:.2f},
saas_st {summ['saas_st']['msll']:.2f} — both &gt; 0), while the joint (C,S) models are
well calibrated (saas_mt {summ['saas_mt']['msll']:.2f}, gem1 {summ['gem1']['msll']:.2f}).
Jointly modelling conversion and selectivity helped the predictive variance more than
fully-Bayesian marginalization did.
</div>

<p>Every model sits below the diagonal: all four <b>systematically under-predict</b> the
top points (signed error equals MAE for each, i.e. all 5 points are under-shot). This is
the expected penalty for extrapolating past the training frontier — the GP regresses the
best experiments toward the population mean.</p>

<h2>Note — the missing fifth model</h2>
<div class="note">
The non-SAAS fully-Bayesian single-task model (<code>fb_st</code>, the NUTS counterpart of
<code>map_st</code>) is excluded: all five of its fits crashed. Its KS1 kernel carried the
production <code>GreaterThan(0.05)</code> lengthscale floor (a MAP regularizer), but NUTS
samples lengthscales directly from the <code>LogNormal</code> prior, whose support extends
below 0.05; once a draw fell under the floor, the constraint's inverse transform raised
<code>RuntimeError: ... out of bounds ... GreaterThan(5.000E-02)</code>. The fix (drop the
floor → <code>Positive</code> constraint for the fully-Bayesian build) is in place but the
re-run was not pursued.
</div>

</body></html>"""


def main() -> None:
    """Write the report to ``HTML_PATH``."""
    HTML_PATH.write_text(build_report())
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
