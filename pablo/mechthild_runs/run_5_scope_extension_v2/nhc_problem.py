"""NHC search space, synthetic yield and featurization-scan surrogate of run_5.

A point is ``[cont (4) | cat (16) | solv (4) | base (4)]`` (``config.TOTAL_DIM``
= 28): the continuous conditions in raw units (formaldehyde eq, catalyst
loading, temperature °C, time h) followed by the ``grover_pca`` catalyst row,
the ``Hansen`` solvent row and the ``grover_pca`` base row. The base block is
zero for a catalyst that needs no external base, as in the featurization
scan's feature assembly. The starting set is every recorded row of
``nhc_doe.xlsx``, replicates included, and the catalysts available at the start
are the ones the workbook marks available for the next batch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from scipy.optimize import minimize as scipy_minimize
from torch import Tensor
from torch.quasirandom import SobolEngine

from benchmark_problems.synthetic import Ackley, Gallagher101, LunacekBiRastrigin
from pablo.featurization_and_kernel_scan import paths as fks_paths
from pablo.featurization_and_kernel_scan.build_experiments import build_experiments_table
from pablo.featurization_and_kernel_scan.core import config as fks_config
from pablo.featurization_and_kernel_scan.core.kernels import make_kernel_from_config
from pablo.featurization_and_kernel_scan.core.models import make_singletask_gp
from pablo.featurization_and_kernel_scan.cross_validation.fit_eval import _fit_multistart
from pablo.main import DEFAULT_EXCEL_FILE, load_available_agents
from pablo.mechthild_runs.run_5_scope_extension_v2.config import (
    BASE_DIM,
    BASE_FEATURIZATION,
    BASE_KERNEL,
    BENCHMARK_CLASS_MAP,
    CAT_DIM,
    CAT_FEATURIZATION,
    CONTINUOUS_COLUMNS,
    FIXED_DIM_INDEX,
    FIXED_DIM_VALUE,
    KERNEL_STRUCTURE,
    LANDSCAPE_OFFSET_HALF_WIDTH,
    LANDSCAPE_WINDOW,
    MEAN_FUNCTION,
    N_CONTINUOUS_DIMS,
    OUTCOME_TRANSFORM,
    PRIOR_SCHEME,
    RANGE_REFINE,
    RANGE_REFINE_MAXITER,
    RANGE_SOBOL_POINTS,
    SOLVENT_DIM,
    SOLVENT_FEATURIZATION,
    TOTAL_DIM,
)
from pablo.utils.feature_assembly import (
    _form_onehot_table,
    _load_numeric_descriptors,
    _resolve_feature_parquet,
)
from pablo.utils.outcome_transforms import forward as forward_outcome

DTYPE = torch.float64


@dataclass(frozen=True)
class NHCSpace:
    """Component tables, bounds and the recorded experiments of the NHC space.

    Attributes:
        cat_table: ``(n_cat, CAT_DIM)`` catalyst rows, one per library catalyst.
        cat_ids: Featurization-scan catalyst id of every ``cat_table`` row.
        cat_is_salt: ``(n_cat,)`` bool, whether the catalyst needs an external
            base (pre-catalyst form ``salt``).
        sol_table: ``(n_sol, SOLVENT_DIM)`` rows of the available solvents.
        sol_names: Solvent name of every ``sol_table`` row.
        bas_table: ``(n_bas, BASE_DIM)`` rows of the available bases.
        bas_names: Base label (``bases.parquet`` comment) of every ``bas_table`` row.
        bounds: ``(2, TOTAL_DIM)`` bounds; continuous from the scan's library
            bounds, discrete the column range of the tables (zero included for
            the base block).
        initial_available: ``(n_cat,)`` bool, catalysts available at the start.
        start_x: ``(n_start, TOTAL_DIM)`` recorded experiments.
        start_cat_index: ``(n_start,)`` ``cat_table`` row of every experiment.
        start_replicate_id: ``(n_start,)`` replicate group (−1 for none).
    """

    cat_table: Tensor
    cat_ids: list[int]
    cat_is_salt: Tensor
    sol_table: Tensor
    sol_names: list[str]
    bas_table: Tensor
    bas_names: list[str]
    bounds: Tensor
    initial_available: Tensor
    start_x: Tensor
    start_cat_index: Tensor
    start_replicate_id: np.ndarray

    @property
    def feature_dims(self) -> dict[str, list[int]]:
        """Column indices of the four blocks in the point layout."""
        c, k, s = N_CONTINUOUS_DIMS, CAT_DIM, SOLVENT_DIM
        return {
            "cont": list(range(c)),
            "cat": list(range(c, c + k)),
            "solv": list(range(c + k, c + k + s)),
            "base": list(range(c + k + s, TOTAL_DIM)),
        }

    def assemble(self, cont: Tensor, cat_idx: Tensor, sol_idx: Tensor, bas_idx: Tensor) -> Tensor:
        """Points from continuous values and component row indices.

        Args:
            cont: ``(n, N_CONTINUOUS_DIMS)`` continuous values.
            cat_idx: ``(n,)`` ``cat_table`` rows.
            sol_idx: ``(n,)`` ``sol_table`` rows.
            bas_idx: ``(n,)`` ``bas_table`` rows; ignored where the catalyst is
                not a salt (the base block is zero there).

        Returns:
            ``(n, TOTAL_DIM)`` points.
        """
        base = self.bas_table[bas_idx] * self.cat_is_salt[cat_idx].to(DTYPE).unsqueeze(-1)
        return torch.cat([cont.to(DTYPE), self.cat_table[cat_idx], self.sol_table[sol_idx], base], dim=-1)

    def library_rows(self) -> tuple[Tensor, Tensor, Tensor]:
        """Every distinct ``(cat, solv, base)`` combination of the library.

        Returns:
            ``(cat_idx, sol_idx, bas_idx)``, each ``(n_rows,)``; a non-salt
            catalyst appears once per solvent with ``bas_idx = 0``.
        """
        n_sol, n_bas = self.sol_table.shape[0], self.bas_table.shape[0]
        cats, sols, bass = [], [], []
        for i in range(self.cat_table.shape[0]):
            n_b = n_bas if bool(self.cat_is_salt[i]) else 1
            cats.append(torch.full((n_sol * n_b,), i, dtype=torch.long))
            sols.append(torch.arange(n_sol).repeat_interleave(n_b))
            bass.append(torch.arange(n_b).repeat(n_sol))
        return torch.cat(cats), torch.cat(sols), torch.cat(bass)


def _table(dirname: str, ids: list[int], width: int) -> Tensor:
    """Descriptor rows of ``ids`` from ``featurizations/<dirname>``, checked for width."""
    feats = _load_numeric_descriptors(_resolve_feature_parquet(fks_paths.FEAT_DIR / dirname))
    if feats.shape[1] != width:
        raise ValueError(f"{dirname} has {feats.shape[1]} columns, config expects {width}")
    missing = sorted(set(ids) - set(feats.index))
    if missing:
        raise ValueError(f"{dirname} has no rows for ids {missing}")
    return torch.tensor(feats.loc[ids].to_numpy(dtype=np.float64), dtype=DTYPE)


def load_nhc_space(excel_file: Path = DEFAULT_EXCEL_FILE) -> NHCSpace:
    """Load the component tables and the recorded experiments.

    Args:
        excel_file: The experimental-design workbook (read-only).

    Returns:
        The NHC space.

    Raises:
        ValueError: If a table's width disagrees with ``config``, a component
            has no descriptor row, an experiment uses an unavailable solvent or
            base, or a salt-catalyst experiment has no base.
    """
    catalysts = pd.read_parquet(fks_paths.CATALYSTS_PARQUET)
    solvents = pd.read_parquet(fks_paths.SOLVENTS_PARQUET)
    bases = pd.read_parquet(fks_paths.BASES_PARQUET)
    avail_cat, avail_sol, avail_bas = load_available_agents(excel_file)

    cat_ids = [int(i) for i in catalysts["id"]]
    sol_ids = [int(i) for i in solvents["id"] if int(i) in set(avail_sol)]
    bas_ids = [int(i) for i in bases["id"] if int(i) in set(avail_bas)]
    cat_table = _table(CAT_FEATURIZATION, cat_ids, CAT_DIM)
    sol_table = _table(f"solvent_{SOLVENT_FEATURIZATION}", sol_ids, SOLVENT_DIM)
    bas_table = _table(f"base_{BASE_FEATURIZATION}", bas_ids, BASE_DIM)
    forms = _form_onehot_table(fks_paths.FEAT_DIR).loc[cat_ids]
    cat_is_salt = torch.tensor(forms["form_salt"].to_numpy() == 1)
    sol_names = solvents.set_index("id").loc[sol_ids, "name"].tolist()
    bas_names = bases.set_index("id").loc[bas_ids, "comment"].tolist()

    meta = json.loads((fks_paths.FEAT_DIR / "continuous" / "metadata.json").read_text())
    cont_lo = [meta["bounds"][c][0] for c in CONTINUOUS_COLUMNS]
    cont_hi = [meta["bounds"][c][1] for c in CONTINUOUS_COLUMNS]
    zero = torch.zeros(1, BASE_DIM, dtype=DTYPE)
    lo = torch.cat([torch.tensor(cont_lo, dtype=DTYPE), cat_table.amin(0), sol_table.amin(0),
                    torch.cat([bas_table, zero]).amin(0)])
    hi = torch.cat([torch.tensor(cont_hi, dtype=DTYPE), cat_table.amax(0), sol_table.amax(0),
                    torch.cat([bas_table, zero]).amax(0)])

    experiments = build_experiments_table(excel_file, fks_paths.LIB_DIR)
    cat_row = {c: i for i, c in enumerate(cat_ids)}
    sol_row = {n: i for i, n in enumerate(sol_names)}
    bas_row = {n: i for i, n in enumerate(bas_names)}
    unknown_sol = sorted(set(experiments["solvent"]) - set(sol_row))
    if unknown_sol:
        raise ValueError(f"Experiments use solvents outside the available set: {unknown_sol}")
    cat_idx = torch.tensor([cat_row[int(c)] for c in experiments["catalyst_id"]])
    sol_idx = torch.tensor([sol_row[s] for s in experiments["solvent"]])
    bas_idx = torch.zeros(len(experiments), dtype=torch.long)
    for r, (b, active) in enumerate(zip(experiments["base"], experiments["base_active"])):
        is_salt = bool(cat_is_salt[cat_idx[r]])
        if bool(active) != is_salt:
            raise ValueError(f"Experiment {r}: base_active={active} but catalyst salt flag {is_salt}")
        if is_salt:
            if b not in bas_row:
                raise ValueError(f"Experiment {r}: salt catalyst with base {b!r} outside the available set")
            bas_idx[r] = bas_row[b]
    cont = torch.tensor(experiments[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64), dtype=DTYPE)

    initial_available = torch.zeros(len(cat_ids), dtype=torch.bool)
    initial_available[[cat_row[int(c)] for c in avail_cat]] = True

    space = NHCSpace(
        cat_table=cat_table,
        cat_ids=cat_ids,
        cat_is_salt=cat_is_salt,
        sol_table=sol_table,
        sol_names=sol_names,
        bas_table=bas_table,
        bas_names=bas_names,
        bounds=torch.stack([lo, hi]),
        initial_available=initial_available,
        start_x=torch.empty(0, TOTAL_DIM, dtype=DTYPE),
        start_cat_index=cat_idx,
        start_replicate_id=experiments["replicate_id"].to_numpy(dtype=np.int64),
    )
    return NHCSpace(**{**space.__dict__, "start_x": space.assemble(cont, cat_idx, sol_idx, bas_idx)})


def yield_noise_sd() -> float:
    """Pooled replicate standard deviation of the raw yield fraction."""
    return math.sqrt(fks_config.NOISE_EVIDENCE[("Y", "raw")].variance)


def build_benchmark(benchmark_name: str, seed: int) -> Ackley | Gallagher101 | LunacekBiRastrigin:
    """The ``TOTAL_DIM``-dimensional benchmark; the BBOB ones take ``seed`` for their rotations.

    Raises:
        ValueError: If ``benchmark_name`` is not a key of ``BENCHMARK_CLASS_MAP``.
    """
    if benchmark_name == "Ackley":
        return Ackley(d=TOTAL_DIM)
    if benchmark_name == "Gallagher101":
        return Gallagher101(d=TOTAL_DIM, seed=seed)
    if benchmark_name == "LunacekBiRastrigin":
        return LunacekBiRastrigin(d=TOTAL_DIM, seed=seed)
    raise ValueError(f"Unknown benchmark: {benchmark_name}; expected one of {list(BENCHMARK_CLASS_MAP)}")


class SyntheticYield:
    """Synthetic yield landscape on the NHC space for one benchmark and seed.

    Column ``j`` of a point is mapped to the benchmark's coordinate ``j`` by
    ``u = (x − lo)/(hi − lo)`` over the space bounds, reversed (``1 − u``) for a
    random half of the columns, then ``b_lo + (b_hi − b_lo)·(o_j + w·u)`` with
    window ``w = LANDSCAPE_WINDOW`` and a random offset ``o_j``. The benchmark
    value ``f`` is rescaled to a yield ``(f − f_min)/(f_max − f_min)`` with the
    extremes over the attainable space (library rows, continuous bounds, time at
    ``FIXED_DIM_VALUE``), so the best attainable point has yield 1.

    Args:
        benchmark_name: Key of ``BENCHMARK_CLASS_MAP``.
        seed: Landscape seed (sign flips, offsets, and the benchmark's own seed
            of Gallagher101 and LunacekBiRastrigin).
        space: The NHC space.

    Raises:
        ValueError: If ``benchmark_name`` is unknown or the attainable range is
            degenerate.
    """

    def __init__(self, benchmark_name: str, seed: int, space: NHCSpace):
        self.benchmark = build_benchmark(benchmark_name, seed)
        self.space = space
        generator = torch.Generator().manual_seed(seed)
        self.flip = torch.rand(TOTAL_DIM, generator=generator) < 0.5
        self.offset = 0.25 + LANDSCAPE_OFFSET_HALF_WIDTH * (
            2.0 * torch.rand(TOTAL_DIM, generator=generator, dtype=DTYPE) - 1.0
        )
        self.bench_bounds = self.benchmark.bounds().to(DTYPE)
        self.f_min, self.f_max, self.x_max = self._attainable_range(seed)
        if not self.f_max > self.f_min:
            raise ValueError(f"Degenerate attainable range [{self.f_min}, {self.f_max}]")

    def to_benchmark(self, x: Tensor) -> Tensor:
        """Benchmark coordinates ``(n, TOTAL_DIM)`` of points ``x`` ``(n, TOTAL_DIM)``."""
        lo, hi = self.space.bounds
        u = (x.to(DTYPE) - lo) / (hi - lo)
        u = torch.where(self.flip, 1.0 - u, u)
        b_lo, b_hi = self.bench_bounds
        return b_lo + (b_hi - b_lo) * (self.offset + LANDSCAPE_WINDOW * u)

    def f(self, x: Tensor) -> Tensor:
        """Benchmark value ``(n,)`` of points ``x`` ``(n, TOTAL_DIM)``."""
        return self.benchmark.value(self.to_benchmark(x)).to(DTYPE).reshape(-1)

    def yield_true(self, x: Tensor) -> Tensor:
        """Noise-free synthetic yield ``(n, 1)`` of points ``x`` ``(n, TOTAL_DIM)``."""
        return ((self.f(x) - self.f_min) / (self.f_max - self.f_min)).unsqueeze(-1)

    def observe(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """Measured yield ``(n, 1)``: true yield plus replicate noise, censored to [0, 1]."""
        noise = yield_noise_sd() * torch.randn(x.shape[0], 1, generator=generator, dtype=DTYPE)
        return (self.yield_true(x) + noise).clamp(0.0, 1.0)

    def _attainable_range(self, seed: int) -> tuple[float, float, Tensor]:
        """``(f_min, f_max, x_max)`` over the attainable space; ``x_max`` is ``(1, TOTAL_DIM)``."""
        cat_idx, sol_idx, bas_idx = self.space.library_rows()
        free = [i for i in range(N_CONTINUOUS_DIMS) if i != FIXED_DIM_INDEX]
        lo, hi = self.space.bounds[:, :N_CONTINUOUS_DIMS]
        u = SobolEngine(len(free), scramble=True, seed=seed).draw(RANGE_SOBOL_POINTS).to(DTYPE)
        cont = torch.empty(RANGE_SOBOL_POINTS, N_CONTINUOUS_DIMS, dtype=DTYPE)
        cont[:, free] = lo[free] + (hi[free] - lo[free]) * u
        cont[:, FIXED_DIM_INDEX] = FIXED_DIM_VALUE
        discrete = self.space.assemble(cont[:1].expand(cat_idx.shape[0], -1), cat_idx, sol_idx, bas_idx)
        discrete = discrete[:, N_CONTINUOUS_DIMS:]  # (n_rows, d_disc)
        values = torch.empty(discrete.shape[0], RANGE_SOBOL_POINTS, dtype=DTYPE)
        chunk = max(1, 4096 // RANGE_SOBOL_POINTS)
        for start in range(0, discrete.shape[0], chunk):
            rows = discrete[start:start + chunk]
            x = torch.cat([
                cont.unsqueeze(0).expand(rows.shape[0], -1, -1),
                rows.unsqueeze(1).expand(-1, RANGE_SOBOL_POINTS, -1),
            ], dim=-1).reshape(-1, TOTAL_DIM)
            values[start:start + chunk] = self.f(x).reshape(rows.shape[0], RANGE_SOBOL_POINTS)

        def refine(flat_idx: int, sign: float) -> tuple[float, Tensor]:
            r, c = divmod(flat_idx, RANGE_SOBOL_POINTS)
            base = torch.cat([cont[c], discrete[r]]).unsqueeze(0)

            def objective(v: np.ndarray) -> float:
                x = base.clone()
                x[0, free] = torch.as_tensor(v, dtype=DTYPE)
                return -sign * float(self.f(x)[0])

            res = scipy_minimize(
                objective, cont[c, free].numpy(), method="L-BFGS-B",
                bounds=list(zip(lo[free].tolist(), hi[free].tolist())),
                options={"maxiter": RANGE_REFINE_MAXITER},
            )
            x = base.clone()
            x[0, free] = torch.as_tensor(res.x, dtype=DTYPE)
            return float(self.f(x)[0]), x

        flat = values.reshape(-1)
        best = [refine(int(i), 1.0) for i in flat.topk(RANGE_REFINE).indices]
        worst = [refine(int(i), -1.0) for i in (-flat).topk(RANGE_REFINE).indices]
        f_max, x_max = max(best, key=lambda t: t[0])
        f_max = max(f_max, float(flat.max()))
        f_min = min(min(v for v, _ in worst), float(flat.min()))
        return f_min, f_max, x_max


def fit_surrogate(train_x: Tensor, train_y: Tensor, feature_dims: dict[str, list[int]]) -> SingleTaskGP:
    """Fit the featurization-scan model on measured yields.

    The model is ``KERNEL_STRUCTURE`` over ``BASE_KERNEL`` block kernels with the
    ``PRIOR_SCHEME`` lengthscale priors, a constant mean, the replicate-evidence
    noise prior of yield on the ``OUTCOME_TRANSFORM`` scale, ``Normalize`` inputs
    and ``Standardize`` outcomes, fitted by the scan's multi-start L-BFGS. The
    model's outcome is the yield on the ``OUTCOME_TRANSFORM`` scale.

    Args:
        train_x: ``(n, TOTAL_DIM)`` points.
        train_y: ``(n, 1)`` measured yields in [0, 1].
        feature_dims: Block name → columns of ``train_x``.

    Returns:
        The fitted model in eval mode.

    Raises:
        ValueError: If ``MEAN_FUNCTION`` is not ``"constant"`` (the only mean
            this layout carries).
        NoConvergedRestartError: If no restart ends with a finite MLL.
    """
    if MEAN_FUNCTION != "constant":
        raise ValueError(f"run_5 builds the constant mean only, config has {MEAN_FUNCTION!r}")
    kernel = make_kernel_from_config(
        kernel_structure=KERNEL_STRUCTURE,
        feature_dims=feature_dims,
        active_flag_dim=-1,
        base_kernel_type=BASE_KERNEL,
        lengthscale_scale=fks_config.LENGTHSCALE_LOGNORMAL_SCALE,
        prior_scheme=PRIOR_SCHEME,
    )
    evidence = fks_config.NOISE_EVIDENCE[("Y", OUTCOME_TRANSFORM)]
    model = make_singletask_gp(
        train_x,
        forward_outcome(train_y, OUTCOME_TRANSFORM),
        kernel,
        mean_module=None,
        form_id_dim=-1,
        noise_prior_alpha=evidence.alpha,
        noise_prior_beta=evidence.beta,
    )
    with gpytorch.settings.cholesky_jitter(double_value=fks_config.CHOLESKY_JITTER_F64):
        _fit_multistart(model, train_x, fks_config.N_RESTARTS, fks_config.MAX_ITER)
    return model
