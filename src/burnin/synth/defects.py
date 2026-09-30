"""Latent defect trajectory models.

Each defect is a pure function that takes a healthy draft of one part (correlated log baseline
plus healthy drift multipliers) and returns the modified clean log trajectory, the affected
parameters and the *onset hour*: the first reading at which the trajectory differs from the healthy
counterfactual. Readings before the onset are, by construction, indistinguishable from a healthy
part; the evaluation uses `onset_hour` to report early-rejection recall honestly.

Mechanism notes and sources for every shape and magnitude: data/README.md.

Extra defect variants (blind holdout) are plain python files that define
`register(registry: DefectRegistry) -> None` and are listed under `defects.extra_modules`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CATASTROPHIC = "catastrophic"
HEALTHY = "healthy"


@dataclass(frozen=True)
class LotContext:
    """What a defect model may know about the lot the part belongs to."""

    part_number: str
    params: tuple[str, ...]
    hours: np.ndarray  # (H,)
    reference_hour: float
    log_median: np.ndarray  # (P,) lot log medians of the healthy population
    part_sigma_log: np.ndarray  # (P,)
    noise_sigma_log: np.ndarray  # (P,)
    limit_max: np.ndarray  # (P,) datasheet max, nan when none
    corr: np.ndarray  # (P, P) baseline correlation
    # (P, H) log offset the lot's readings will receive after the defect is applied: tester
    # offset plus any whole-lot shift. Defects that promise to stay inside (or outside) the
    # datasheet limit must allow for it; measurement noise is added on top.
    nuisance_log: np.ndarray | None = None

    def index(self, param: str) -> int:
        return self.params.index(param)

    def nuisance(self, p: int) -> np.ndarray:
        """(H,) log offset applied to parameter index `p` after the defect."""
        if self.nuisance_log is None:
            return np.zeros(len(self.hours))
        return self.nuisance_log[p]


@dataclass
class PartDraft:
    """A healthy part before noise and nuisances: baseline (log) and drift multipliers."""

    log_baseline: np.ndarray  # (P,)
    drift_mult: np.ndarray  # (P, H), >= 0.5

    def clean_log_values(self) -> np.ndarray:
        return self.log_baseline[:, None] + np.log(self.drift_mult)


@dataclass
class DefectOutcome:
    """What a defect function returns.

    log_values: (P, H) natural-log clean trajectory after the defect, BEFORE tester offsets,
        lot shifts and measurement noise (the generator adds those).
    defect_type: must equal the name the function was registered under.
    affected: names of the parameter(s) carrying the defect, a subset of `ctx.params`.
    onset_hour: first reading that differs from the healthy counterfactual; one of `ctx.hours`.
    details: free-form numbers for debugging (not written to the labels file).
    """

    log_values: np.ndarray
    defect_type: str
    affected: tuple[str, ...]
    onset_hour: int
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self, name: str, ctx: LotContext) -> None:
        """Reject anything a (plugin) defect function returns that would corrupt the labels."""
        problems = []
        shape = (len(ctx.params), len(ctx.hours))
        if not isinstance(self.log_values, np.ndarray) or self.log_values.shape != shape:
            problems.append(f"log_values must be an array of shape {shape}")
        elif not np.isfinite(self.log_values).all():
            problems.append("log_values contains non-finite numbers")
        if self.defect_type != name:
            problems.append(f"defect_type is {self.defect_type!r}, expected {name!r}")
        if not self.affected or not set(self.affected) <= set(ctx.params):
            problems.append(
                f"affected {self.affected!r} must be a non-empty subset of {ctx.params}"
            )
        elif len(set(self.affected)) != len(self.affected):
            problems.append(f"affected {self.affected!r} repeats a parameter")
        if self.onset_hour not in {int(h) for h in ctx.hours}:
            problems.append(
                f"onset_hour {self.onset_hour!r} is not one of the readings "
                f"{[int(h) for h in ctx.hours]}"
            )
        if problems:
            raise ValueError(f"defect {name!r} returned an invalid outcome: " + "; ".join(problems))


DefectFn = Callable[[PartDraft, LotContext, np.random.Generator, Any], DefectOutcome]


class DefectRegistry:
    """Name -> defect function. The blind team extends it through `extra_modules`."""

    def __init__(self) -> None:
        self._fns: dict[str, DefectFn] = {}

    def add(self, name: str, fn: DefectFn, *, replace: bool = False) -> None:
        if name in (HEALTHY,):
            raise ValueError(f"{name!r} is reserved")
        if name in self._fns and not replace:
            raise ValueError(
                f"defect type {name!r} is already registered; choose another name "
                "(or pass replace=True to override it deliberately)"
            )
        self._fns[name] = fn

    def get(self, name: str) -> DefectFn:
        try:
            return self._fns[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown defect type {name!r}; registered: {sorted(self._fns)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._fns)

    def load_extra_modules(self, paths: list[str]) -> None:
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_file():
                raise ImportError(f"extra defect module not found: {path}")
            spec = importlib.util.spec_from_file_location(f"burnin_extra_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load extra defect module {path}")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise ImportError(f"extra defect module {path} failed to import: {exc}") from exc
            if not hasattr(module, "register"):
                raise ImportError(f"{path} must define register(registry)")
            module.register(self)


# --------------------------------------------------------------------------- helpers
def _weighted_param_order(
    rng: np.random.Generator, ctx: LotContext, weights: dict[str, float]
) -> list[int]:
    """Parameter indices in a random order biased by `weights` (first = chosen parameter)."""
    w = np.array([weights.get(p, 0.0) for p in ctx.params], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(ctx.params))
    order: list[int] = []
    remaining = list(range(len(ctx.params)))
    while remaining:
        probs = w[remaining] / w[remaining].sum()
        pick = rng.choice(remaining, p=probs)
        order.append(int(pick))
        remaining.remove(int(pick))
    return order


def _t_norm(ctx: LotContext) -> np.ndarray:
    return ctx.hours / ctx.reference_hour


# --------------------------------------------------------------------------- defect models
def baseline_outlier(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """Far from the lot median at 0h, but inside the datasheet limit at every reading.

    The offset is `shift_sigma` part-sigmas above the lot median, capped so that every reading
    stays below `max_fraction_of_limit` x limit after drift, the lot's tester offset and any
    whole-lot shift, with a further 3 noise sigmas of margin. If the cap leaves less than
    `min_shift_sigma` of room, the next parameter is tried; if none has room the part gets the
    largest shift the limit allows (the cap always wins over the shift).
    """
    logv = draft.clean_log_values()
    weights = spec.parameter_weights

    def headroom(p: int) -> float:
        if not np.isfinite(ctx.limit_max[p]):
            return np.inf
        worst = (np.log(draft.drift_mult[p]) + ctx.nuisance(p)).max()
        limit = np.log(spec.baseline_outlier.max_fraction_of_limit * ctx.limit_max[p])
        return float(limit - worst - 3.0 * ctx.noise_sigma_log[p])

    chosen = None
    best = None  # best effort if no parameter has min_shift_sigma of room
    for p in _weighted_param_order(rng, ctx, weights):
        s = float(spec.baseline_outlier.shift_sigma.draw(rng))
        target = ctx.log_median[p] + s * ctx.part_sigma_log[p]
        room = headroom(p)
        capped = target > room
        target = min(target, room)
        achieved = (target - ctx.log_median[p]) / ctx.part_sigma_log[p]
        if achieved >= spec.baseline_outlier.min_shift_sigma:
            chosen = (p, target, achieved, capped)
            break
        if best is None or achieved > best[2]:
            best = (p, target, achieved, True)
    if chosen is None:
        chosen = best
    p, target, achieved, capped = chosen
    logv[p, :] += target - draft.log_baseline[p]
    return DefectOutcome(
        logv,
        "baseline_outlier",
        (ctx.params[p],),
        onset_hour=int(ctx.hours[0]),
        details={"shift_sigma": round(float(achieved), 3), "capped_by_limit": capped},
    )


def accelerating_drift(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """Super-linear growth: V(t) = healthy(t) * (1 + g (t/t_ref)^m), m > 1.

    Weak at the first post-0h reading, obvious by t_ref. This is the key case for Module B.
    The 0h reading is the healthy counterfactual; `onset_hour` is the first reading that differs
    from it, however slightly (it does not mean "detectable at that reading").
    """
    p = _weighted_param_order(rng, ctx, spec.parameter_weights)[0]
    g = float(spec.accelerating_drift.growth_for(ctx.params[p]).draw(rng))
    m = float(spec.accelerating_drift.exponent.draw(rng))
    mult = 1.0 + g * _t_norm(ctx) ** m
    logv = draft.clean_log_values()
    logv[p, :] += np.log(mult)
    return DefectOutcome(
        logv,
        "accelerating_drift",
        (ctx.params[p],),
        onset_hour=int(ctx.hours[1]),
        details={
            "growth_at_ref": round(g, 3),
            "exponent": round(m, 3),
            "growth_at_first_reading": round(float(mult[1] - 1.0), 4),
        },
    )


def excess_linear_drift(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """Steady drift several times the lot's typical rate: V(t) = healthy(t) * (1 + b t/t_ref)."""
    p = _weighted_param_order(rng, ctx, spec.parameter_weights)[0]
    b = float(spec.excess_linear_drift.growth_for(ctx.params[p]).draw(rng))
    mult = 1.0 + b * _t_norm(ctx)
    logv = draft.clean_log_values()
    logv[p, :] += np.log(mult)
    return DefectOutcome(
        logv,
        "excess_linear_drift",
        (ctx.params[p],),
        onset_hour=int(ctx.hours[1]),
        details={"growth_at_ref": round(b, 3)},
    )


def step_change(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """A sudden multiplicative jump between two readings.

    With probability `late_onset_fraction` the jump happens after the first post-0h reading, so
    those parts have no 24h signature by construction.
    """
    p = _weighted_param_order(rng, ctx, spec.parameter_weights)[0]
    jump = float(spec.step_change.jump_for(ctx.params[p]).draw(rng))
    late_possible = len(ctx.hours) > 2
    if late_possible and rng.random() < spec.step_change.late_onset_fraction:
        onset_idx = int(rng.integers(2, len(ctx.hours)))
    else:
        onset_idx = 1
    mult = np.where(np.arange(len(ctx.hours)) >= onset_idx, 1.0 + jump, 1.0)
    logv = draft.clean_log_values()
    logv[p, :] += np.log(mult)
    return DefectOutcome(
        logv,
        "step_change",
        (ctx.params[p],),
        onset_hour=int(ctx.hours[onset_idx]),
        details={"jump": round(jump, 3)},
    )


def multivariate(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """Each parameter individually unremarkable, jointly inconsistent.

    Takes the most strongly correlated pair (IDDQ/ILEAK by default: both are leakage-driven), puts
    the part on the consistent low side of both, then lifts one of them to the mildly high side, as
    a defect current that adds to the intrinsic leakage would. Individually both stay within
    +/- 2.5 sigma; the pair is far off the lot's correlation ridge.
    """
    P = len(ctx.params)
    if P < 2:
        raise ValueError("multivariate defect needs at least two parameters")
    corr = np.abs(ctx.corr - np.eye(P))
    if corr.max() <= 0:
        raise ValueError("multivariate defect needs a correlated pair; set `correlation` entries")
    i, j = (int(x) for x in np.unravel_index(int(np.argmax(corr)), corr.shape))
    if rng.random() < 0.5:
        i, j = j, i
    low = float(spec.multivariate.low_side_z.draw(rng))  # negative
    lift = float(spec.multivariate.defect_z.draw(rng))  # positive
    logv = draft.clean_log_values()
    for k, z in ((i, lift), (j, low)):
        new_baseline = ctx.log_median[k] + z * ctx.part_sigma_log[k]
        logv[k, :] += new_baseline - draft.log_baseline[k]
    return DefectOutcome(
        logv,
        "multivariate",
        (ctx.params[i], ctx.params[j]),
        onset_hour=int(ctx.hours[0]),
        details={"lifted": ctx.params[i], "lifted_z": round(lift, 3), "pair_z": round(low, 3)},
    )


def catastrophic(
    draft: PartDraft, ctx: LotContext, rng: np.random.Generator, spec: Any
) -> DefectOutcome:
    """Fails the static datasheet limit outright at every reading (the static screen's baseline).

    A negative tester offset or lot shift is compensated so the measured value still overshoots.
    """
    candidates = [
        k
        for k in _weighted_param_order(rng, ctx, spec.parameter_weights)
        if np.isfinite(ctx.limit_max[k])
    ]
    if not candidates:
        raise ValueError("catastrophic defect needs a parameter with a datasheet max")
    p = candidates[0]
    overshoot = float(spec.catastrophic_overshoot.draw(rng))
    logv = draft.clean_log_values()
    logv[p, :] = (
        np.log(ctx.limit_max[p] * overshoot)
        + np.log(draft.drift_mult[p])
        - np.minimum(ctx.nuisance(p), 0.0)
    )
    return DefectOutcome(
        logv,
        CATASTROPHIC,
        (ctx.params[p],),
        onset_hour=int(ctx.hours[0]),
        details={"overshoot": round(overshoot, 3)},
    )


def default_registry() -> DefectRegistry:
    reg = DefectRegistry()
    reg.add("baseline_outlier", baseline_outlier)
    reg.add("accelerating_drift", accelerating_drift)
    reg.add("excess_linear_drift", excess_linear_drift)
    reg.add("step_change", step_change)
    reg.add("multivariate", multivariate)
    reg.add(CATASTROPHIC, catastrophic)
    return reg
