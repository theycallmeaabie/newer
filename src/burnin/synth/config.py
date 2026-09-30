"""Generator configuration model (pydantic) and loader, including the blind-holdout path.

All keys are validated with `extra="forbid"` so a typo in a blind config fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from burnin.schema import PARAMETERS

BLIND_CONFIG_ENV = "BLIND_CONFIG"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Range(_Model):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.max < self.min:
            raise ValueError(f"range max {self.max} < min {self.min}")
        return self

    def draw(self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None):
        return rng.uniform(self.min, self.max, size=size)


class ParameterSpec(_Model):
    """Healthy population of one parameter for one part number (log-normal)."""

    unit: str
    typical: float = Field(gt=0, description="typical lot median, canonical units")
    limit_max: float | None = Field(
        default=None, gt=0, description="datasheet max, canonical units"
    )
    limit_min: float | None = Field(default=None, gt=0)
    lot_sigma_log: float = Field(ge=0, description="lot-to-lot spread of the lot median (log)")
    part_sigma_log: float = Field(gt=0, description="part-to-part spread within a lot (log)")
    noise_sigma_log: float = Field(ge=0, description="measurement repeatability (log)")


class PartNumberSpec(_Model):
    weight: float = Field(default=1.0, gt=0, description="relative share of lots")
    parameters: dict[str, ParameterSpec]

    @field_validator("parameters")
    @classmethod
    def _known_parameters(cls, v: dict[str, ParameterSpec]) -> dict[str, ParameterSpec]:
        unknown = set(v) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; allowed {list(PARAMETERS)}")
        if not v:
            raise ValueError("a part number needs at least one parameter")
        return v


class TimeSpec(_Model):
    hours: list[int] = Field(min_length=2)
    temperature_c: float = 125.0
    first_lot_start: date = date(2025, 1, 6)
    lot_spacing_days: Range = Range(min=0.0, max=2.0)

    @field_validator("hours")
    @classmethod
    def _sorted_unique_from_zero(cls, v: list[int]) -> list[int]:
        if v != sorted(set(v)) or v[0] != 0:
            raise ValueError("hours must be strictly increasing and start at 0")
        return v


class LotSizeSpec(_Model):
    min: int = Field(ge=2)
    max: int
    distribution: Literal["log_uniform", "uniform"] = "log_uniform"

    @model_validator(mode="after")
    def _ordered(self) -> LotSizeSpec:
        if self.max < self.min:
            raise ValueError("lot size max < min")
        return self


class LotsSpec(_Model):
    count: int = Field(ge=1)
    size: LotSizeSpec


class CorrelationSpec(_Model):
    """Correlation of per-part log baselines. Sign conventions: data/README.md."""

    IDDQ_ILEAK: float = Field(default=0.0, ge=-1, le=1)
    IDDQ_TPD: float = Field(default=0.0, ge=-1, le=1)
    ILEAK_TPD: float = Field(default=0.0, ge=-1, le=1)

    def matrix(self, params: tuple[str, ...]) -> np.ndarray:
        idx = {p: i for i, p in enumerate(params)}
        m = np.eye(len(params))
        for (a, b), rho in (
            (("IDDQ", "ILEAK"), self.IDDQ_ILEAK),
            (("IDDQ", "TPD"), self.IDDQ_TPD),
            (("ILEAK", "TPD"), self.ILEAK_TPD),
        ):
            if a in idx and b in idx:
                m[idx[a], idx[b]] = m[idx[b], idx[a]] = rho
        if np.linalg.eigvalsh(m).min() <= 0:
            raise ValueError("correlation matrix is not positive definite")
        return m


class DriftSpec(_Model):
    """Healthy drift V(t) = V0 * (1 + a * (t / reference_hour)^n); `a` is the fraction at t_ref."""

    a: Range
    n: Range

    @model_validator(mode="after")
    def _sane(self) -> DriftSpec:
        if self.a.min < -0.5 or self.n.min <= 0:
            raise ValueError("healthy drift: a >= -0.5 and n > 0 required")
        return self


class HealthyDriftSpec(_Model):
    reference_hour: float | None = Field(default=None, description="defaults to the last hour")
    lot_sigma_log_a: float = Field(default=0.3, ge=0)
    part_sigma_log_a: float = Field(default=0.3, ge=0)
    parameters: dict[str, DriftSpec]


class TesterSpec(_Model):
    count: int = Field(ge=1)
    offset_sigma_log: float = Field(ge=0)


class WholeLotShiftSpec(_Model):
    probability: float = Field(ge=0, le=1)
    magnitude_log: Range
    kind_weights: dict[Literal["tester_drift", "tester_change"], float] = {
        "tester_drift": 0.5,
        "tester_change": 0.5,
    }


class MissingSpec(_Model):
    cell_fraction: float = Field(ge=0, le=1, description="single readings missing at random")
    part_dropout_fraction: float = Field(
        ge=0, le=1, description="parts with no readings after a hour"
    )


class NuisanceSpec(_Model):
    testers: TesterSpec
    whole_lot_shift: WholeLotShiftSpec
    missing: MissingSpec


class BaselineOutlierSpec(_Model):
    shift_sigma: Range = Field(description="offset from lot median in units of part sigma")
    max_fraction_of_limit: float = Field(default=0.95, gt=0, le=1)
    min_shift_sigma: float = Field(default=3.0, gt=0, description="below this, try another param")


def _check_parameter_keys(v: dict[str, Range]) -> dict[str, Range]:
    unknown = set(v) - set(PARAMETERS)
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)}; allowed {list(PARAMETERS)}")
    return v


class AcceleratingDriftSpec(_Model):
    growth: Range = Field(description="extra fractional growth at reference_hour")
    growth_by_parameter: dict[str, Range] = Field(
        default_factory=dict, description="per-parameter override of `growth` (e.g. TPD)"
    )
    exponent: Range = Field(description="(t/t_ref)^m with m > 1: weak early, obvious late")

    _keys = field_validator("growth_by_parameter")(_check_parameter_keys)

    def growth_for(self, parameter: str) -> Range:
        return self.growth_by_parameter.get(parameter, self.growth)

    @model_validator(mode="after")
    def _superlinear(self) -> AcceleratingDriftSpec:
        if self.exponent.min <= 1.0:
            raise ValueError("accelerating_drift exponent must be > 1")
        return self


class ExcessLinearDriftSpec(_Model):
    growth: Range = Field(description="extra fractional growth at reference_hour, linear in t")
    growth_by_parameter: dict[str, Range] = Field(
        default_factory=dict, description="per-parameter override of `growth` (e.g. TPD)"
    )

    _keys = field_validator("growth_by_parameter")(_check_parameter_keys)

    def growth_for(self, parameter: str) -> Range:
        return self.growth_by_parameter.get(parameter, self.growth)


class StepChangeSpec(_Model):
    jump: Range = Field(description="fractional jump, e.g. 0.5 = x1.5")
    jump_by_parameter: dict[str, Range] = Field(
        default_factory=dict, description="per-parameter override of `jump` (e.g. TPD)"
    )

    _keys = field_validator("jump_by_parameter")(_check_parameter_keys)

    def jump_for(self, parameter: str) -> Range:
        return self.jump_by_parameter.get(parameter, self.jump)

    late_onset_fraction: float = Field(
        ge=0, le=1, description="share whose step happens after the first post-0h reading"
    )


class MultivariateSpec(_Model):
    low_side_z: Range = Field(description="consistent low-side position of the correlated pair")
    defect_z: Range = Field(description="where the defect lifts one of the pair, in sigma")


class DefectsSpec(_Model):
    prevalence: float = Field(ge=0, le=1)
    mix: dict[str, float]
    parameter_weights: dict[str, float]
    catastrophic_fraction: float = Field(ge=0, le=1)
    catastrophic_overshoot: Range = Range(min=1.2, max=3.0)
    baseline_outlier: BaselineOutlierSpec
    accelerating_drift: AcceleratingDriftSpec
    excess_linear_drift: ExcessLinearDriftSpec
    step_change: StepChangeSpec
    multivariate: MultivariateSpec
    extra_modules: list[str] = Field(
        default_factory=list, description="paths to python files registering extra defect types"
    )
    extra: dict[str, dict] = Field(
        default_factory=dict, description="parameters for extra defect types, keyed by name"
    )

    @field_validator("mix")
    @classmethod
    def _mix_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in v.values()):
            raise ValueError("defect mix weights must be non-negative")
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"defect mix must sum to 1, got {sum(v.values()):.6f}")
        return v

    @field_validator("parameter_weights")
    @classmethod
    def _param_weights(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unknown parameter(s) in parameter_weights: {sorted(unknown)}")
        if sum(v.values()) <= 0:
            raise ValueError("parameter_weights must have positive total")
        return v


class GeneratorConfig(_Model):
    seed: int = Field(ge=0)
    output_dir: str = "data/synthetic"
    data_source: Literal["synthetic"] = "synthetic"
    time: TimeSpec
    lots: LotsSpec
    part_numbers: dict[str, PartNumberSpec]
    correlation: CorrelationSpec = CorrelationSpec()
    healthy_drift: HealthyDriftSpec
    defects: DefectsSpec
    nuisances: NuisanceSpec

    @model_validator(mode="after")
    def _cross_checks(self) -> GeneratorConfig:
        if not self.part_numbers:
            raise ValueError("at least one part number is required")
        for pn, spec in self.part_numbers.items():
            for p in spec.parameters:
                if p not in self.healthy_drift.parameters:
                    raise ValueError(f"healthy_drift.parameters lacks {p} (used by {pn})")
        if self.defects.prevalence + self.defects.catastrophic_fraction > 1:
            raise ValueError("prevalence + catastrophic_fraction must be <= 1")
        # Configs that would only fail part-way through generation are rejected here instead.
        if self.defects.catastrophic_fraction > 0:
            for pn, spec in self.part_numbers.items():
                if all(s.limit_max is None for s in spec.parameters.values()):
                    raise ValueError(
                        f"catastrophic_fraction > 0 but part number {pn} has no limit_max"
                    )
        if self.defects.prevalence > 0 and self.defects.mix.get("multivariate", 0) > 0:
            for pn, spec in self.part_numbers.items():
                params = tuple(spec.parameters)
                if len(params) < 2:
                    raise ValueError(
                        f"the multivariate defect needs >= 2 parameters; {pn} has {len(params)}"
                    )
                off_diagonal = np.abs(self.correlation.matrix(params) - np.eye(len(params)))
                if off_diagonal.max() <= 0:
                    raise ValueError(
                        "the multivariate defect needs a correlated pair of parameters; "
                        f"all `correlation` entries for {pn} are 0"
                    )
        return self

    @property
    def reference_hour(self) -> float:
        return self.healthy_drift.reference_hour or float(self.time.hours[-1])

    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_generator_config(
    path: str | Path | None = None, *, blind: bool = False
) -> tuple[GeneratorConfig, Path]:
    """Load a generator config.

    `blind=True` reads the path from the BLIND_CONFIG environment variable. That file is written
    by team members who do not work on the models and must live outside the repository; nothing
    in development reads it (data/README.md, "Blind holdout").
    """
    if blind:
        env = os.environ.get(BLIND_CONFIG_ENV)
        if not env:
            raise FileNotFoundError(
                f"{BLIND_CONFIG_ENV} is not set; point it at the blind generator config"
            )
        path = env
    if path is None:
        path = Path("configs/generator.yaml")
        if not path.exists():  # not run from the repo root: fall back to the source checkout
            path = Path(__file__).resolve().parents[3] / "configs" / "generator.yaml"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"generator config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return GeneratorConfig.model_validate(raw), path


def format_config_error(exc: Exception, path: Path | str | None) -> str:
    """One readable line per problem, for people editing a YAML file (not a traceback)."""
    import difflib

    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return f"config error: {exc}"
    lines = [f"config error in {path}:" if path else "config error:"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(top level)"
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            parent = GeneratorConfig
            known: list[str] = []
            try:
                for part in err["loc"][:-1]:
                    parent = parent.model_fields[part].annotation
                known = list(parent.model_fields)
            except (AttributeError, KeyError, TypeError):
                known = []
            guess = difflib.get_close_matches(str(err["loc"][-1]), known, n=1)
            msg = "unknown key" + (f" (did you mean `{guess[0]}`?)" if guess else "")
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
