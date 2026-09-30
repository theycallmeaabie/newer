"""Data schema: canonical long format, units, validation and the checkpoint filter.

Canonical long format, one row per measurement (CONVENTIONS.md rule 4 for units):

    lot_id, part_id, part_number, parameter, unit, hour, value, temperature_c, tester_id, timestamp

`as_of()` is the single place where the leakage rule (CONVENTIONS.md rule 1) is applied to a
frame: every inference-side function receives data through it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Parameter = Literal["IDDQ", "ILEAK", "TPD"]
PARAMETERS: tuple[str, ...] = ("IDDQ", "ILEAK", "TPD")
CANONICAL_UNITS: dict[str, str] = {"IDDQ": "uA", "ILEAK": "uA", "TPD": "ns"}

# Accepted unit spellings -> (canonical unit, multiply-by factor). Ingestion converts with this.
UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    "A": ("uA", 1e6),
    "mA": ("uA", 1e3),
    "uA": ("uA", 1.0),
    "µA": ("uA", 1.0),
    "μA": ("uA", 1.0),
    "nA": ("uA", 1e-3),
    "pA": ("uA", 1e-6),
    "s": ("ns", 1e9),
    "ms": ("ns", 1e6),
    "us": ("ns", 1e3),
    "µs": ("ns", 1e3),
    "μs": ("ns", 1e3),
    "ns": ("ns", 1.0),
    "ps": ("ns", 1e-3),
}

MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "lot_id",
    "part_id",
    "part_number",
    "parameter",
    "unit",
    "hour",
    "value",
    "temperature_c",
    "tester_id",
    "timestamp",
)
KEY_COLUMNS: tuple[str, ...] = ("lot_id", "part_id", "parameter", "hour")


class SchemaError(ValueError):
    """Raised when a frame does not match the canonical schema. The message is inspector-facing."""


class Measurement(BaseModel):
    """One measurement row (used for API payload validation)."""

    model_config = ConfigDict(extra="forbid")

    lot_id: str
    part_id: str
    part_number: str
    parameter: str
    unit: str
    hour: int = Field(ge=0)
    value: float | None = None
    temperature_c: float | None = None
    tester_id: str | None = None
    timestamp: datetime | None = None


def convert_to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Convert `value` in `unit` to the canonical unit for that quantity.

    >>> convert_to_canonical(2.5, "mA")
    (2500.0, 'uA')
    """
    try:
        canonical, factor = UNIT_CONVERSIONS[unit.strip()]
    except KeyError as exc:
        known = ", ".join(sorted(UNIT_CONVERSIONS))
        raise SchemaError(f"unknown unit {unit!r}; known units: {known}") from exc
    return value * factor, canonical


def validate_measurements(
    df: pd.DataFrame,
    *,
    allowed_parameters: tuple[str, ...] | None = PARAMETERS,
    require_canonical_units: bool = True,
) -> pd.DataFrame:
    """Validate and normalise a long-format frame. Returns a new frame with coerced dtypes.

    Raises SchemaError with a plain-language message listing every problem found.
    Extra columns are preserved. Row order is preserved; the index is reset.
    """
    problems: list[str] = []
    missing = [c for c in MEASUREMENT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"missing required column(s): {missing}. Expected columns: {list(MEASUREMENT_COLUMNS)}"
        )

    out = df.copy()
    for col in ("lot_id", "part_id", "part_number", "parameter", "unit", "tester_id"):
        out[col] = out[col].astype("string")

    hour = pd.to_numeric(out["hour"], errors="coerce")
    bad_hour = hour.isna() | (hour < 0) | (hour != np.floor(hour))
    if bad_hour.any():
        problems.append(
            f"{int(bad_hour.sum())} row(s) have a non-integer or negative `hour` "
            f"(first at index {int(np.flatnonzero(bad_hour)[0])})"
        )
    else:
        out["hour"] = hour.astype("int64")

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="coerce")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    if allowed_parameters is not None:
        unknown = sorted(set(out["parameter"].dropna()) - set(allowed_parameters))
        if unknown:
            problems.append(f"unknown parameter(s) {unknown}; allowed: {list(allowed_parameters)}")

    if require_canonical_units:
        expected = out["parameter"].map(CANONICAL_UNITS)
        wrong = expected.notna() & (out["unit"] != expected)
        if wrong.any():
            examples = (
                out.loc[wrong, ["parameter", "unit"]].drop_duplicates().head(5).to_dict("records")
            )
            problems.append(
                f"{int(wrong.sum())} row(s) are not in canonical units {CANONICAL_UNITS}; "
                f"e.g. {examples}. Convert with burnin.ingest before validating."
            )

    dup = out.duplicated(list(KEY_COLUMNS), keep=False)
    if dup.any():
        example = out.loc[dup, list(KEY_COLUMNS)].iloc[0].to_dict()
        problems.append(
            f"{int(dup.sum())} row(s) share the same (lot_id, part_id, parameter, hour); "
            f"e.g. {example}"
        )

    if problems:
        raise SchemaError(
            "measurements do not match the canonical schema:\n- " + "\n- ".join(problems)
        )
    return out.reset_index(drop=True)


def as_of(df: pd.DataFrame, checkpoint_h: int) -> pd.DataFrame:
    """Return only the readings an inference function may see at checkpoint `checkpoint_h`.

    CONVENTIONS.md rule 1: rows with `hour > checkpoint_h` are the future and are dropped.
    Every screening / prediction entry point takes its lot data through this function.
    """
    if checkpoint_h < 0:
        raise ValueError("checkpoint_h must be non-negative")
    return df.loc[df["hour"] <= checkpoint_h].copy()


def assert_as_of(df: pd.DataFrame, checkpoint_h: int) -> None:
    """Callee-side guard: raise if a frame contains readings after the checkpoint."""
    if (df["hour"] > checkpoint_h).any():
        latest = int(df["hour"].max())
        raise ValueError(
            f"leakage: frame contains readings up to {latest}h but checkpoint is {checkpoint_h}h"
        )


def pivot_hours(df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Wide view: index (lot_id, part_id, part_number, parameter), one int column per hour.

    Rows exist only for (part, parameter) pairs present in `df` (a `pivot`, not a `pivot_table`
    with `dropna=False`, which would build the cartesian product of the index levels). Missing
    readings stay NaN. Requires unique keys, which `validate_measurements` guarantees.
    """
    wide = df.pivot(  # noqa: PD010 - pivot_table(dropna=False) would build a cartesian product
        index=["lot_id", "part_id", "part_number", "parameter"], columns="hour", values=column
    )
    wide.columns = [int(c) for c in wide.columns]
    return wide.sort_index()


# --------------------------------------------------------------------------- datasheet limits
class ParameterLimits(BaseModel):
    """Datasheet limits for one parameter of one part number, in canonical units."""

    model_config = ConfigDict(extra="forbid")

    unit: str
    max: float | None = None
    min: float | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> ParameterLimits:
        if self.max is None and self.min is None:
            raise ValueError("a limit needs at least one of `max` or `min`")
        if self.max is not None and self.min is not None and self.min >= self.max:
            raise ValueError("`min` must be below `max`")
        return self


class DatasheetLimits(BaseModel):
    """Per-part-number datasheet limits. This is production knowledge, not a label."""

    model_config = ConfigDict(extra="forbid")

    part_numbers: dict[str, dict[str, ParameterLimits]]

    def get(self, part_number: str, parameter: str) -> ParameterLimits | None:
        return self.part_numbers.get(part_number, {}).get(parameter)

    @classmethod
    def load(cls, path: str | Path) -> DatasheetLimits:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def save(self, path: str | Path) -> None:
        payload = {
            "part_numbers": {
                pn: {p: lim.model_dump(exclude_none=True) for p, lim in params.items()}
                for pn, params in self.part_numbers.items()
            }
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
