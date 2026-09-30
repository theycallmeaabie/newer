"""Schema helpers: unit conversion, validation messages, the checkpoint filter, limits."""

from __future__ import annotations

import pandas as pd
import pytest

from burnin.schema import (
    CANONICAL_UNITS,
    MEASUREMENT_COLUMNS,
    DatasheetLimits,
    ParameterLimits,
    SchemaError,
    as_of,
    assert_as_of,
    convert_to_canonical,
    pivot_hours,
    validate_measurements,
)


def _frame(**overrides) -> pd.DataFrame:
    base = {
        "lot_id": ["L1", "L1", "L1", "L1"],
        "part_id": ["P1", "P1", "P2", "P2"],
        "part_number": ["PN"] * 4,
        "parameter": ["IDDQ"] * 4,
        "unit": ["uA"] * 4,
        "hour": [0, 24, 0, 24],
        "value": [10.0, 10.5, 9.0, None],
        "temperature_c": [125.0] * 4,
        "tester_id": ["T1"] * 4,
        "timestamp": ["2025-01-01 00:00", "2025-01-02 00:00"] * 2,
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (2.5, "mA", (2500.0, "uA")),
        (300, "nA", (0.3, "uA")),
        (1.5, "ps", (0.0015, "ns")),
        (7, "µA", (7.0, "uA")),
        (2, "us", (2000.0, "ns")),
    ],
)
def test_convert_to_canonical(value, unit, expected):
    got = convert_to_canonical(value, unit)
    assert got[1] == expected[1]
    assert got[0] == pytest.approx(expected[0])


def test_convert_unknown_unit_lists_known_units():
    with pytest.raises(SchemaError, match="unknown unit 'furlong'"):
        convert_to_canonical(1.0, "furlong")


def test_validate_ok_coerces_dtypes_and_keeps_extra_columns():
    df = _frame(extra=[1, 2, 3, 4])
    out = validate_measurements(df)
    assert list(out.columns[: len(MEASUREMENT_COLUMNS)]) == list(MEASUREMENT_COLUMNS)
    assert "extra" in out.columns
    assert out["hour"].dtype == "int64"
    assert str(out["timestamp"].dtype).startswith("datetime64")
    assert out["value"].isna().sum() == 1


def test_validate_missing_columns_message():
    with pytest.raises(SchemaError, match=r"missing required column\(s\): \['tester_id'\]"):
        validate_measurements(_frame().drop(columns=["tester_id"]))


def test_validate_reports_every_problem_at_once():
    df = _frame(unit=["mA"] * 4, parameter=["IDDQ", "IDDQ", "FOO", "FOO"])
    df.loc[1, "hour"] = 0  # duplicate (L1, P1, IDDQ, 0)
    with pytest.raises(SchemaError) as exc:
        validate_measurements(df)
    msg = str(exc.value)
    assert "unknown parameter(s) ['FOO']" in msg
    assert "not in canonical units" in msg
    assert "share the same (lot_id, part_id, parameter, hour)" in msg


def test_validate_negative_or_fractional_hour():
    with pytest.raises(SchemaError, match="non-integer or negative `hour`"):
        validate_measurements(_frame(hour=[0, 24, 0, 23.5]))


def test_canonical_units_table_matches_parameters():
    assert set(CANONICAL_UNITS) == {"IDDQ", "ILEAK", "TPD"}


def test_as_of_drops_future_and_does_not_mutate():
    df = validate_measurements(_frame(hour=[0, 168, 0, 24]))
    seen = as_of(df, 24)
    assert set(seen["hour"]) == {0, 24}
    assert len(df) == 4  # untouched
    seen["value"] = 0.0  # a copy, not a view
    assert df["value"].iloc[0] == 10.0
    with pytest.raises(ValueError, match="leakage"):
        assert_as_of(df, 24)
    assert_as_of(seen, 24)
    with pytest.raises(ValueError):
        as_of(df, -1)


def test_pivot_hours():
    wide = pivot_hours(validate_measurements(_frame()))
    assert list(wide.columns) == [0, 24]
    assert wide.shape == (2, 2)
    assert wide.loc[("L1", "P1", "PN", "IDDQ"), 24] == 10.5


def test_limits_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        ParameterLimits(unit="uA")
    with pytest.raises(ValueError, match="below"):
        ParameterLimits(unit="uA", min=5, max=1)
    lim = DatasheetLimits(part_numbers={"PN": {"IDDQ": ParameterLimits(unit="uA", max=50)}})
    lim.save(tmp_path / "limits.yaml")
    back = DatasheetLimits.load(tmp_path / "limits.yaml")
    assert back.get("PN", "IDDQ").max == 50
    assert back.get("PN", "TPD") is None
    assert back.get("OTHER", "IDDQ") is None
