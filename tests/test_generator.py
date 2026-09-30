"""Synthetic data generator: determinism, prevalence, schema validity, defect construction,
nuisances, blind config and extra defect modules."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from burnin.cli import main
from burnin.schema import CANONICAL_UNITS, MEASUREMENT_COLUMNS, pivot_hours, validate_measurements
from burnin.synth.config import GeneratorConfig, load_generator_config
from burnin.synth.generator import LABEL_COLUMNS, SyntheticDataset, generate


def with_lots(cfg: GeneratorConfig, n: int, **updates) -> GeneratorConfig:
    return cfg.model_copy(update={"lots": cfg.lots.model_copy(update={"count": n}), **updates})


@pytest.fixture(scope="module")
def cfg(repo_root: Path) -> GeneratorConfig:
    return load_generator_config(repo_root / "configs/generator.yaml")[0]


@pytest.fixture(scope="module")
def small(cfg: GeneratorConfig) -> SyntheticDataset:
    return generate(with_lots(cfg, 40))


@pytest.fixture(scope="module")
def medium(cfg: GeneratorConfig) -> SyntheticDataset:
    """~200 lots, ~30k parts: enough for statistical checks, ~2 s."""
    return generate(with_lots(cfg, 200))


def _joined(ds: SyntheticDataset, parameter: str) -> pd.DataFrame:
    wide = pivot_hours(ds.measurements.loc[ds.measurements["parameter"] == parameter])
    return wide.reset_index().merge(ds.labels, on=["lot_id", "part_id"])


def _healthy_lot_medians(joined: pd.DataFrame, hours: list[int]) -> pd.DataFrame:
    return joined.loc[~joined["is_defective"]].groupby("lot_id")[hours].median()


# --------------------------------------------------------------------------- determinism
def test_same_seed_is_bit_identical(cfg):
    a, b = generate(with_lots(cfg, 15)), generate(with_lots(cfg, 15))
    pd.testing.assert_frame_equal(a.measurements, b.measurements)
    pd.testing.assert_frame_equal(a.labels, b.labels)
    pd.testing.assert_frame_equal(a.lot_events, b.lot_events)
    assert a.manifest == b.manifest
    buf_a, buf_b = io.StringIO(), io.StringIO()
    a.measurements.to_csv(buf_a, index=False, float_format="%.6g")
    b.measurements.to_csv(buf_b, index=False, float_format="%.6g")
    assert buf_a.getvalue() == buf_b.getvalue()


def test_different_seed_differs(cfg):
    a = generate(with_lots(cfg, 10))
    b = generate(with_lots(cfg, 10, seed=cfg.seed + 1))
    assert not a.measurements["value"].equals(b.measurements["value"])


def test_first_k_lots_do_not_depend_on_lot_count(cfg):
    a, b = generate(with_lots(cfg, 12)), generate(with_lots(cfg, 30))
    prefix = b.measurements.loc[b.measurements["lot_id"].isin(a.measurements["lot_id"].unique())]
    pd.testing.assert_frame_equal(a.measurements, prefix.reset_index(drop=True))


# --------------------------------------------------------------------------- schema
def test_measurements_match_canonical_schema(small, cfg):
    m = validate_measurements(small.measurements)
    assert list(m.columns) == list(MEASUREMENT_COLUMNS)
    assert set(m["hour"].unique()) == set(cfg.time.hours)
    assert (m["unit"] == m["parameter"].map(CANONICAL_UNITS)).all()
    per_part = m.groupby(["lot_id", "part_id"]).size()
    assert (per_part == 3 * len(cfg.time.hours)).all()
    assert (m["temperature_c"] == cfg.time.temperature_c).all()
    # timestamps advance with hour within a part
    t = m.sort_values(["lot_id", "part_id", "parameter", "hour"])
    dt = t.groupby(["lot_id", "part_id", "parameter"])["timestamp"].diff().dropna()
    assert (dt > pd.Timedelta(0)).all()


def test_labels_cover_every_part_once_and_carry_onset(small):
    parts = small.measurements[["lot_id", "part_id"]].drop_duplicates()
    assert len(small.labels) == len(parts)
    assert small.labels.duplicated(["lot_id", "part_id"]).sum() == 0
    assert list(small.labels.columns) == list(LABEL_COLUMNS)
    healthy = small.labels.loc[~small.labels["is_defective"]]
    assert (healthy["defect_type"] == "healthy").all() and (healthy["onset_hour"] == -1).all()
    defective = small.labels.loc[small.labels["is_defective"]]
    assert (defective["onset_hour"] >= 0).all()
    assert (defective["affected_parameters"] != "").all()


def test_measurements_carry_no_label_information(small):
    assert not set(small.measurements.columns) & set(LABEL_COLUMNS[2:])


def test_manifest_and_limits(small, cfg):
    assert small.manifest["data_source"] == "synthetic"
    assert small.manifest["seed"] == cfg.seed
    assert small.manifest["parts"] == len(small.labels)
    assert (
        small.limits.get("PN-A", "IDDQ").max
        == cfg.part_numbers["PN-A"].parameters["IDDQ"].limit_max
    )


# --------------------------------------------------------------------------- statistics
def test_prevalence_within_binomial_tolerance(medium, cfg):
    n = len(medium.labels)
    latent = medium.labels["defect_type"].isin(cfg.defects.mix).sum()
    cat = (medium.labels["defect_type"] == "catastrophic").sum()
    for count, p in ((latent, cfg.defects.prevalence), (cat, cfg.defects.catastrophic_fraction)):
        tol = 3 * np.sqrt(n * p * (1 - p))
        assert abs(count - n * p) < tol, (count, n * p, tol)


def test_defect_mix_within_tolerance(medium, cfg):
    latent = medium.labels.loc[medium.labels["defect_type"].isin(cfg.defects.mix)]
    counts = latent["defect_type"].value_counts()
    total = len(latent)
    for name, share in cfg.defects.mix.items():
        expected = total * share
        assert abs(counts.get(name, 0) - expected) < 3 * np.sqrt(total * share * (1 - share)) + 1


def test_lot_sizes_span_the_configured_range(medium, cfg):
    sizes = medium.labels.groupby("lot_id").size()
    assert sizes.min() >= cfg.lots.size.min and sizes.max() <= cfg.lots.size.max
    assert (sizes < 50).sum() >= 10, "small lots must be well represented"
    assert sizes.max() > 300


def test_missing_fraction_close_to_config(medium, cfg):
    f = medium.manifest["missing_fraction"]
    miss = cfg.nuisances.missing
    expected = miss.cell_fraction + miss.part_dropout_fraction * 0.5  # dropout removes ~half
    assert 0.5 * expected < f < 2.0 * expected


# --------------------------------------------------------------------------- defect construction
def test_baseline_outlier_far_from_median_but_inside_limit(medium, cfg):
    hours = cfg.time.hours
    checked = 0
    for p in ("IDDQ", "ILEAK", "TPD"):
        j = _joined(medium, p)
        med = _healthy_lot_medians(j, hours)
        rows = j.loc[(j["defect_type"] == "baseline_outlier") & (j["affected_parameters"] == p)]
        for _, r in rows.iterrows():
            limit = medium.limits.get(r["part_number"], p).max
            values = r[hours].to_numpy(dtype=float)
            assert np.nanmax(values) < limit
            sigma = cfg.part_numbers[r["part_number"]].parameters[p].part_sigma_log
            z0 = np.log(values[0] / med.loc[r["lot_id"], 0]) / sigma
            if np.isfinite(z0):
                assert z0 > 2.5, (p, r["lot_id"], r["part_id"], z0)
                checked += 1
    assert checked > 50


def test_catastrophic_exceeds_the_datasheet_limit(small):
    for p in ("IDDQ", "ILEAK", "TPD"):
        j = _joined(small, p)
        rows = j.loc[(j["defect_type"] == "catastrophic") & (j["affected_parameters"] == p)]
        for _, r in rows.iterrows():
            limit = small.limits.get(r["part_number"], p).max
            vals = r[[c for c in j.columns if isinstance(c, int)]].to_numpy(dtype=float)
            assert np.nanmin(vals) > limit


def test_step_change_late_onset_has_no_early_signature(medium, cfg):
    hours = cfg.time.hours
    late_share = []
    for p in ("IDDQ", "ILEAK"):
        j = _joined(medium, p)
        med = _healthy_lot_medians(j, hours)
        rows = j.loc[(j["defect_type"] == "step_change") & (j["affected_parameters"] == p)]
        healthy = j.loc[~j["is_defective"]]
        h_ratio = np.log(healthy[24] / healthy[0]).dropna()
        late = rows.loc[rows["onset_hour"] > 24]
        late_share.append((len(late), len(rows)))
        l_ratio = np.log(late[24] / late[0]).dropna()
        if len(l_ratio) >= 10:
            assert abs(l_ratio.mean() - h_ratio.mean()) < 0.02
        for _, r in late.iterrows():
            k = hours.index(int(r["onset_hour"]))
            prev, at = r[hours[k - 1]], r[hours[k]]
            if np.isfinite(prev) and np.isfinite(at):
                assert at / prev > 1.25
        _ = med
    n_late, n_all = (sum(x) for x in zip(*late_share, strict=True))
    assert abs(n_late / n_all - cfg.defects.step_change.late_onset_fraction) < 0.15


def test_accelerating_drift_is_weak_early_and_obvious_late(medium, cfg):
    hours = cfg.time.hours
    j = _joined(medium, "IDDQ")
    med = _healthy_lot_medians(j, hours)
    rows = j.loc[(j["defect_type"] == "accelerating_drift") & (j["affected_parameters"] == "IDDQ")]
    healthy = j.loc[~j["is_defective"]]
    early_h = np.log(healthy[24] / healthy[0]).dropna()
    early_d = np.log(rows[24] / rows[0]).dropna()
    late_d = np.log(rows[168] / rows[0]).dropna()
    assert len(rows) > 20
    assert early_d.median() - early_h.median() < 0.10  # weakly visible at 24 h
    assert late_d.median() > np.log(1.3)  # obvious by 168 h
    _ = med


def test_multivariate_parts_are_individually_unremarkable(medium, cfg):
    hours = cfg.time.hours
    for p in ("IDDQ", "ILEAK"):
        j = _joined(medium, p)
        med = _healthy_lot_medians(j, hours)
        rows = j.loc[j["defect_type"] == "multivariate"]
        for _, r in rows.iterrows():
            sigma = cfg.part_numbers[r["part_number"]].parameters[p].part_sigma_log
            z0 = np.log(r[0] / med.loc[r["lot_id"], 0]) / sigma
            if np.isfinite(z0):
                assert abs(z0) < 3.0, (p, r["lot_id"], r["part_id"], z0)


# --------------------------------------------------------------------------- nuisances
def test_whole_lot_shift_events_are_visible_in_the_data(medium, cfg):
    hours = cfg.time.hours
    ev = medium.lot_events
    assert len(ev) > 0
    for _, e in ev.iterrows():
        j = _joined(medium, e["parameter"])
        lot = j.loc[(j["lot_id"] == e["lot_id"]) & ~j["is_defective"]]
        k = hours.index(int(e["hour"]))
        jump = np.log(lot[hours[k]] / lot[hours[k - 1]]).median()
        healthy_all = j.loc[~j["is_defective"]]
        base = np.log(healthy_all[hours[k]] / healthy_all[hours[k - 1]]).median()
        assert abs((jump - base) - e["log_shift"]) < 0.05, e.to_dict()
        if e["event_type"] == "tester_change":
            m = medium.measurements
            testers = m.loc[m["lot_id"] == e["lot_id"]].groupby("hour")["tester_id"].first()
            assert testers.loc[hours[k]] != testers.loc[hours[k - 1]]


# --------------------------------------------------------------------------- config & blind holdout
def test_config_rejects_unknown_keys_and_bad_mix(cfg):
    raw = yaml.safe_load(Path("configs/generator.yaml").read_text())
    raw["defects"]["prevalance"] = 0.1  # typo
    with pytest.raises(ValidationError, match="prevalance"):
        GeneratorConfig.model_validate(raw)
    raw = yaml.safe_load(Path("configs/generator.yaml").read_text())
    raw["defects"]["mix"]["step_change"] += 0.1
    with pytest.raises(ValidationError, match="sum to 1"):
        GeneratorConfig.model_validate(raw)


def test_unknown_defect_type_fails_before_generating(cfg):
    bad = with_lots(cfg, 2, defects=cfg.defects.model_copy(update={"mix": {"nope": 1.0}}))
    with pytest.raises(KeyError, match="unknown defect type 'nope'"):
        generate(bad)


def test_blind_config_is_read_from_env(tmp_path, monkeypatch, repo_root):
    raw = yaml.safe_load((repo_root / "configs/generator.yaml").read_text())
    raw["seed"] = 777
    raw["lots"]["count"] = 30
    raw["defects"]["prevalence"] = 0.06
    raw["output_dir"] = str(tmp_path / "blind")
    blind_path = tmp_path / "blind.yaml"
    blind_path.write_text(yaml.safe_dump(raw))

    monkeypatch.delenv("BLIND_CONFIG", raising=False)
    with pytest.raises(FileNotFoundError, match="BLIND_CONFIG"):
        load_generator_config(blind=True)

    monkeypatch.setenv("BLIND_CONFIG", str(blind_path))
    blind_cfg, path = load_generator_config(blind=True)
    assert path == blind_path and blind_cfg.seed == 777
    ds = generate(blind_cfg)
    assert ds.manifest["realised_prevalence"] > 0.04
    assert (
        ds.manifest["config_sha256"]
        != load_generator_config(repo_root / "configs/generator.yaml")[0].sha256()
    )


def test_extra_defect_module_is_loaded(tmp_path, cfg):
    module = tmp_path / "my_defects.py"
    module.write_text(
        "import numpy as np\n"
        "from burnin.synth.defects import DefectOutcome\n"
        "def doubled_iddq(draft, ctx, rng, spec):\n"
        "    logv = draft.clean_log_values()\n"
        "    p = ctx.index('IDDQ')\n"
        "    logv[p, 1:] += np.log(2.0)\n"
        "    return DefectOutcome(logv, 'doubled_iddq', ('IDDQ',), int(ctx.hours[1]), {})\n"
        "def register(registry):\n"
        "    registry.add('doubled_iddq', doubled_iddq)\n"
    )
    defects = cfg.defects.model_copy(
        update={"mix": {"doubled_iddq": 0.5, "step_change": 0.5}, "extra_modules": [str(module)]}
    )
    ds = generate(with_lots(cfg, 20, defects=defects))
    assert (ds.labels["defect_type"] == "doubled_iddq").sum() > 5
    expected = {"healthy", "doubled_iddq", "step_change", "catastrophic"}
    assert set(ds.labels["defect_type"]) <= expected


def test_cli_generate_writes_all_files(tmp_path, capsys):
    assert main(["generate", "--lots", "5", "--out", str(tmp_path)]) == 0
    names = ("measurements.csv", "labels.csv", "lot_events.csv", "limits.yaml", "manifest.json")
    for name in names:
        assert (tmp_path / name).exists(), name
    out = capsys.readouterr().out
    assert "synthetic" in out.lower() and "lots=5" in out
