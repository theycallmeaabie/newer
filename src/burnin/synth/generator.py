"""Seeded, configurable synthetic burn-in data generator.

Produces the canonical long-format `measurements.csv`, the quarantined `labels.csv`, the
lot-level ground truth `lot_events.csv` (tester drift / tester change), the datasheet
`limits.yaml`, and a `manifest.json`. Every assumption is documented in data/README.md.

Determinism: one `SeedSequence(seed)`; global draws (tester offsets) come from the master stream,
each lot from its own spawned child. Lot k is therefore identical whatever the lot count, and a
run is bit-for-bit reproducible on the same machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from burnin import __version__
from burnin.schema import MEASUREMENT_COLUMNS, DatasheetLimits, ParameterLimits
from burnin.synth.config import GeneratorConfig
from burnin.synth.defects import (
    CATASTROPHIC,
    HEALTHY,
    DefectRegistry,
    LotContext,
    PartDraft,
    default_registry,
)

GENERATOR_VERSION = "1.1"

LABEL_COLUMNS = (
    "lot_id",
    "part_id",
    "is_defective",
    "defect_type",
    "affected_parameters",
    "onset_hour",
    # Clean (noise-free) |log deviation| from the part's healthy counterfactual, largest over the
    # parameters, at the first post-0h reading and at the last reading. 0 for healthy parts.
    "signal_first_reading_log",
    "signal_final_log",
    # First hour at which a MEASURED value exceeds its datasheet max (-1: never). A latent defect
    # with static_fail_hour >= 0 is caught by the static screen from that reading on.
    "static_fail_hour",
)
LOT_EVENT_COLUMNS = ("lot_id", "hour", "event_type", "parameter", "log_shift")


@dataclass
class SyntheticDataset:
    measurements: pd.DataFrame
    labels: pd.DataFrame
    lot_events: pd.DataFrame
    limits: DatasheetLimits
    manifest: dict

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "measurements": out / "measurements.csv",
            "labels": out / "labels.csv",
            "lot_events": out / "lot_events.csv",
            "limits": out / "limits.yaml",
            "manifest": out / "manifest.json",
        }
        self.measurements.to_csv(paths["measurements"], index=False, float_format="%.6g")
        self.labels.to_csv(paths["labels"], index=False)
        self.lot_events.to_csv(paths["lot_events"], index=False, float_format="%.6g")
        self.limits.save(paths["limits"])
        paths["manifest"].write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        return paths


def _lot_size(rng: np.random.Generator, cfg: GeneratorConfig) -> int:
    lo, hi = cfg.lots.size.min, cfg.lots.size.max
    if cfg.lots.size.distribution == "log_uniform":
        return int(np.rint(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    return int(rng.integers(lo, hi + 1))


def _limits_from_config(cfg: GeneratorConfig) -> DatasheetLimits:
    return DatasheetLimits(
        part_numbers={
            pn: {
                p: ParameterLimits(unit=s.unit, max=s.limit_max, min=s.limit_min)
                for p, s in spec.parameters.items()
                if s.limit_max is not None or s.limit_min is not None
            }
            for pn, spec in cfg.part_numbers.items()
        }
    )


def generate(
    cfg: GeneratorConfig,
    *,
    registry: DefectRegistry | None = None,
    role: str = "development",
) -> SyntheticDataset:
    """Generate a full synthetic dataset from `cfg`. Pure function of the config.

    `role` ("development" or "blind") is recorded in the manifest so reports can keep the two
    apart; it does not change the data.
    """
    reg = registry or default_registry()
    reg.load_extra_modules(cfg.defects.extra_modules)
    for name in cfg.defects.mix:
        reg.get(name)  # fail early on unknown defect types
    unknown_extra = sorted(set(cfg.defects.extra) - set(reg.names()))
    if unknown_extra:
        raise ValueError(
            f"defects.extra has parameters for unregistered defect type(s) {unknown_extra}; "
            f"registered: {reg.names()}"
        )

    hours = np.asarray(cfg.time.hours, dtype=float)
    H = len(hours)
    t_ref = cfg.reference_hour
    all_params = tuple(
        p
        for p in ("IDDQ", "ILEAK", "TPD")
        if any(p in s.parameters for s in cfg.part_numbers.values())
    )

    ss = np.random.SeedSequence(cfg.seed)
    master = np.random.default_rng(ss)
    lot_seeds = ss.spawn(cfg.lots.count)

    # Global nuisance: tester offsets, per tester per parameter (log), fixed for the whole run.
    n_testers = cfg.nuisances.testers.count
    tester_offsets = master.normal(
        0.0, cfg.nuisances.testers.offset_sigma_log, size=(n_testers, len(all_params))
    )
    param_col = {p: k for k, p in enumerate(all_params)}

    pn_names = list(cfg.part_numbers)
    pn_weights = np.array([cfg.part_numbers[pn].weight for pn in pn_names], dtype=float)
    pn_weights /= pn_weights.sum()

    defect_names = list(cfg.defects.mix)
    defect_probs = np.array([cfg.defects.mix[n] for n in defect_names], dtype=float)
    defect_probs /= defect_probs.sum()

    frames: list[pd.DataFrame] = []
    label_rows: list[tuple] = []
    event_rows: list[tuple] = []
    lot_start = pd.Timestamp(cfg.time.first_lot_start)
    counts = {"parts": 0, "rows": 0, "missing_cells": 0, "lots_with_shift": 0}
    type_counts: dict[str, int] = {}

    for k in range(cfg.lots.count):
        rng = np.random.default_rng(lot_seeds[k])
        lot_id = f"L{k + 1:04d}"
        lot_start = (
            lot_start + pd.Timedelta(days=float(cfg.time.lot_spacing_days.draw(rng)))
        ).floor("min")
        pn = pn_names[int(rng.choice(len(pn_names), p=pn_weights))]
        pn_spec = cfg.part_numbers[pn]
        params = tuple(p for p in all_params if p in pn_spec.parameters)
        P = len(params)
        n = _lot_size(rng, cfg)
        tester = int(rng.integers(n_testers))
        off_cols = [param_col[p] for p in params]

        typical = np.array([pn_spec.parameters[p].typical for p in params])
        lot_sigma = np.array([pn_spec.parameters[p].lot_sigma_log for p in params])
        part_sigma = np.array([pn_spec.parameters[p].part_sigma_log for p in params])
        noise_sigma = np.array([pn_spec.parameters[p].noise_sigma_log for p in params])
        limit_max = np.array(
            [pn_spec.parameters[p].limit_max or np.nan for p in params], dtype=float
        )
        corr = cfg.correlation.matrix(params)

        # Lot-level baseline and drift
        log_median = np.log(typical) + rng.normal(0.0, lot_sigma)
        drift_specs = [cfg.healthy_drift.parameters[p] for p in params]
        lot_a = np.array([d.a.draw(rng) for d in drift_specs]) * np.exp(
            rng.normal(0.0, cfg.healthy_drift.lot_sigma_log_a, size=P)
        )

        # Part-level healthy population (correlated log baselines, individual drift)
        z = rng.standard_normal((n, P)) @ np.linalg.cholesky(corr).T
        log_baseline = log_median + z * part_sigma  # (n, P)
        part_a = lot_a * np.exp(rng.normal(0.0, cfg.healthy_drift.part_sigma_log_a, size=(n, P)))
        part_n = np.stack([d.n.draw(rng, size=n) for d in drift_specs], axis=1)  # (n, P)
        t_norm = hours / t_ref
        drift_mult = 1.0 + part_a[:, :, None] * t_norm[None, None, :] ** part_n[:, :, None]
        drift_mult = np.maximum(drift_mult, 0.5)
        clean_log = log_baseline[:, :, None] + np.log(drift_mult)  # (n, P, H)

        # Lot-level nuisances are decided before the defects so that a defect which promises to
        # stay inside the datasheet limit can allow for them: tester offset, and in a few lots a
        # whole-lot shift from some reading onward (tester drift on one parameter, or a change of
        # tester, which shifts every parameter by the difference of the two tester offsets).
        tester_by_hour = np.full(H, tester, dtype=int)
        nuisance_log = np.repeat(tester_offsets[tester, off_cols][:, None], H, axis=1)  # (P, H)
        wls = cfg.nuisances.whole_lot_shift
        if rng.random() < wls.probability and H > 1:
            counts["lots_with_shift"] += 1
            kinds = list(wls.kind_weights)
            kw = np.array([wls.kind_weights[x] for x in kinds], dtype=float)
            kind = kinds[int(rng.choice(len(kinds), p=kw / kw.sum()))]
            h_idx = int(rng.integers(1, H))
            if kind == "tester_change" and n_testers > 1:
                new_tester = int((tester + rng.integers(1, n_testers)) % n_testers)
                delta = tester_offsets[new_tester, off_cols] - tester_offsets[tester, off_cols]
                nuisance_log[:, h_idx:] += delta[:, None]
                tester_by_hour[h_idx:] = new_tester
                for p, d in zip(params, delta, strict=True):
                    event_rows.append((lot_id, int(hours[h_idx]), "tester_change", p, float(d)))
            else:
                p_idx = int(rng.integers(P))
                d = float(wls.magnitude_log.draw(rng)) * (1.0 if rng.random() < 0.5 else -1.0)
                nuisance_log[p_idx, h_idx:] += d
                event_rows.append((lot_id, int(hours[h_idx]), "tester_drift", params[p_idx], d))

        # Defect assignment: one uniform per part; catastrophic first, then latent defects.
        u = rng.random(n)
        cat_mask = u < cfg.defects.catastrophic_fraction
        def_mask = (~cat_mask) & (u < cfg.defects.catastrophic_fraction + cfg.defects.prevalence)
        ctx = LotContext(
            part_number=pn,
            params=params,
            hours=hours,
            reference_hour=t_ref,
            log_median=log_median,
            part_sigma_log=part_sigma,
            noise_sigma_log=noise_sigma,
            limit_max=limit_max,
            corr=corr,
            nuisance_log=nuisance_log,
        )
        part_ids = [f"P{i + 1:04d}" for i in range(n)]
        defect_type = np.full(n, HEALTHY, dtype=object)
        affected = np.full(n, "", dtype=object)
        onset = np.full(n, -1, dtype=int)
        signal_first = np.zeros(n)
        signal_final = np.zeros(n)
        for i in np.flatnonzero(cat_mask | def_mask):
            name = (
                CATASTROPHIC
                if cat_mask[i]
                else defect_names[int(rng.choice(len(defect_names), p=defect_probs))]
            )
            draft = PartDraft(log_baseline=log_baseline[i].copy(), drift_mult=drift_mult[i].copy())
            outcome = reg.get(name)(draft, ctx, rng, cfg.defects)
            outcome.validate(name, ctx)
            deviation = np.abs(outcome.log_values - clean_log[i])  # vs healthy counterfactual
            signal_first[i] = deviation[:, 1].max()
            signal_final[i] = deviation[:, -1].max()
            clean_log[i] = outcome.log_values
            defect_type[i] = name
            affected[i] = ";".join(outcome.affected)
            onset[i] = outcome.onset_hour
            type_counts[name] = type_counts.get(name, 0) + 1

        # Apply the nuisances, then measurement noise, then missing readings.
        clean_log += nuisance_log[None, :, :]
        noisy_log = (
            clean_log + rng.normal(0.0, 1.0, size=clean_log.shape) * noise_sigma[None, :, None]
        )
        values = np.exp(noisy_log)

        miss = cfg.nuisances.missing
        cell_missing = rng.random(values.shape) < miss.cell_fraction
        dropout_parts = np.flatnonzero(rng.random(n) < miss.part_dropout_fraction)
        for i in dropout_parts:
            start_idx = int(rng.integers(1, H))
            cell_missing[i, :, start_idx:] = True
        values = np.where(cell_missing, np.nan, values)
        counts["missing_cells"] += int(cell_missing.sum())

        # First reading at which a measured value is above its datasheet max (NaN never is).
        with np.errstate(invalid="ignore"):
            over = (values > limit_max[None, :, None]).any(axis=1)  # (n, H)
        static_fail = np.where(over.any(axis=1), hours[over.argmax(axis=1)], -1).astype(int)

        # Assemble the long frame for this lot
        part_rep = np.repeat(np.arange(n), P * H)
        param_rep = np.tile(np.repeat(np.arange(P), H), n)
        hour_rep = np.tile(np.arange(H), n * P)
        timestamps = lot_start + pd.to_timedelta(hours[hour_rep], unit="h")
        frames.append(
            pd.DataFrame(
                {
                    "lot_id": lot_id,
                    "part_id": np.array(part_ids, dtype=object)[part_rep],
                    "part_number": pn,
                    "parameter": np.array(params, dtype=object)[param_rep],
                    "unit": np.array([pn_spec.parameters[p].unit for p in params], dtype=object)[
                        param_rep
                    ],
                    "hour": hours[hour_rep].astype(int),
                    "value": values.reshape(-1),
                    "temperature_c": cfg.time.temperature_c,
                    "tester_id": np.array([f"T{t + 1}" for t in tester_by_hour], dtype=object)[
                        hour_rep
                    ],
                    "timestamp": timestamps,
                }
            )
        )
        label_rows.extend(
            (
                lot_id,
                part_ids[i],
                defect_type[i] != HEALTHY,
                defect_type[i],
                affected[i],
                onset[i],
                round(float(signal_first[i]), 5),
                round(float(signal_final[i]), 5),
                static_fail[i],
            )
            for i in range(n)
        )
        counts["parts"] += n
        counts["rows"] += n * P * H

    measurements = pd.concat(frames, ignore_index=True)[list(MEASUREMENT_COLUMNS)]
    labels = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS))
    lot_events = pd.DataFrame(event_rows, columns=list(LOT_EVENT_COLUMNS))
    n_defective = int(labels["is_defective"].sum())
    n_parts = max(counts["parts"], 1)
    is_cat = labels["defect_type"] == CATASTROPHIC
    latent = labels.loc[labels["is_defective"] & ~is_cat]
    latent_static = latent.loc[latent["static_fail_hour"] >= 0]
    manifest = {
        "data_source": "synthetic",
        "dataset_role": role,
        "generator_version": GENERATOR_VERSION,
        "burnin_version": __version__,
        "seed": cfg.seed,
        "config_sha256": cfg.sha256(),
        "hours": cfg.time.hours,
        "lots": cfg.lots.count,
        "parts": counts["parts"],
        "rows": counts["rows"],
        "missing_fraction": counts["missing_cells"] / max(counts["rows"], 1),
        "lots_with_whole_lot_shift": counts["lots_with_shift"],
        "defective_parts": n_defective,
        "realised_prevalence": n_defective / n_parts,  # all non-healthy parts
        "latent_defective_parts": len(latent),
        "latent_prevalence": len(latent) / n_parts,
        "catastrophic_parts": int(is_cat.sum()),
        "catastrophic_fraction": int(is_cat.sum()) / n_parts,
        "defect_type_counts": dict(sorted(type_counts.items())),
        # Latent parts whose measured value crosses a datasheet max at some reading: the static
        # screen catches these too, so recall on "truly latent" parts is reported without them.
        "latent_static_fail_counts": {
            k: int(v) for k, v in sorted(latent_static["defect_type"].value_counts().items())
        },
        "healthy_static_fail_parts": int(
            (~labels["is_defective"] & (labels["static_fail_hour"] >= 0)).sum()
        ),
        "defect_types_registered": reg.names(),
        "extra_modules": [
            {
                "file": Path(m).name,
                "sha256": hashlib.sha256(Path(m).expanduser().read_bytes()).hexdigest(),
            }
            for m in cfg.defects.extra_modules
        ],
        "lot_size_min": int(labels.groupby("lot_id").size().min()),
        "lot_size_max": int(labels.groupby("lot_id").size().max()),
    }
    return SyntheticDataset(measurements, labels, lot_events, _limits_from_config(cfg), manifest)
