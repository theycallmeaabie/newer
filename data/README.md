# Data

> Everything under `data/synthetic/` is **synthetic**, produced by `burnin.synth` from
> `configs/generator.yaml` with a fixed seed. No real burn-in measurements are included.

## Files

| file | producer | readers |
|---|---|---|
| `synthetic/measurements.csv` | generator | ingest → Module A / B (production path) |
| `synthetic/labels.csv` | generator | `burnin.evaluate` and training code **only** (CONVENTIONS.md, rule 3) |

`measurements.csv` is long format, one row per measurement:

| column | type | notes |
|---|---|---|
| `lot_id` | str | |
| `part_id` | str | serial number, unique within lot |
| `part_number` | str | device type; groups historical lots |
| `parameter` | str | `IDDQ`, `ILEAK`, `TPD` |
| `unit` | str | `uA`, `ns` (canonical units, CONVENTIONS.md rule 4) |
| `hour` | int | 0, 24, 96, 168 |
| `value` | float | may be missing |
| `temperature_c` | float | |
| `tester_id` | str | |
| `timestamp` | datetime | |

`labels.csv`: `lot_id, part_id, is_defective, defect_type`.

## Generator assumptions (Phase 1 fills these in)

Every numeric choice in `configs/generator.yaml` must be listed here with the physical mechanism it
imitates and a source (paper, standard or textbook) for the shape and magnitude. Where no source
has been found, a visible `TODO` stays in place. **Never invent a citation.**

### Healthy population
- Lot baseline: log-normal across lots. TODO(Phase 1): mechanism + source.
- Part baseline within lot: log-normal. TODO(Phase 1): source.
- Drift: `V(t) = V0 · (1 + a · t^n)`, small `a`, `n ≈ 0.3–0.5`. TODO(Phase 1): source for
  sub-linear (saturating) burn-in drift.
- Measurement noise: multiplicative. TODO(Phase 1): typical parametric tester repeatability.
- Correlation between parameters. TODO(Phase 1): source.

### Latent defect types
| label | mechanism imitated | shape | source |
|---|---|---|---|
| `baseline_outlier` | TODO | far from lot median at 0h, inside datasheet limit | TODO |
| `accelerating_drift` | TODO | exponent > 1 or exponential; weak at 24h, obvious at 168h | TODO |
| `excess_linear_drift` | TODO | several × the lot's typical rate | TODO |
| `step_change` | TODO | jump between readings; half occur after 24h by construction | TODO |
| `multivariate` | TODO | each parameter unremarkable, jointly abnormal | TODO |

### Nuisances
- Tester offset per `tester_id`; occasional whole-lot shift (tester drift, not a defect);
  missing readings; a few catastrophic parts failing static limits. TODO(Phase 1): magnitudes.

## Blind holdout

A second generator config, kept **outside the repository** and referenced by the `BLIND_CONFIG`
environment variable, is written by team members who do not work on the models. It uses different
parameter ranges, prevalence and nuisance levels and may add defect variants as a separate module.
Development never reads it; the final evaluation reports blind results separately.
