# Conventions

These rules apply to everything under `src/`, `api/`, `tests/`, `scripts/` and `web/`. They exist
because the judges are reliability/QA engineers: a wrong number in a reason sentence, a leaked
future reading, or an unlabelled synthetic result costs more than any model improvement gains.

Numbered rules are referenced from code comments and tests as `CONVENTIONS.md, rule N`.

---

## 1. The leakage rule (no future information at inference)

**Definition.** A *checkpoint* is a burn-in hour `H` at which a decision is made. When screening a
part at checkpoint `H`, the only data that may influence the result are:

1. the part's own readings with `hour <= H`;
2. readings with `hour <= H` of the **other parts in the same lot**;
3. **historical lots** of the same `part_number`: complete lots (all hours), explicitly passed in
   as `history`, and never including the lot being screened;
4. configuration (`configs/screening.yaml`) and datasheet limits.

Readings at hours `> H` — in particular 96h and 168h when `H = 24` — are **training targets
only**. They never enter a prediction, a z-score, a lot statistic or a decision at checkpoint `H`.

**Enforcement in code.**

- Every inference-side function takes `checkpoint_h: int` and receives data through
  `burnin.schema.as_of(df, checkpoint_h)`, which drops rows with `hour > checkpoint_h`. Callers
  filter; callees may assert (`assert (df.hour <= checkpoint_h).all()`).
- Module B's predictor signature carries only what it is allowed to see:
  `predict(v0, v24, lot_context, prior) -> Prediction`. There is no argument through which a
  later reading can arrive.
- Functions that *may* see targets are named `fit_*` / `train_*` / `calibrate_*` and live next
  to their inference counterparts (`predict_*`, `screen_*`, `decide_*`). A fitted object is
  frozen after fitting; `predict_*` never mutates it.
- Historical lots are passed explicitly (`history: pd.DataFrame`), never discovered implicitly
  from the same frame as the lot under test.

**Test.** `tests/test_no_leakage.py` (Phase 3) takes a test lot, randomly perturbs all readings
with `hour > 24` (including deleting them), and asserts that every 24h output — z-scores, lot
statistics, predictions, intervals, decisions, reason sentences — is **bit-for-bit identical**.
`make test` fails if it is not.

## 2. Lot-grouped splits

All train / validation / test splits are grouped by `lot_id` (`sklearn.model_selection.GroupKFold`
or equivalent). Parts from one lot never appear on both sides of a split: lot-level effects
(baseline, tester offset, whole-lot shift) would otherwise leak and inflate every metric.

## 3. Ground-truth labels are quarantined

`labels.csv` (`lot_id, part_id, is_defective, defect_type`) exists only for synthetic data and
for evaluation. It is **written** by the `burnin/synth/` package (which also defines the defect
types) and **read** by `burnin/evaluate.py` and the training entry points in `burnin/cli.py`; the
labels schema lives in `evaluate.py`, not in `schema.py`. No other module under `src/burnin/` and
nothing under `api/` may reference the file or its columns (case-insensitive);
`tests/test_scaffold.py::test_production_code_never_reads_labels` greps for it. The grep is a
tripwire, not a proof: code review still checks that nothing on the screening path takes a labels
argument. Screening must work on files that have no labels at all — that is the production case.

## 4. Units policy

| quantity | canonical unit | column `unit` | notes |
|---|---|---|---|
| standby current `IDDQ` | microampere | `uA` | |
| leakage current `ILEAK` | microampere | `uA` | |
| propagation delay `TPD` | nanosecond | `ns` | |
| burn-in time `hour` | hour | — | integer; checkpoints 0/24/96/168 but any value is accepted |
| drift rate | canonical unit per hour | e.g. `uA/h` | `r = (V_last - V_0) / (t_last - t_0)`; `t_last = 168 h` by default, taken from the data/config, never hard-coded |
| temperature | degree Celsius | `temperature_c` | |

- **Ingestion converts** to canonical units (`mA`, `nA`, `A` → `uA`; `ps`, `us` → `ns`) and
  records the conversion in the ingest report. A file with no unit column is assumed to be in
  canonical units and the assumption is stated in the ingest report.
- **Statistics for currents are computed in natural-log space** (`configs/screening.yaml`
  `parameters.<P>.log_space`). Non-positive current readings cannot be logged; they are treated as
  missing and given reason code `INVALID_READING`. `TPD` is treated linearly by default.
- **Everything shown to a human is in linear canonical units**: limits, medians, values, margins.
  Robust z-scores are dimensionless and shown as `6.3σ`. Never show a log value in a sentence.
- **Formatting.** Values: 3 significant figures (`45.1 uA`, `0.512 uA`, `12.3 ns`). z-scores:
  1 decimal. Percentages: 1 decimal. Use the ASCII unit string from the table in machine output
  (`uA`); the dashboard may render `µA`.
- Wide-format columns are named `Value_<N>h` on import/export for compatibility with the problem
  statement.

## 5. Robust statistics

- Location: **median**. Scale: **σ_robust = 1.4826 × MAD**. Never `mean ± 3σ` for limits:
  outliers inflate σ and hide themselves.
- Robust z of a value `x`: `z = (x - median) / σ_robust`, computed in the parameter's working space
  (log for currents).
- **MAD = 0 guard** (quantised readings): floor σ_robust at a *relative* spread
  `f = mad_floor_fraction`, applied in the parameter's working space: `ln(1 + f)` for log-space
  parameters (dimensionless, so it does not depend on whether the file was in nA or µA, nor on the
  sign of the log), and `f × |median|` in canonical units for linear parameters. `f` must sit well
  below the smallest natural spread of any parameter (TPD ≈ 4–5 %), and the output records that
  the floor was used.
- **Minimum sample size**: below `robust_stats.min_parts_for_lot_stats`, lot statistics are not
  trusted on their own; shrink toward historical priors and mark the lot *low confidence*.
- **Leave-one-out** (excluding the part under test from its own lot statistics) is decided in
  Phase 2 by measurement on the harness, not assumed: median and MAD already bound a single
  part's influence. Whatever is chosen, the limits *displayed* for a lot are computed from the
  full lot, so one pair of limit lines describes the lot.

## 6. Determinism and offline operation

- Randomness comes only from `numpy.random.default_rng(seed)` with seeds from config. No
  `np.random.seed`, no `random.random()` without a seeded instance (ruff `NPY002` is on).
- LightGBM: `seed`, `deterministic: true`, fixed `num_threads`, `force_row_wise: true`.
- No network access at runtime: no HTTP clients in `src/` or `api/`, no CDN scripts/fonts in
  `web/` or in API-served pages (FastAPI's default `/docs` and `/redoc` are disabled for this
  reason), no telemetry. Dependencies are fetched once by `make setup`, pinned by
  `requirements.lock` and `web/package-lock.json`; `make lock` regenerates the Python lock.
- Runs on a laptop CPU. Nothing requires a GPU.

## 7. Every flag carries a reason

Every non-`PASS` decision returns at least one `Reason(code, sentence, evidence)`:

- `code`: upper snake case, `<WHAT>_<PARAM>_<HOUR>H` where it applies, e.g.
  `DYN_OUTLIER_IDDQ_24H`, `DELTA_OUTLIER_ILEAK_24H`, `DRIFT_EXCEEDS_SLOPE_IDDQ`,
  `MULTIVAR_OUTLIER_24H`, `STATIC_FAIL_TPD_0H`, `LOT_LOW_CONFIDENCE`, `LOT_SHIFT_SUSPECTED`.
- `sentence`: plain English, with the value, the unit, the lot reference and the relevant limit:
  > Iddq@24h = 45.1 uA, 6.3σ above lot median 10.2 uA (n = 212); within datasheet limit 50 uA.
- `evidence`: the numbers used, as a dict, so the dashboard and reports can render them.
- Order: statistical/physical reasons first, ML contributions (SHAP) second and clearly labelled
  as model internals.

## 8. Synthetic data is always labelled as such

Every dataset carries `data_source ∈ {synthetic, uploaded}`. The API exposes it, the dashboard
shows a persistent banner for `synthetic`, reports print it in the header, and README result
tables say "synthetic" in the caption. Never present a synthetic number without the label.

## 9. Configuration over constants

Thresholds (`k` values, safety-slope parameters, recall target, costs, lot-size thresholds) live
in `configs/screening.yaml` and are loaded into a pydantic model once. No magic numbers in
module code; a constant that is not in config needs a comment saying why.

## 10. Code style

- Python ≥ 3.12 (developed on 3.13; the pinned numpy/scipy/shap need 3.12),
  `from __future__ import annotations`, full type hints on public functions. `ruff check` and `ruff format` must pass (`make lint`).
- pandas DataFrames at module boundaries are validated with `burnin.schema` helpers; pydantic
  models for API payloads and config.
- Pure functions for statistics (`numpy` in, `numpy` out); DataFrame plumbing kept separate so
  the maths is testable with five numbers.
- Docstrings say *what the function may see* (checkpoint rule) when it is an inference function.
- Tests: `pytest`, in `tests/`, named `test_<module>.py`. Each numeric algorithm gets a
  hand-computable case. Determinism tests compare full outputs, not summaries.
- `notebooks/` is for EDA only. No core logic; nothing imports from a notebook.

## 11. Naming

- Parameters: `IDDQ`, `ILEAK`, `TPD` (upper case). Decisions: `PASS`, `REVIEW`, `REJECT`.
- Hours in identifiers: `v0`, `v24`, `v96`, `v168`; columns `Value_0h`, … in wide format.
- Config files: `configs/*.yaml`. Generated data: `data/synthetic/`. Blind holdout config:
  path in env var `BLIND_CONFIG`, never committed, never read by anything but the generator CLI
  and the final evaluation.
