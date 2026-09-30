# Data

> Everything under `data/synthetic/` is **synthetic**, produced by `burnin.synth` from
> `configs/generator.yaml` with a fixed seed. No real burn-in measurements are included. Every
> number reported from it in this repository is labelled as synthetic.

## Files

| file | producer | readers | contents |
|---|---|---|---|
| `synthetic/measurements.csv` | generator | ingest → Module A / B (the production path) | canonical long format, one row per reading |
| `synthetic/labels.csv` | generator | `burnin.evaluate`, training entry points **only** (CONVENTIONS rule 3) | `lot_id, part_id, is_defective, defect_type, affected_parameters, onset_hour` |
| `synthetic/lot_events.csv` | generator | evaluation of lot-level alerts only | whole-lot shifts: `lot_id, hour, event_type, parameter, log_shift` |
| `synthetic/limits.yaml` | generator | production (datasheet limits are not labels) | per part number, per parameter `unit`, `max` |
| `synthetic/manifest.json` | generator | everyone | seed, config hash, counts, realised prevalence, `data_source: synthetic` |
| `synthetic/example_trajectories.png`, `lot_overview.png` | `burnin plot-examples` | humans | healthy vs each defect type; lot-to-lot variation |

`measurements.csv` columns: `lot_id, part_id, part_number, parameter, unit, hour, value,
temperature_c, tester_id, timestamp`. `parameter ∈ {IDDQ, ILEAK, TPD}`, units `uA` / `ns`
(CONVENTIONS rule 4), `hour ∈ {0, 24, 96, 168}` by default, `value` may be missing.

`labels.csv`: `defect_type` is `healthy`, one of the five latent types below, `catastrophic`, or
a name added by the blind team. `onset_hour` is the **first reading at which the trajectory differs
from the healthy counterfactual** (`-1` for healthy parts). Readings before the onset are generated
exactly as for a healthy part, so a defect with `onset_hour > 24` has *no 24 h signature by
construction*; the evaluation reports 24 h recall with and without those parts.

Regenerate with `burnin generate` (≈ 5 s, 600 lots, ~88 k parts, ~1.06 M rows, 67 MB) and
`burnin plot-examples`.

## How a dataset is built

For each lot (independent random stream spawned from the master seed, so lot *k* is identical
whatever the lot count):

1. **Part number** (`PN-A` 60 %, `PN-B` 40 %), **lot size** log-uniform in [20, 500] (mean ≈ 149;
   small lots are common in hi-rel procurement and must be represented), one **tester** of four,
   a **start date** staggered 0–2 days after the previous lot so that "history" is time-ordered.
2. **Lot baseline** per parameter: `log(median) = log(typical) + N(0, lot_sigma_log)`.
3. **Healthy parts**: log baselines `median + z·part_sigma_log`, `z ~ N(0, R)` with the
   correlation matrix `R` below; healthy drift `V(t) = V0 · (1 + a · (t/168)^n)` with `a` drawn per
   lot then jittered per part (log-normal, σ = 0.3), `n ~ U(0.3, 0.5)`.
4. **Defects** (below) replace the clean trajectory of 2 % of parts (+ 0.2 % catastrophic).
5. **Nuisances**: a fixed multiplicative offset per tester and parameter; in 5 % of lots a
   whole-lot shift from some reading onward (tester drift on one parameter, or a tester change that
   shifts all parameters and changes `tester_id`); multiplicative measurement noise; 0.5 % of single
   readings missing at random; 0.3 % of parts with no readings after some hour (removed from the
   board).

All statistics are done in log space for currents and reported in linear units.

## Assumptions and where they come from

Legend: **S** = shape/mechanism supported by the cited source (magnitudes are still ours);
**A** = assumption, no source yet — visible TODO for the team. **We never invent a citation**:
sources listed here are ones we know exist; each carries a verification status.

### Healthy population

| item | value (dev config) | status | basis |
|---|---|---|---|
| Leakage currents are log-normal within a lot | `part_sigma_log` 0.20 (IDDQ), 0.25 (ILEAK) | S (shape), A (magnitude) | Subthreshold leakage is exponential in threshold voltage, which is close to normal across dice, so leakage is close to log-normal [Rao 2004]. The spread values are ours. TODO: fit to a real lot. |
| Propagation delay is close to normal, narrow | `part_sigma_log` 0.04 | A | Delay varies far less than leakage across a lot; treated linearly in screening. TODO: source. |
| Lot-to-lot variation of the lot median | `lot_sigma_log` 0.25–0.30 (currents), 0.05 (TPD) | A | Lot/wafer-level process variation exists and is what motivates per-lot limits (PAT, [AEC-Q001]); magnitudes are ours. |
| IDDQ–ILEAK correlation | ρ = 0.7 | S (sign), A (magnitude) | Both are leakage currents driven by the same threshold-voltage / temperature physics [Rao 2004; Keshavarzi 1997]. |
| IDDQ–TPD correlation | ρ = +0.3 | **A — sign under discussion** | The build prompt asks for "higher Iddq loosely tied to slower delay" (positive). Process-corner physics argues the opposite: a fast corner (low Vth) is both leakier and faster, giving a *negative* correlation [Keshavarzi 1997]. A positive correlation can arise from thermal non-uniformity across a burn-in board (hotter socket → more leakage and slower delay). We follow the prompt and flag it; flipping the sign is a one-line config change and the blind team may do so. TODO: confirm with a reliability engineer. |
| Healthy drift is small and sub-linear: `V(t)=V0(1+a(t/168)^n)` | `a` 1–5 % (IDDQ), 1–6 % (ILEAK), 0–1 % (TPD); `n` 0.3–0.5 | S (shape), A (magnitude) | Wear-out mechanisms follow sub-linear power laws in time (NBTI [Schroder 2003], hot-carrier injection [Hu 1985]); burn-in is intended to sit on the saturating early part of that curve. The fractions are ours; delta criteria in hi-rel screening are of the order of 10–25 % ([MIL-STD-883] Method 5004 references device-spec delta limits), so healthy drift must sit well below that. TODO: real drift magnitudes at 125 °C for the device class. |
| Measurement noise is multiplicative | σ_log 0.02 (IDDQ), 0.03 (ILEAK), 0.01 (TPD) | A | Parametric testers specify repeatability as a percentage of reading; values are ours. TODO: tester datasheet or gauge R&R data. |
| Burn-in at 125 °C, readings at 0/24/96/168 h | | S | Burn-in test conditions and durations are specified in [MIL-STD-883] Method 1015 and [JESD22-A108]; 168 h = 7 days is the problem statement's schedule. |

### Latent defect types

| label | mechanism imitated | shape in the generator | status | basis |
|---|---|---|---|---|
| `baseline_outlier` | Gross but non-fatal defect present from time zero: e.g. a resistive bridging defect adding quiescent current, or a marginal die from the tail of the process distribution | 4–8 σ above the lot median in one parameter at every reading, capped at 95 % of the datasheet limit; `onset_hour = 0` | S (concept), A (magnitude) | Population-relative IDDQ screening ("current signatures", "current ratios", PAT) exists precisely because defective dice often sit inside the absolute limit but outside their population [Gattiker 1996; Maxwell 1999; AEC-Q001]. |
| `accelerating_drift` | Progressive dielectric degradation: stress-induced leakage current growing towards breakdown | multiply by `1 + g (t/168)^m`, `g` 0.3–2.5, `m` 1.5–3: at 24 h only 0.04–0.3 % of the final growth is present, obvious by 168 h; `onset_hour = 24` | S (mechanism), A (shape and magnitude) | SILC is a documented precursor of oxide breakdown [DiMaria 1995]; that it accelerates in the run-up to breakdown is the assumption the super-linear exponent encodes. TODO: source for the time profile. |
| `excess_linear_drift` | Ionic contamination drifting under bias and temperature; also over-fast wear-out | multiply by `1 + b t/168`, `b` 0.3–1.0 (several × the healthy rate); `onset_hour = 24` | S (mechanism), A (magnitude) | Mobile-ion transport in oxides under bias-temperature stress [Snow 1965]. |
| `step_change` | Latent open/short that becomes manifest during stress: crack propagation, intermittent bond, particle-induced short; for delay, an interconnect void (electromigration) raising resistance | ×1.4–×3 from one reading onward; **half of them jump after 24 h** (no early signature) | S (mechanism for TPD), A (leakage steps, timing of onset) | Electromigration-induced resistance rise and abrupt failure [Black 1969]; latent mechanical/ESD defects are catalogued in [JEP122]. TODO: a source for step-like leakage jumps during burn-in. |
| `multivariate` | An added defect current on a die whose intrinsic leakage is low: IDDQ looks ordinary, ILEAK looks ordinary, the *pair* is off the lot's correlation ridge | correlated pair placed at −1 … −2.5 σ, one member lifted to +0.75 … +2 σ; each stays within ±2.5 σ; the joint distance spans borderline to obvious on purpose; `onset_hour = 0` | S (concept), A (magnitude) | This is the rationale of the current-ratio and current-signature methods [Maxwell 1999; Gattiker 1996]. |
| `catastrophic` | Dead-on-arrival / gross defect | 1.2–3 × the datasheet limit at every reading | — | Provides the static-limit baseline; not a latent defect. |

Prevalence 2 % latent + 0.2 % catastrophic (prompt: "default 1–3 %"). Defect mix
25/25/20/20/10 %. Parameter carrying a single-parameter defect: IDDQ 50 %, ILEAK 35 %, TPD 15 %.
All **A**: the team should replace these with lot-history rates where available.

### Nuisances

| item | value | status | basis |
|---|---|---|---|
| Tester offset per tester and parameter | σ_log 0.03 | A | Tester-to-tester correlation offsets are a routine calibration concern; magnitude is ours. |
| Whole-lot shift mid-burn-in | 5 % of lots, 0.08–0.20 log | A | Tester drift / recalibration / tester change between readings. It is *not* a part defect and the system must raise a lot-level alert instead of rejecting the lot. TODO: source or team experience. |
| Missing readings | 0.5 % cells, 0.3 % parts drop out | A | Contact failures, removed parts. |

### Sizing the dataset

A recall claim of ≥ 99 % with zero misses needs ≈ 300 defective parts in the test set (rule of
three: an upper 95 % bound of 3/n on the miss rate [Hanley 1983]; confidence intervals reported
with Wilson's method [Wilson 1927]). With lot-grouped five-fold splits one fold is the test set,
so 600 lots × ≈ 149 parts × 2 % ≈ 1 800 latent defects → ≈ 360 in the test fold.

### Robust statistics used in the generator's own checks

Median and MAD with the 1.4826 consistency factor [Rousseeuw 1993; Leys 2013].

## References

Verification status is recorded by the team; an unverified entry must not be quoted as a source
in the report. (Status set by web check on 2026-09-30 — see the Phase 1 review notes.)

| key | reference | status |
|---|---|---|
| [MIL-STD-883] | US DoD, *MIL-STD-883, Test Method Standard: Microcircuits*, Method 1015 (Burn-in test), Method 5004 (Screening procedures). | TODO verify revision and exact conditions |
| [JESD22-A108] | JEDEC, *JESD22-A108, Temperature, Bias, and Operating Life*. | TODO verify |
| [JEP122] | JEDEC, *JEP122, Failure Mechanisms and Models for Semiconductor Devices*. | TODO verify |
| [AEC-Q001] | Automotive Electronics Council, *AEC-Q001, Guidelines for Part Average Testing*. | TODO verify |
| [Rao 2004] | R. Rao, A. Srivastava, D. Blaauw, D. Sylvester, "Statistical analysis of subthreshold leakage current for VLSI circuits," *IEEE Trans. VLSI Systems*, vol. 12, no. 2, 2004. | TODO verify |
| [Keshavarzi 1997] | A. Keshavarzi, K. Roy, C. F. Hawkins, "Intrinsic leakage in low power deep submicron CMOS ICs," *Proc. Int. Test Conf.*, 1997. | TODO verify |
| [Gattiker 1996] | A. E. Gattiker, W. Maly, "Current signatures," *Proc. IEEE VLSI Test Symp.*, 1996. | TODO verify |
| [Maxwell 1999] | P. Maxwell, P. O'Neill, R. Aitken, R. Dudley, N. Jaarsma, M. Quach, D. Wiseman, "Current ratios: a self-scaling technique for production IDDQ testing," *Proc. Int. Test Conf.*, 1999. | TODO verify |
| [Schroder 2003] | D. K. Schroder, J. A. Babcock, "Negative bias temperature instability: Road to cross in deep submicron silicon semiconductor manufacturing," *J. Appl. Phys.*, vol. 94, no. 1, 2003. | TODO verify |
| [Hu 1985] | C. Hu, S. C. Tam, F.-C. Hsu, P.-K. Ko, T.-Y. Chan, K. W. Terrill, "Hot-electron-induced MOSFET degradation—Model, monitor, and improvement," *IEEE Trans. Electron Devices*, vol. 32, no. 2, 1985. | TODO verify |
| [DiMaria 1995] | D. J. DiMaria, E. Cartier, "Mechanism for stress-induced leakage currents in thin silicon dioxide films," *J. Appl. Phys.*, vol. 78, 1995. | TODO verify |
| [Black 1969] | J. R. Black, "Electromigration—A brief survey and some recent results," *IEEE Trans. Electron Devices*, vol. 16, no. 4, 1969. | TODO verify |
| [Snow 1965] | E. H. Snow, A. S. Grove, B. E. Deal, C. T. Sah, "Ion transport phenomena in insulating films," *J. Appl. Phys.*, vol. 36, 1965. | TODO verify |
| [Rousseeuw 1993] | P. J. Rousseeuw, C. Croux, "Alternatives to the median absolute deviation," *J. Amer. Statist. Assoc.*, vol. 88, no. 424, 1993. | TODO verify |
| [Leys 2013] | C. Leys, C. Ley, O. Klein, P. Bernard, L. Licata, "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median," *J. Exp. Soc. Psychol.*, vol. 49, no. 4, 2013. | TODO verify |
| [Hanley 1983] | J. A. Hanley, A. Lippman-Hand, "If nothing goes wrong, is everything all right? Interpreting zero numerators," *JAMA*, vol. 249, no. 13, 1983. | TODO verify |
| [Wilson 1927] | E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, 1927. | TODO verify |

## Blind holdout

A second generator config, kept **outside the repository** and referenced by the `BLIND_CONFIG`
environment variable, is written by team members who do not work on the models. Development
never reads it; the final evaluation reports blind results separately from the development test
set. This mimics the organisers' hidden ground truth.

How to make one:

1. Copy `configs/generator.yaml` somewhere outside the repo (e.g. `~/blind/generator.yaml`).
2. Change what you like: `seed`, lot count and size range, `typical`/`limit_max` values, drift
   ranges, `prevalence`, the defect `mix` and magnitudes, nuisance levels, even `time.hours`.
   Keys are validated strictly, so a misspelt key is an error, not a silent default.
3. Optionally add defect variants: write a python file that defines `register(registry)` and
   calls `registry.add("<name>", fn)` where `fn(draft, ctx, rng, spec) -> DefectOutcome` (see
   `burnin/synth/defects.py`; `tests/test_generator.py::test_extra_defect_module_is_loaded` has a
   complete example). List the file under `defects.extra_modules` and give `<name>` a share in
   `defects.mix`.
4. Set `output_dir` to a path outside the repo (or `data/blind/`, which is git-ignored) and run
   `BLIND_CONFIG=~/blind/generator.yaml burnin generate --blind`.
5. Do not share the config or the labels with the people tuning the models until the final
   evaluation.
