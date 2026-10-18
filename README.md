# Hardware-Aware Design and Layout Selection in Quantum Reservoir Computing for Insurance Claims Forecasting

Code, data and results for the article *Hardware-Aware Design and Layout Selection in Quantum
Reservoir Computing for Insurance Claims Forecasting*
(submitted to IEEE Transactions on Quantum Engineering).

Two questions are studied:

1. Can a six-qubit gate-based quantum reservoir (QRC) forecast next-day insurance claim counts
   competitively with classical models?
2. Can such a reservoir be designed and placed on a noisy superconducting processor so that its
   forecasts survive the transfer from simulation to hardware?

Everything in the article can be regenerated from this repository. The hardware runs themselves
need IBM Quantum time; their complete records are included, so all hardware analysis runs offline.

## Contents

```
requirements.txt            pinned Python packages
setup_venv.bat, setup_venv.sh   create the virtual environment
data/
  raw/qrc_daily_dataset.csv insurer's daily export, the starting point of the study
  daily_claims.csv          forecasting data set built from it: 1692 forecast origins (2022-01-01 .. 2026-08-19)
  classical_predictions/    2026 predictions of the 36 classical runs used in the article (one configuration per family)
  design/                   recorded inputs and outputs of the reservoir design stages (data/design/README.md)
  ibm_kingston_runs/        complete records of all IBM Kingston rounds (7-11 Oct 2026)
  iqm/                      IQM Emerald probe: reservoir definition and returned counts
  iqm/design/               recorded outputs of the IQM design stages (data/iqm/design/README.md)
  optimization_levels/      transpiler audit of Qiskit levels 1-3 made during the study
configs/
  classical_selected.json   the three cross-validated configurations of every classical family
  classical_final.json      the configuration of each family selected on 2025 (written by pipeline/04)
  fold_local_cv_summary.csv ridge penalties and CV RMSE of the five frozen reservoirs
qrc/                        the quantum reservoir
  candidates.py             the five frozen candidates C1-C5, the graphs H0-H4, data loading
  reservoir.py              input encoding, Trotterized Hamiltonian, density-matrix features (RWP, CONT)
  circuits.py               Qiskit circuits and conversion of counts to features
  readout.py                ridge readout
  layouts.py                native embeddings, strict compilation, HA proxy, random and Qiskit layouts
  noise.py                  storing and loading the calibrated Aer noise model of a round
  design.py                 memory capacity, washout, CV, reservoirs with any number of memory qubits
classical/models.py         Ridge, Poisson and NB2 GLMs, XGBoost, ESN, RNN, GRU
pipeline/                   the study in the order it was carried out (00 .. 13)
analysis/                   results, tables and figures of the article
results/                    outputs of pipeline/ and analysis/ (all numbers quoted in the article)
figures/                    article figures (PDF and PNG)
article/                    LaTeX source, compiled PDF, single-file LaTeX, evidence map, audit notes
```

## Installation

Python 3.11, 3.12 or 3.13 (64-bit). The pinned versions in `requirements.txt` are the ones used
for the article (the hardware rounds ran on Python 3.13 with Qiskit 2.5.2, Aer 0.17.2 and
qiskit-ibm-runtime 0.49.0).

```
setup_venv.bat                 # Windows: creates C:\venvs\qrc_reproducibility
./setup_venv.sh                # macOS / Linux: creates .venv
```

or by hand: `python -m venv .venv`, activate it, `pip install -r requirements.txt`.
Submitting the IQM probe needs `qiskit-iqm` in a separate environment.

## Data

`data/daily_claims.csv` has one row per forecast origin t:

| column | meaning |
|---|---|
| input_date, target_date | day t and day t+1 |
| property_damage_claim_count_t | C_t, property-damage claims reported on day t |
| active_policy_count_t | P_t, active policies on day t |
| day_of_week_t_plus_1 | D_{t+1}, 1 = Monday ... 7 = Sunday |
| is_public_holiday_t_plus_1 | H_{t+1}, Slovak public holiday |
| target_property_damage_claim_count | y_t = C_{t+1} |
| split | train (2022-01-02 .. 2024-12-31, 1095), validation (2025, 365), test (2026-01-01 .. 2026-08-20, 232) |
| C_t_z, P_t_z | z-scores with training mean and population SD |
| weekday_angle, D_sin, D_cos | weekday on the unit circle |

`data/raw/qrc_daily_dataset.csv` is the insurer's daily export from which everything starts: one row
per calendar day from 2021-01-01 to 2026-08-31 with all claim types (total, property damage `maj`,
injury `nem`, loss of profit `usly`), active policies, policy-age counts and shares, legal-entity
policies, mean vehicle age and calendar flags. Claims after 2026-08-20 are incomplete and are not used.
`pipeline/00_prepare_data.py` builds `daily_claims.csv` from it, and `pipeline/02_data_analysis.py`
uses it for the screening of all candidate columns.

`data/ibm_kingston_runs/candidate<k>/<round>/` holds one round of candidate k: the calibration
snapshot (`backend_configuration.json`, `backend_properties.json`), the ideal predictions, and one
folder per layout policy (`hardware-aware/`, `random/`, `qiskit/`) with the selected layout, the
selection files (HA: `proxy.csv`, `simulation_summary.csv`, `probe_*`; random:
`random_layout_selection.json`; Qiskit: `transpiler_layout_trace.json`), the calibrated Aer
results (`noisy_*`), the execution calibration, the job records and the 2026 features and
predictions from the processor (`2026_features_predictions.csv`, `2026_metrics.json`). Two folders
without a complete run are listed in `results/runs_excluded.csv`.

## Reproducing the article

Run from the repository root. Times are wall-clock on a two-core machine. A complete run of the
sequence below was made on a fresh copy of the repository; all outputs, tables, figures and the
article text were identical to the stored ones.

```
python pipeline/00_prepare_data.py                    # data/raw -> data/daily_claims.csv
python pipeline/01_check_data.py                      # data checks
python pipeline/02_data_analysis.py                   # input selection, 5-10 min
python pipeline/03_classical_search.py --verify       # classical CV at the selected configurations, 15 min
python pipeline/04_classical_2026.py --compare data/classical_predictions   # 2025 selection + 36 runs on 2026, 30-60 min
python pipeline/05_design_architecture.py             # 30-60 min
python pipeline/06_design_graphs.py                   # 10 min
python pipeline/07_design_dynamics.py                 # 1-1.5 h
python pipeline/08_design_rules.py                    # 15 min
python pipeline/09_design_noise_screen.py             # seconds
python pipeline/10_design_freeze.py                   # 10 min
python pipeline/11_iqm_probe.py                       # builds the IQM probe circuits
python pipeline/12_ideal_reservoir.py                 # ideal forecasts of C1-C5
python pipeline/14_iqm_design.py                      # IQM: embeddings, H5/H6 winners, five IQM roles, 3 min

python analysis/build_run_table.py                    # results/runs.csv, results/daily.csv
python analysis/observables.py
python analysis/statistics.py                         # results/tables/*.csv
python analysis/distortion_model.py
python analysis/ideal_diagnostics.py                  # 5-15 min
python analysis/optimization_levels.py
python analysis/iqm_probe.py
python analysis/compute_final_candidate_memory.py --check   # 8 min
python analysis/plot_final_candidate_memory.py
python analysis/analyze_nonlinear_capacity.py              # 5 min, needs results/memory
python analysis/plot_nonlinear_capacity.py
python analysis/figures.py
python analysis/tables.py                             # article/tables/*.tex

cd article && pdflatex main && bibtex main && pdflatex main && pdflatex main && python flatten.py
```

Steps 03 and 05-08 re-evaluate the recorded selections by default. The full searches are
available with `--family <name>` (03), `--search` (06, 07, 08) and `--scan` (07); they take
hours to days.

### Where each result of the article comes from

| Article | Script | Output |
|---|---|---|
| Table III (predictor screening), Sec. III-B | pipeline/02 | results/eda/ |
| Table V (candidates) | pipeline/10, pipeline/12 | results/design/freeze_candidates.csv, results/ideal/ |
| Table VI, Fig. 4 (forecasting) | pipeline/04, analysis/statistics.py | results/classical/, results/tables/forecasting_comparison.csv, classical.csv |
| Tables VII, VIII (input range, refit through 2025) | analysis/ideal_diagnostics.py | results/tables/ideal_*.csv |
| Tables IX-XI, Fig. 5 (hardware, pooled, paired) | analysis/build_run_table.py, statistics.py | results/runs.csv, results/tables/hardware_*.csv, paired_layouts.csv |
| Figs. 6, 7 (reproducibility, mechanism) | analysis/observables.py | results/observables.csv |
| Table XII, Fig. 8 (pre-deployment predictors) | analysis/distortion_model.py | results/tables/distortion_predictors.csv |
| Table XIV (nonlinear capacity of the frozen candidates) | analysis/analyze_nonlinear_capacity.py | results/capacity/ |
| Table XIII, Fig. 9 (memory of the frozen candidates) | analysis/compute_final_candidate_memory.py, plot_final_candidate_memory.py | results/memory/ |
| Sec. VI-A (optimization levels) | analysis/optimization_levels.py | results/tables/optimization_levels.csv |
| Sec. VII-I (IQM) | pipeline/14_iqm_design.py, analysis/iqm_probe.py | results/iqm/, results/tables/iqm_probe.csv |
| Appendix A, Table XV (classical configurations, 2025 selection) | pipeline/03, pipeline/04 | configs/classical_selected.json, results/classical_validation/validation_2025.csv, configs/classical_final.json |
| Appendix B, Table XVI (all runs) | analysis/tables.py | article/tables/tab_allruns.tex |

`article/evidence_map.md` lists every number of the text with its source file.

## The stages

**00 Data.** Pairs day t (claims, policies) with the calendar and target of day t+1, splits by
target date and standardizes with training statistics.

**02 Input selection.** Training data only. Leakage audit, target overdispersion, autocorrelation,
weekday ANOVA, holiday effect beyond the weekday (permutation of holiday labels within weekdays,
10^4 shuffles), exposure against a time trend, value of older claim lags, mutual information with
500 target permutations, VIF and Cramer's V, stability across years. Then the F2/F3/F4 gate:
Ridge, Poisson and NB2 fitted on 2022-2024 and scored on 2025. F4 = (C_t, D_{t+1}, P_t, H_{t+1}).

**03-04 Classical models.** Three steps for every family.
1. 03: five expanding-window folds on 2022-2024 (scikit-learn TimeSeriesSplit) rank the
   configurations (Poisson and NB2: seven predefined exposure specifications); the three best are
   kept in `configs/classical_selected.json`.
2. 04: each of the three is fitted on 2022-2024 and scored on 2025 (ESN, RNN and GRU: mean over
   ten seeds); the lowest 2025 RMSE wins (`results/classical_validation/validation_2025.csv`,
   `configs/classical_final.json`). The three naive forecasts are compared on 2025 directly.
3. 04: only the winner, still fitted on 2022-2024, forecasts 2026 (`results/classical/`).

**05-10 Reservoir design** (noiseless unless stated, 2022-2025 only).
05 memory size (1-4 qubits), continuous (CONT) vs reset-window replay (RWP), exposure encoding.
06 memory-optimal seven-edge graph T2, native embeddings on IBM Fez, Kingston and Marrakesh, five
heavy-hex trees H0-H4 and an equal-budget coupling search on each.
07 Y field on the memory qubits, observable sets, Trotter depth r = 1-3.
08 replay windows W = 1-28, local refinement and the selection rules: R1 lowest CV RMSE,
R2 largest delayed-input memory, R3 shot robustness at 1024 shots, R4 cheapest circuit within one
standard error of the best CV.
09 calibrated-noise screening of the 33 rule representatives and the Pareto front.
10 the five candidates, replay windows for the continuous designs (C4: 117, C5: 64) and readouts
refitted with fold-local standardization, then frozen.

**11, 14 IQM Emerald** (54 qubits, square lattice). The same design procedure on a second platform.
14 counts native embeddings of H0-H6 on the calibrated CZ couplers of Emerald, re-evaluates the
memory winners of the two seven-edge graphs native to it (H5, H6; `--search` repeats the 60-candidate
coupling search) and rebuilds the five IQM roles chosen by R1-R4: two H6 replay reservoirs and three
H2 designs shared with the IBM study. Their readouts are refitted on 2022-2024 with the recorded
penalty (CV identity check), scored on 2025, and forecast 2026 in ideal simulation only. The
Y-field, Trotter-depth, replay and rule stages, R5 and the IQM calibrated-simulator layout screen
are kept as recorded outputs in `data/iqm/design/`. 11 builds the probe of role 1 on Emerald
(two layouts, 25 training dates, 512 shots); `analysis/iqm_probe.py` turns the returned counts into
forecasts. No IQM 2026 hardware run exists.

**12 Ideal reservoir.** Features, readouts and 2025/2026 forecasts of the frozen candidates.

**13 Hardware round.** For one candidate and one policy: store the calibration and noise model
(shared by the three policies of a round), choose the layout, simulate with the calibrated noise
model and run the 232 test days on the processor (2048 shots, two measurement settings,
optimization level 1, routing disabled).
- HA: all native embeddings avoiding faulty qubits, workload error proxy L, calibrated Aer on
  20 training dates for the six best, QPU probe on 100 training dates for the best two, the
  layout with the smaller probe distortion is used.
- Random: uniform draw among native embeddings avoiding faulty qubits.
- Qiskit: initial layout from the level-1 transpiler (VF2Layout, or TrivialLayout when it fits).

`--dry-run --limit 6` runs the same code on a simulated heavy-hex device without an account.

## Memory of the frozen candidates

`analysis/compute_final_candidate_memory.py` is a post-hoc description of the five finished
candidates. It changes no parameter, readout, layout or 2026 result.

- Probe: the four encoded input channels of 2022-2025 are permuted independently with seeds
  79001, 42, 101, 505, 707 (the seeds of Rule R2). Same seeds for every candidate.
- Features: the 18 one-body Pauli expectations after RWP replay with the candidate's frozen W
  (4, 2, 2, 117, 64). The features at endpoint e use inputs e-W+1 .. e, so delays k <= W-1 lie
  inside the window.
- Readout: standardized ridge (alpha 1e-6) trained on endpoints in 2022-2024, evaluated on the
  365 endpoints of 2025. MC_{j,k} = max(Corr^2 - 1/(365-1), 0).
- Delays k = 1..120 (the longest window is 117).
- MC_ch^{1:20}: the Rule-2 score (sum over k <= 20, mean over channels and seeds). The design
  selection used only this truncated score; k = 20 was a fixed diagnostic horizon. The extended
  range is a later characterization of the finalists.
- Effective horizon K_eff, fixed before the curves were computed: the last delay of the first run
  k = 1, 2, ... where the seed-mean MC_k exceeds the 99th percentile of the same statistic with
  permuted probe targets (20 permutations at 12 delays). Also reported: 95th percentile and a
  fixed 0.01.
- `--check` first recomputes the recorded Rule-2 value of RWP_H4_R2 (1.1634, SD 0.0861).

Outputs in `results/memory/`: `memory_by_seed.csv` (candidate, seed, channel, delay, raw and
corrected MC), `memory_curve.csv` (seed mean and SD per delay), `memory_null.csv`,
`memory_summary.csv` (MC_ch^{1:20}, K_eff, thresholds, configuration hash) and
`memory_vs_forecast.csv` (with 2026 ideal and hardware results and circuit duration).
`analysis/plot_final_candidate_memory.py` draws Fig. 9 from these files only.

## Nonlinear capacity of the frozen candidates

`analysis/analyze_nonlinear_capacity.py` decomposes the capacity of the five candidates in the sense
of Dambre et al. (Sci. Rep. 2, 514, 2012). It is post-hoc and changes nothing in the candidates.

- Same probe, seeds, replay windows, 18 one-body features, ridge readout (alpha 1e-6) and split
  (train 2022-2024 endpoints, evaluate 2025) as the memory audit.
- Basis: for each channel and seed, polynomials phi_1, phi_2, phi_3 orthonormal on the empirical
  distribution of the probe training rows (Gram-Schmidt on 1, u, u^2, u^3). The probe is not uniform,
  so Legendre polynomials would not be orthogonal.
- Targets at delays k in {0, 1, 2, 3, 5, 10, 18, 20, 25}: phi_1(u_j(t-k)) (linear), phi_2 and phi_3
  (single-delay nonlinear) and phi_1(u_i(t-k1)) phi_1(u_j(t-k2)) for all pairs of channel-delay items
  (cross). The binary holiday channel enters linear targets only. 441 targets per candidate and seed.
- Capacity = squared correlation on 2025, counted if above the 99th percentile of a permutation null
  (300 random targets per candidate) and then reduced by 1/(365-1).
- Outputs in `results/capacity/`: `capacity_by_target.csv/.json` (candidate, seed, family, degree,
  channels, delays, raw and corrected capacity), `capacity_summary.csv/.json` (family sums, current
  versus past inputs, feature rank, MC_ch^{1:20}, K_eff, 2026 ideal and HA results),
  `capacity_null.csv` and `capacity_metadata.json` (basis, targets, split, correction).
- `analysis/plot_nonlinear_capacity.py` draws a supplementary figure (`figures/fig_capacity`, not in the article) from these files only.

## New hardware rounds

```
python pipeline/13_hardware_round.py --candidate 2 --policy ha     --round runs/c2_r6 --token $IBM_TOKEN
python pipeline/13_hardware_round.py --candidate 2 --policy random --round runs/c2_r6 --token $IBM_TOKEN
python pipeline/13_hardware_round.py --candidate 2 --policy qiskit --round runs/c2_r6 --token $IBM_TOKEN
```

Start the three commands at the same time. A full round of C1-C3 takes about 6 minutes of QPU
time per policy plus 30 s for the HA probe; C4 and C5 are split into three jobs.

## Checks made when the repository was assembled

| Stage | Check | Result |
|---|---|---|
| 00 | daily_claims.csv rebuilt from the insurer's export | identical |
| 02 | MI, permutation p-values, VIF, Cramer's V, ANOVA (eta^2 0.237), holiday Delta R^2 0.069 (p 1e-4), exposure Delta R^2 0.047/0.007 | equal to the original analysis |
| 02 | export-column screen | original correlations -0.972/-0.927 reproduced on 2022-2026; training-only -0.962/-0.916 are used in the article |
| 02 | F2/F3/F4 gate (Ridge, Poisson, NB2 on 2025) | Poisson and NB2 equal to the original gate; Ridge within 0.03 after the fold-local scaling fix |
| 03 | CV RMSE of the 20 selected classical configurations | 1e-14 (deterministic), 5e-8 (RNN, GRU) |
| 04 | 2025 scores of the 27 cross-validated configurations, 2026 forecasts of the 36 selected runs | 1e-5 against the stored predictions |
| 05 | CV and 2025 RMSE of 32 memory-size/protocol settings | equal |
| 06 | MC of T2 (1.2705) and of the coupling-search winners; embeddings 1072/516/264/1072/516, none for T2 | equal |
| 07 | Y-field memory and washout, readout recheck, 135 Trotter-search rows | equal |
| 08 | CV, 2025 RMSE and shot metrics of the 33 rule representatives; rule properties | 1e-13; all pass |
| 09 | Pareto front (9 points); 17 improved / 16 degraded under noise | equal |
| 10 | replay windows 117 and 64; fold-local penalties and CV | equal |
| 11 | IQM probe circuits vs ideal replay features | 2e-16 |
| 14 | native embeddings of H0-H6 on Emerald (5974/4162/4662/5974/4162/188/188); MC of H5 and H6 winners (0.7451, 1.1697), washout 139 and 19 | equal; `--search` reproduces both winners (about 15 min) |
| 14 | CV RMSE of the five IQM roles; 2025 RMSE of roles 1-3 | equal; role 4: 4.804 against a recorded 4.815 |
| 12 | features and readout weights vs the stored ideal runs | 2e-14; 3e-10 |
| 13 | `--dry-run` for all three policies | runs end to end |
| analysis | per-run RMSE from the raw IBM records | 6e-14 |
| memory | Rule-2 value of RWP_H4_R2 | equal (1.1634) |
| capacity | targets a short window cannot reach (noise check) | 0.11-0.16 of 8.8-9.9 total |
| optimization levels | selectors and qubits vs the original transpiler audit | equal |

## Notes

- 2026 was used only for scoring. The same dates had been run in an earlier exploratory hardware
  campaign (HA and Qiskit only); that campaign is not part of these results (article, Sec. VI-A).
- The calibration snapshot of the September 2026 noise screening (stage 09) was not archived;
  stage 09 analyses the recorded screening and can repeat the simulation on any stored calibration.
- RNN and GRU reproduce to about 1e-5; bit-exact results need the same PyTorch build.
- One C4 Qiskit job was resubmitted with the same frozen layout and model after the original
  account became unavailable (`replaced_jobs/` in that run folder).
