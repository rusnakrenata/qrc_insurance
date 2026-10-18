# Where the numbers in the article come from

Paths are relative to the repository root.

| Article | Quantity | Source |
|---|---|---|
| Table II, III-A | splits 1095/365/232, alignment | pipeline/01_check_data.py |
| Table III, III-B | r, MI and permutation p, eta^2 0.237, holiday Delta R^2 0.069 (p 1e-4), exposure Delta R^2 0.047/0.007, ACF, lag Delta R^2, VIF, Cramer's V | results/eda/ (pipeline/02) |
| III-B | F2/F3/F4 gate on Ridge, Poisson, NB2 (2025) | results/eda/feature_set_gate.csv |
| III-B | export columns: correlation with P_t of legal-entity share -0.96, 4+ year share -0.92, vehicle age 0.69; injury and loss-of-profit r 0.02/0.11 | results/eda/raw_column_screening.csv (pipeline/02 on data/raw/qrc_daily_dataset.csv) |
| IV | encoding, Hamiltonian, Trotter order, RWP | qrc/reservoir.py, qrc/circuits.py |
| IV-B | Y field: memory +2 to +43 % for H2-H4, CV change <= 0.07 | results/design/dynamics_y_scan.csv, dynamics_readout_recheck.csv (pipeline/07) |
| V-A | memory qubits, protocol, encoding; F4 chosen by every classical temporal family | results/design/architecture.csv (pipeline/05); configs/classical_selected.json |
| V-B | T2 memory 1.27, no native embedding of T2, embeddings 1072/516/264 | results/design/graphs_*.csv (pipeline/06) |
| V-C | memory retained 97/90/79/2/2 %; r = 3 best in 32/45 branches; process fidelity 0.37 | results/design/graphs_winners.csv, dynamics_trotter_check.csv |
| V-D | Rule-2 definition and score; 33 reservoirs; 17 improved / 16 degraded by noise; 9-point Pareto front | qrc/design.py, results/design/rules_*.csv, noise_*.csv (pipeline/08, 09) |
| V-E, Table V | candidate parameters, CV, 2025 RMSE, CZ, duration; replay windows 117/64 | qrc/candidates.py, configs/fold_local_cv_summary.csv, results/design/freeze_*.csv, results/runs.csv |
| VI-A | policy definitions | qrc/layouts.py, pipeline/13_hardware_round.py |
| VI-A | 40/63 calibration changes; start spread median 21 min | results/runs.csv |
| VI-A | optimization levels 1-3 | results/tables/optimization_levels.csv, data/optimization_levels/ |
| VI-B, Appendix A | classical models: top 3 by CV on 2022-24, one selected on 2025 (Poisson, ESN, GRU, XGB squared not CV rank 1), only it scored on 2026 | configs/classical_selected.json, results/classical_validation/validation_2025.csv, configs/classical_final.json |
| VII-A, Table VI, Fig. 4 | ideal QRC 5.66-6.12, Ridge 5.36, XGBoost 6.34, bootstrap intervals, cRMSE, bias, halves, block-length sensitivity | results/tables/forecasting_comparison.csv, block_sensitivity.csv, results/ideal/ |
| Tables VII, VIII | clipping shares, exposure angles, claim frequency; readout refit through 2025 | results/tables/ideal_encoding_range.csv, ideal_refit_2025.csv |
| VII-A | window ablation of C4 and C5 | results/tables/ideal_window_ablation.csv |
| VII-B, Tables IX-XI, Fig. 5 | all hardware results; HA vs Qiskit 21/21; HA vs random 12/21 | results/runs.csv, results/tables/hardware_*.csv, paired_layouts.csv |
| VII-B | random layouts with CZ error > 4 % (8 of 21) | results/tables/random_by_coupler_flag.csv |
| VII-B | training-date bias of ideal and QPU predictions | data/ibm_kingston_runs/*/*/hardware-aware/probe_daily_features.csv |
| VII-C, Figs. 6, 7 | level-shift shares, contraction, observable offsets | results/runs.csv, results/observables.csv |
| VII-D, Table XII, Fig. 8 | probe, proxy and Aer predictors; shot floor | results/tables/distortion_predictors.csv, ideal_shot_floor.csv |
| VII-E, Table XIII, Fig. 9 | memory curves, MC_ch^{1:20}, K_eff, null thresholds | results/memory/ (analysis/compute_final_candidate_memory.py) |
| VII-F, Table XIV | capacity decomposition: total 8.8-9.9, rank 16-18, current linear 3.5-3.7, current nonlinear 2.2-4.1 (encoding origin stated as "most likely", not tested), past nonlinear 2.0 (C1), ~1.0 (C2, C3), 0.08/0.13 (C4, C5), past linear 2.4 (C5); noise 0.11-0.16 | results/capacity/ (analysis/analyze_nonlinear_capacity.py) |
| VII-G, VII-H | circuit durations, T1/T2, QPU and preparation time | results/runs.csv |
| VII-I | IQM d 8.29/8.34, level shares 0.85/0.88 | results/tables/iqm_probe.csv |
| II, Table I | related work | article/references.bib |
