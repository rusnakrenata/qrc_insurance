# Internal review log

Three simulated reviewer passes (A: quantum hardware, B: forecasting/statistics, C: QML/reservoir computing)
and an associate-editor summary were run on the manuscript. Final recommendation after round 3: minor revision.

| # | Concern | Reviewer | Action in manuscript |
|---|---|---|---|
| 1 | 2026 test period used in an earlier exploratory campaign | B, AE | Disclosed in Sec. VI-A and VIII-C; readouts frozen before 2026 scoring; audit_report.md |
| 2 | Random baseline avoids faulty qubits but not faulty couplers | A | Stated in VI-A; all 21 random runs kept; post-hoc split by coupler quality labelled as such (VII-B, VIII-C) |
| 3 | HA benefit over random overstated | A, AE | Claim restricted to reliability (no catastrophic runs), not typical error (12/21, p = 0.66) |
| 4 | Justification of the Y field | C | Memory-capacity numbers from design stage; non-integrability hedged with refs |
| 5 | Bootstrap block length arbitrary | B | 7/14/28-day sensitivity table |
| 6 | "First" claim | AE | Removed; novelty stated as complete problem-to-processor pipeline with layout selection on forecaster distortion |
| 7 | Contribution of each HA stage not separable | A | Stated as limitation |
| 8 | QPU sometimes better than ideal against observations | B | Explained by level offset compensating the ideal model's growth bias; selection uses distortion, not error |
| 9 | Missing QELM / ESN context | C | QELM mention added in related work; ESN included as classical baseline |
| 10 | Fold-local readout correction after design search | B | Added to limitations |
| 11 | Calibration drift between selection and execution | A | 40/63 runs, start spread median 21 min (max 5 h) reported |
| 12 | Resource fairness vs classical models | C | Discussion VIII-A states the cost asymmetry explicitly |
| 13 | Only optimization level 1 for Qiskit | A | Levels 2-3 use the same VF2Layout; stated in VI-A and limitations |

Remaining minor items a real reviewer may still raise: a single dataset, 3-5 rounds per candidate, single IQM batch.
