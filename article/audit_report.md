# Audit notes

## Data used
* IBM Kingston: all 21 complete rounds (63 evaluations, 7-11 October 2026) in data/ibm_kingston_runs.
  Two folders without a complete run are listed in results/runs_excluded.csv: one preparation without
  workers and one round stopped by an IBM API error before execution.
* Classical 2026 predictions: data/classical_predictions (36 runs, one configuration per family selected on 2025).
* Design stage: recorded outputs in data/design; every selection is re-evaluated by pipeline/05-10.
* IQM: the one archived probe batch in data/iqm.

## Reproduction
* Per-run RMSE recomputed from daily predictions (6e-14), ideal features and readouts re-derived
  independently (2e-14, 3e-10), deterministic classical models identical, RNN/GRU within 2e-5,
  IQM metrics equal to the reported values.
* The full design and classical searches can be repeated (--search, --family) but take CPU-days;
  they were re-evaluated at the recorded selections.

## Disclosures
* The 2026 dates were used in an earlier exploratory hardware campaign (HA and Qiskit only). Reservoirs
  and readouts were frozen before any 2026 data were scored (article Sec. VI-A and VIII-C).
* An early exposure regression of the original analysis used all years including 2026, and the
  original screening of the export columns used 2022-2026. Both are recomputed here on training data
  only, and the article quotes the training values.
* Readouts were refitted with fold-local standardization before 2026 scoring; the design search was not
  repeated after this correction.
* The random policy avoided faulty qubits but not faulty couplers; 8 of 21 random layouts contained a
  coupler with reported CZ error above 4 % (4 at 100 %). All runs are kept.
* Calibration changed between layout selection and execution in 40 of 63 runs.
* C4 round 1 HA was stopped after its results were saved and is complete; one C4 Qiskit job was resubmitted
  with the same frozen layout and model.
* Aer noise models exclude readout assignment error.
* The memory audit (results/memory) was made after the 2026 runs and changes nothing in the candidates.
  The Rule-2 selection used only the k <= 20 score.
* The nonlinear capacity analysis (results/capacity) was also made after the 2026 runs; the holiday channel was
  restricted to linear targets after a first run showed spurious cross-term capacity from its rare binary values.
