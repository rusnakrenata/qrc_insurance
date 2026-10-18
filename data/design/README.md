# Design-stage records

Inputs and recorded outputs used by `pipeline/05`-`10`. All were produced on 2022-2025 data only.

| File | Content | Used by |
|---|---|---|
| `chain_couplings.csv` | fixed chain couplings (seed 42) of the architecture study, up to four memory qubits | 05 |
| `heron_coupling_edges.csv` | CZ coupling maps of IBM Fez, Kingston and Marrakesh (September 2026) | 06 |
| `graph_topology_summary.csv` | native-embedding counts and calibration summary per graph and processor | evidence |
| `graph_coupling_winners.csv` | winning coupling vector of each graph H0-H4 (memory capacity search) | 06, 07 |
| `y_field_memory.csv` | memory with and without the Y field (h_y = 0.3, five probes) | evidence |
| `trotter_search_best.csv` | best configuration per (dynamics seed, observable set, r) of the Trotter-depth search | 07, 08 |
| `trotter_depth_comparison.csv` | the same rows with process fidelity, memory and compiled resources | evidence |
| `rule_representatives.csv` | the 33 unique reservoirs selected by rules R1-R4 (parameters, readout, penalty, source) | 08, 09, 10 |
| `rule_map.csv` | which reservoir each rule selected per graph and protocol | evidence |
| `noise_sampled_summary.csv` | calibrated-noise finite-shot results of the 33 reservoirs at five operating points | 09 |
| `noise_direct_best.csv` | best direct (density-matrix) noisy RMSE per reservoir and the ideal RMSE at the same depth | 09 |
