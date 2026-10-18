# IQM Emerald design records

Outputs of the IQM design stages, produced on 2022-2025 data only (no 2026 data). Used by
`pipeline/14_iqm_design.py`; the stages marked "recorded" are not recomputed.

| File | Content |
|---|---|
| `emerald_coupling_edges.csv` | calibrated CZ couplers of IQM Emerald (calibration set of 28 Sep 2026) |
| `iqm_region_summary.csv` | native embeddings and calibration summary of H0-H6 on Emerald and Garnet |
| `graph_coupling_winners.csv` | memory-optimal couplings of H5 and H6 (32 global + 28 local candidates each) |
| `graph_coupling_finalists.csv` | five memory finalists per graph with their forecast CV (recorded) |
| `y_field_summary.csv` | memory and CV with and without the Y field on the memory qubits (recorded) |
| `trotter_search_best.csv`, `trotter_search_winner.csv` | best configuration per branch and depth r = 1-3 (recorded) |
| `rule_candidates_rwp.csv` | replay reservoirs on H5/H6 selected by R1-R4, with parameters (recorded) |
| `rule2_memory_winners_rwp.csv` | R2 winners of the replay search (recorded) |
| `rule_candidates_cont.csv` | continuous H5/H6 reservoirs selected by R1-R4 (recorded) |
| `rule_candidates_all.csv` | R1-R4 candidates of H0-H6 under both protocols (recorded) |
| `rule5_layouts.csv` | three best Emerald layouts per candidate under R5 (recorded) |
| `candidate_pool.csv` | the 47 candidates considered for Emerald (recorded) |
| `final_five.csv`, `final_five_audit.csv` | the five IQM roles and how they were chosen (recorded) |
| `sim_layout_screen.csv`, `sim_layout_screen_daily.csv` | IQM calibrated-simulator screen of role 1: 12 layouts, 20 training dates, 64 shots (recorded) |
| `probe_layouts.csv`, `probe_endpoints.csv` | the two layouts and the training dates sent to the Emerald probe |
