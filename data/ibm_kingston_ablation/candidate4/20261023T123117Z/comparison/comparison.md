# C4 memory ablation

## 2026 RMSE (claims/day)

| arm | level | n | rmse | bias |
|---|---|---|---|---|
| w1 | ideal | 232 | 5.737 | -1.843 |
| w1 | qpu | 232 | 5.599 | -1.336 |
| scrambled | ideal | 232 | 5.646 | -1.489 |
| scrambled | qpu | 232 | 5.526 | -0.6071 |
| full | ideal | 232 | 5.661 | -1.541 |
| full | qpu | 232 | 5.494 | -0.5919 |

## Noiseless model against the ideal W = 117 forecast

| arm | n | rmse_a | rmse_b | delta_rmse | dm_p | tost_p | margin | verdict | max_abs_pred_diff |
|---|---|---|---|---|---|---|---|---|---|
| w1 | 232 | 5.737 | 5.661 | 0.07649 | 0.000217 | 2.356e-90 | 0.7137 | equivalent within +-0.71 | 0.5782 |
| scrambled | 232 | 5.646 | 5.661 | -0.01465 | 0.001352 | 3.575e-229 | 0.7137 | equivalent within +-0.71 | 0.3675 |

## Paired tests against the full-window runs (TOST margin 0.714)

| arm | reference | n | rmse_a | rmse_b | delta_rmse | dm_p | tost_p | margin | verdict |
|---|---|---|---|---|---|---|---|---|---|
| w1 | this session/full | 232 | 5.599 | 5.494 | 0.1058 | 0.07075 | 1.541e-23 | 0.7137 | equivalent within +-0.71 |
| w1 | 20261009T062717217983Z/hardware-aware | 232 | 5.599 | 6.285 | -0.6856 | 0.007034 | 0.4583 | 0.7137 | different (ablation better) |
| w1 | 20261009T062717217983Z/qiskit | 232 | 5.599 | 7.137 | -1.538 | 7.726e-06 | 0.9893 | 0.7137 | different (ablation better) |
| w1 | 20261009T062717217983Z/random | 232 | 5.599 | 5.476 | 0.1237 | 0.3011 | 1.64e-07 | 0.7137 | equivalent within +-0.71 |
| w1 | 20261009T171726181781Z/hardware-aware | 232 | 5.599 | 5.648 | -0.04892 | 0.7897 | 0.0004004 | 0.7137 | equivalent within +-0.71 |
| w1 | 20261009T171726181781Z/qiskit | 232 | 5.599 | 6.705 | -1.105 | 0.0002926 | 0.8896 | 0.7137 | different (ablation better) |
| w1 | 20261009T171726181781Z/random | 232 | 5.599 | 6.435 | -0.8359 | 0.002501 | 0.6627 | 0.7137 | different (ablation better) |
| w1 | 20261010T232403022032Z/hardware-aware | 232 | 5.599 | 5.571 | 0.0281 | 0.6235 | 6.957e-27 | 0.7137 | equivalent within +-0.71 |
| w1 | 20261010T232403022032Z/qiskit | 232 | 5.599 | 6.189 | -0.5892 | 0.02215 | 0.3241 | 0.7137 | different (ablation better) |
| w1 | 20261010T232403022032Z/random | 232 | 5.599 | 5.524 | 0.07513 | 0.5077 | 3.774e-09 | 0.7137 | equivalent within +-0.71 |
| scrambled | this session/full | 232 | 5.526 | 5.494 | 0.03224 | 0.3063 | 1.854e-59 | 0.7137 | equivalent within +-0.71 |
| scrambled | 20261009T062717217983Z/hardware-aware | 232 | 5.526 | 6.285 | -0.7591 | 0.00103 | 0.5741 | 0.7137 | different (ablation better) |
| scrambled | 20261009T062717217983Z/qiskit | 232 | 5.526 | 7.137 | -1.612 | 1.324e-07 | 0.9977 | 0.7137 | different (ablation better) |
| scrambled | 20261009T062717217983Z/random | 232 | 5.526 | 5.476 | 0.05018 | 0.5184 | 2.055e-17 | 0.7137 | equivalent within +-0.71 |
| scrambled | 20261009T171726181781Z/hardware-aware | 232 | 5.526 | 5.648 | -0.1225 | 0.3824 | 5.09e-05 | 0.7137 | equivalent within +-0.71 |
| scrambled | 20261009T171726181781Z/qiskit | 232 | 5.526 | 6.705 | -1.179 | 1.074e-05 | 0.9521 | 0.7137 | different (ablation better) |
| scrambled | 20261009T171726181781Z/random | 232 | 5.526 | 6.435 | -0.9094 | 0.0001219 | 0.7852 | 0.7137 | different (ablation better) |
| scrambled | 20261010T232403022032Z/hardware-aware | 232 | 5.526 | 5.571 | -0.04543 | 0.1876 | 8.883e-47 | 0.7137 | equivalent within +-0.71 |
| scrambled | 20261010T232403022032Z/qiskit | 232 | 5.526 | 6.189 | -0.6628 | 0.002103 | 0.4113 | 0.7137 | different (ablation better) |
| scrambled | 20261010T232403022032Z/random | 232 | 5.526 | 5.524 | 0.001599 | 0.9818 | 9.816e-19 | 0.7137 | equivalent within +-0.71 |

## Arm RMSE among the stored full-window runs

| arm | group | n_runs | arm_rmse | ref_min | ref_mean | ref_max |
|---|---|---|---|---|---|---|
| w1 | all | 9 | 5.599 | 5.476 | 6.108 | 7.137 |
| w1 | hardware-aware | 3 | 5.599 | 5.571 | 5.835 | 6.285 |
| scrambled | all | 9 | 5.526 | 5.476 | 6.108 | 7.137 |
| scrambled | hardware-aware | 3 | 5.526 | 5.571 | 5.835 | 6.285 |
| full | all | 9 | 5.494 | 5.476 | 6.108 | 7.137 |
| full | hardware-aware | 3 | 5.494 | 5.571 | 5.835 | 6.285 |
