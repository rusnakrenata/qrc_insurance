"""Sensitivity of the C4/C5 replay windows to the washout threshold epsilon.

  python analysis/washout_threshold.py

Writes results/design/washout_epsilon.csv.
"""
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import INITIAL_STATES, Reservoir, _zridge, select_lambda  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

spec = importlib.util.spec_from_file_location("freeze", os.path.join(HERE, "..", "pipeline", "10_design_freeze.py"))
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)
EPS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]


def distances(S, F, n=1095):
    """Largest pairwise trace and feature distances between the five initial memory states."""
    vs = [r.reshape(-1) for r in INITIAL_STATES]
    td, fd = [], []
    for t in range(n):
        feats = [(F[t] @ v).real for v in vs]
        vs = [S[t] @ v for v in vs]
        rs = [v.reshape(4, 4) for v in vs]
        td.append(max(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum() for a, b in itertools.combinations(rs, 2)))
        fd.append(max(np.linalg.norm(a - b) for a, b in itertools.combinations(feats, 2)))
    return td, fd


def steps(curve, eps):
    tail = np.maximum.accumulate(np.asarray(curve)[::-1])[::-1]
    return int(np.flatnonzero(tail <= eps)[0]) + 1


def main():
    reps = pd.read_csv(os.path.join(HERE, "..", "data", "design", "rule_representatives.csv")).set_index("candidate_key")
    df = load_data().iloc[:1460].reset_index(drop=True)
    y = df.target_property_damage_claim_count.values.astype(float)
    rows = []
    for k, (key, r) in freeze.chosen_candidates().items():
        rep = reps.loc[key]
        if rep.protocol != "CONT":
            continue
        J = json.loads(rep.J_json)
        edges = GRAPHS[rep.topology]
        res = Reservoir(2, edges, [J[f"J{min(e)}{max(e)}"] for e in edges], rep.hx, rep.hy, rep["dt"], r)
        S, F = res.maps(encode(df, rep.alpha, **training_scales(df)), freeze.FEATURES[rep.test_readout])
        td, fd = distances(S, F)
        Xc = res.cont(S, F)
        lam = select_lambda(Xc[:1095], y[:1095], freeze.LAMBDAS)[0]
        pc = _zridge(Xc[:1095], y[:1095], Xc[1095:], lam)
        for eps in EPS:
            W = max(steps(td, eps), steps(fd, eps))
            ends = np.arange(W - 1, 1460)
            pr = _zridge(Xc[:1095], y[:1095], res.rwp(S, F, ends, W)[ends >= 1095], lam)
            rows.append(dict(candidate=k, key=key, epsilon=eps, trace_steps=steps(td, eps), feature_steps=steps(fd, eps),
                             window=W, cz_per_setting=2 * 5 * r * W,
                             rmse_2025_replay=np.sqrt(np.mean((pr - y[1095:]) ** 2)),
                             rmse_2025_cont=np.sqrt(np.mean((pc - y[1095:]) ** 2)),
                             max_abs_forecast_diff=np.max(np.abs(pr - pc))))
            rows[-1]["replay_minus_cont"] = rows[-1]["rmse_2025_replay"] - rows[-1]["rmse_2025_cont"]
            print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)
    out = os.path.join(HERE, "..", "results", "design", "washout_epsilon.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
