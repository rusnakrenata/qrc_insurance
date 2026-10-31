"""Design stage 6: the five candidates, replay windows of C4/C5 and frozen readouts.

  python pipeline/10_design_freeze.py

Writes results/design/freeze_*.csv.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import INITIAL_STATES, Reservoir, _zridge, select_lambda  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "design")
OUT = os.path.join(HERE, "..", "results", "design")
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0]
FEATURES = {"XZ_injection": ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3"],
            "XZinj_dropX3": ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"],
            "XZinj_dropX3_plus_YX45": ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3", "YX45"],
            "XZinj_plus_YX45": ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3", "YX45"]}


def chosen_candidates():
    """Reservoirs with a Pareto-nondominated point (RMSE, CZ, shots), numbered by RMSE."""
    s = pd.read_csv(os.path.join(DATA, "noise_sampled_summary.csv"))
    v = s[["rmse_mean", "feature_vector_n_cz", "shots_per_setting"]].to_numpy(float)
    front = s[[not (np.all(v <= x, axis=1) & np.any(v < x, axis=1)).any() for x in v]]
    best = front.sort_values("rmse_mean").groupby("candidate_key", sort=False).head(1)
    return {k: (row.candidate_key, int(row.r)) for k, row in enumerate(best.itertuples(), 1)}


def washout_steps(res, S, F, eps=0.01):
    """Steps after which trace and feature distances between initial memory states stay below eps."""
    vs = [r.reshape(-1) for r in INITIAL_STATES]
    td, fd = [], []
    for t in range(len(S)):
        feats = [(F[t] @ v).real for v in vs]
        vs = [S[t] @ v for v in vs]
        rs = [v.reshape(4, 4) for v in vs]
        td.append(max(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum() for a, b in itertools.combinations(rs, 2)))
        fd.append(max(np.linalg.norm(a - b) for a, b in itertools.combinations(feats, 2)))

    def steps(curve):
        tail = np.maximum.accumulate(np.asarray(curve)[::-1])[::-1]
        return int(np.flatnonzero(tail <= eps)[0]) + 1
    return steps(td), steps(fd)


def fold_local_lambda(df, rep, edges, J, r, W, names, y):
    """Ridge penalty by chronological CV with input scaling refitted in each fold."""
    from sklearn.model_selection import TimeSeriesSplit
    ends = np.arange(W - 1, 1095) if rep.protocol == "RWP" else np.arange(1095)
    res = Reservoir(2, edges, J, rep.hx, rep.hy, rep["dt"], r)
    folds = []
    for tr, va in TimeSeriesSplit(5).split(ends):
        last = ends[tr][-1]
        theta = encode(df, rep.alpha, **training_scales(df, n_train=last + 1))
        S, F = res.maps(theta[:1095], names)
        X = res.rwp(S, F, ends, W) if rep.protocol == "RWP" else res.cont(S, F)[: len(ends)]
        folds.append((X[tr], y[ends[tr]], X[va], y[ends[va]]))
    scores = {lam: np.mean([np.sqrt(np.mean((_zridge(a, b, c, lam) - d) ** 2)) for a, b, c, d in folds])
              for lam in LAMBDAS}
    lam = min(scores, key=lambda l: (scores[l], l))
    return lam, scores[lam]


def main():
    os.makedirs(OUT, exist_ok=True)
    reps = pd.read_csv(os.path.join(DATA, "rule_representatives.csv")).set_index("candidate_key")
    df = load_data().iloc[:1460].reset_index(drop=True)
    y = df.target_property_damage_claim_count.values.astype(float)
    rows, wash = [], []
    chosen = chosen_candidates()
    print("chosen from the Pareto front:", chosen)
    for k, (key, r) in chosen.items():
        rep = reps.loc[key]
        J = json.loads(rep.J_json)
        edges = GRAPHS[rep.topology]
        Jl = [J[f"J{min(e)}{max(e)}"] for e in edges]
        names = FEATURES[rep.test_readout]
        theta = encode(df, rep.alpha, **training_scales(df))
        res = Reservoir(2, edges, Jl, rep.hx, rep.hy, rep["dt"], r)
        S, F = res.maps(theta, names)
        if rep.protocol == "CONT":
            tw_trace, tw_feat = washout_steps(res, S, F)
            W = max(tw_trace, tw_feat)
            wash.append(dict(candidate=k, key=key, trace_steps=tw_trace, feature_steps=tw_feat, window=W))
            Xc = res.cont(S, F)
            ends = np.arange(W - 1, 1460)
            Xr = res.rwp(S, F, ends, W)
            # continuous readout applied to replayed features on 2025
            lam = select_lambda(Xc[:1095], y[:1095], LAMBDAS)[0]
            pc = _zridge(Xc[:1095], y[:1095], Xc[1095:], lam)
            pr = _zridge(Xc[:1095], y[:1095], Xr[ends >= 1095], lam)
            wash[-1]["replay_minus_cont_rmse_2025"] = float(np.sqrt(np.mean((pr - y[1095:]) ** 2))
                                                           - np.sqrt(np.mean((pc - y[1095:]) ** 2)))
            X, fit_rows = Xc[:1095], np.arange(1095)
        else:
            W = int(rep.window)
            ends = np.arange(W - 1, 1095)
            X, fit_rows = res.rwp(S, F, ends, W), ends
        # fold-local readout
        lam_old, cv_old, _ = select_lambda(X, y[fit_rows], LAMBDAS)       # before the correction
        lam, cvm = fold_local_lambda(df, rep, edges, Jl, r, W if rep.protocol == "RWP" else 1, names, y)
        frozen = CANDIDATES[k]
        rows.append(dict(candidate=k, key=key, graph=rep.topology, protocol=rep.protocol, r=r, window=W,
                         alpha=rep.alpha, hx=rep.hx, hy=rep.hy, lam_before=lam_old, cv_before=cv_old, lam=lam, cv_rmse=cvm, readout=rep.test_readout,
                         matches_frozen=bool(frozen["graph"] == rep.topology and frozen["r"] == r
                                             and frozen["window"] == W and np.allclose(frozen["J"], Jl)
                                             and np.isclose(frozen["hx"], rep.hx) and np.isclose(frozen["hy"], rep.hy)
                                             and np.isclose(frozen["alpha"], rep.alpha)
                                             and frozen["features"] == names and np.isclose(frozen["lam"], lam))))
        print(rows[-1], flush=True)
    pd.DataFrame(wash).to_csv(os.path.join(OUT, "freeze_washout.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "freeze_candidates.csv"), index=False)
    print(pd.DataFrame(wash).to_string(index=False))


if __name__ == "__main__":
    main()
