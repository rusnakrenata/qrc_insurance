"""Design stage 3: Y field on the memory qubits, observable sets, Trotter depth.

  python pipeline/07_design_dynamics.py [--scan] [--search]

Writes results/design/dynamics_*.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import (ONE_BODY, PROBE_SEED, Reservoir, cv_rmse, memory_capacity,  # noqa: E402
                        process_fidelity, select_lambda, washout_S)
from qrc.reservoir import encode  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "design")
OUT = os.path.join(HERE, "..", "results", "design")
READOUTS = {"XZ_injection": ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3"],
            "XZinj_plus_YX45": ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3", "YX45"],
            "XYZ_all": ONE_BODY}
ALL_NAMES = ONE_BODY + ["YX45"]
HY_GRID = [-0.6, -0.4, -0.3, -0.2, 0.0, 0.2, 0.3, 0.4, 0.6]
HY_KEPT = {"H0": [-0.4, 0.0], "H1": [-0.6, 0.0], "H2": [0.0, 0.3, 0.4, 0.6], "H3": [-0.2, 0.0, 0.3],
           "H4": [-0.6, 0.0, 0.4, 0.6]}                 # kept after inspecting the scan (15 dynamics seeds)
PROBE_SEEDS = [PROBE_SEED, 42, 101, 505, 707]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0]
EDGE_COL = lambda e: f"J{min(e)}{max(e)}"                          # noqa: E731


def stage2_couplings():
    w = pd.read_csv(os.path.join(DATA, "graph_coupling_winners.csv")).set_index("topology")
    return {g: json.loads(w.loc[g, "J_vector"]) for g in GRAPHS}


def probes(theta):
    out = []
    for s in PROBE_SEEDS:
        rng = np.random.default_rng(s)
        out.append(np.column_stack([theta[rng.permutation(len(theta)), k] for k in range(4)]))
    return out


def memory(res, probe_list):
    """Mean MC per channel and mean washout over the probe permutations."""
    mcs, tws = [], []
    for pr in probe_list:
        S, F = res.maps(pr, ONE_BODY)
        tw = washout_S(S)
        burnin = min(max(100, tw + 20) if tw is not None else 995, 995)
        mcs.append(memory_capacity(None, pr, 1095, burnin, X=res.cont(S, F)))
        tws.append(np.nan if tw is None else tw)
    return float(np.mean(mcs)), float(np.nanmean(tws))


def real_features(res, theta):
    S, F = res.maps(theta, ALL_NAMES)
    return pd.DataFrame(res.cont(S, F), columns=ALL_NAMES)


def forecast(X, y):
    """Best readout penalty by CV and the 2025 diagnostic RMSE."""
    from qrc.design import _zridge
    lam, cv, sd = select_lambda(X[:1095], y[:1095], LAMBDAS)
    val = np.sqrt(np.mean((_zridge(X[:1095], y[:1095], X[1095:], lam) - y[1095:]) ** 2))
    return dict(lam=lam, cv_rmse=cv, cv_sd=sd, rmse_2025=val)


def y_scan(theta, y, J2, full):
    rows = []
    pl = probes(theta)
    for g, edges in GRAPHS.items():
        for hy in (HY_GRID if full else sorted(set(HY_KEPT[g]))):
            res = Reservoir(2, edges, J2[g], 0.5, hy, 1.6, 2)
            mc, tw = memory(res, pl)
            X = real_features(res, theta)
            rows.append(dict(graph=g, hy=hy, mc_per_channel=mc, washout=tw,
                             cv_xz_lam001=cv_rmse(X[READOUTS["XZ_injection"]].values[:1095], y[:1095], 0.01)[0],
                             cv_xyz_lam001=cv_rmse(X[ONE_BODY].values[:1095], y[:1095], 0.01)[0]))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def readout_recheck(theta, y, J2):
    rows = []
    for g, edges in GRAPHS.items():
        for hy in HY_KEPT[g]:
            X = real_features(Reservoir(2, edges, J2[g], 0.5, hy, 1.6, 2), theta)
            for name, cols in READOUTS.items():
                rows.append(dict(graph=g, hy=hy, readout=name, **forecast(X[cols].values, y)))
    return pd.DataFrame(rows)


def trotter_search(theta, y, J2):
    """Equal-budget re-optimization of (hx, hy, J, lambda) for r = 1, 2, 3."""
    rows = []
    for g, edges in GRAPHS.items():
        for hy0 in HY_KEPT[g]:
            seed_id = f"{g}_hy{hy0:+.1f}"
            for r in (1, 2, 3):
                cache = {}

                def evaluate(hx, hy, J, stage):
                    key = (round(hx, 12), round(hy, 12), tuple(np.round(J, 12)))
                    if key in cache:
                        return cache[key]
                    X = real_features(Reservoir(2, edges, list(J), hx, hy, 1.6, r), theta)
                    fid = process_fidelity(edges, list(J), hx, hy, 1.6, r)
                    out = [dict(seed=seed_id, graph=g, r=r, stage=stage, readout=name, hx=hx, hy=hy,
                                J=list(map(float, J)), process_fidelity=fid, **forecast(X[c].values, y))
                           for name, c in READOUTS.items()]
                    cache[key] = out
                    rows.extend(out)
                    return out

                block = list(evaluate(0.5, hy0, np.array(J2[g]), "baseline"))
                hys = [0.0] if hy0 == 0 else [float(np.clip(hy0 * f, -1, 1)) for f in (.5, 1, 1.5, -1)]
                for hx in (.25, .5, .75):
                    for hy in hys:
                        for s in (.5, 1, 1.5):
                            block += evaluate(hx, hy, np.clip(np.array(J2[g]) * s, -2, 2), "grid")
                b = pd.DataFrame(block)
                parents = {tuple(b[b.readout == n].sort_values(["cv_rmse", "process_fidelity"],
                                                               ascending=[True, False]).iloc[0][["hx", "hy"]])
                           + tuple(b[b.readout == n].sort_values("cv_rmse").iloc[0].J) for n in READOUTS}
                best_f = b.sort_values(["process_fidelity", "cv_rmse"], ascending=[False, True]).iloc[0]
                parents.add((best_f.hx, best_f.hy) + tuple(best_f.J))
                for k, p in enumerate(sorted(parents)):
                    rng = np.random.default_rng(910000 + sum(map(ord, seed_id)) + 1000 * r + 100 * k)
                    for _ in range(4):
                        hx = float(np.clip(rng.normal(p[0], .12), .1, 1))
                        hy = 0.0 if hy0 == 0 else float(np.clip(rng.normal(p[1], max(.08, .2 * abs(p[1]))), -1, 1))
                        if hy0 != 0 and abs(hy) < .05:
                            hy = .05 if p[1] >= 0 else -.05
                        J = [float(np.clip(rng.normal(v, max(.1, .2 * max(abs(v), .5))), -2, 2)) for v in p[2:]]
                        evaluate(hx, hy, np.array(J), "local")
                print(seed_id, "r =", r, "done", flush=True)
    return pd.DataFrame(rows)


def check_recorded(theta, y):
    """Recompute CV RMSE and process fidelity of the 135 recorded forecast-best rows."""
    rec = pd.read_csv(os.path.join(DATA, "trotter_search_best.csv"))
    rows = []
    for _, r in rec.iterrows():
        edges = GRAPHS[r.topology]
        J = [r[EDGE_COL(e)] for e in edges]
        res = Reservoir(2, edges, J, r.hx, r.hy, 1.6, int(r.trotter_r))
        cols = READOUTS[r.readout] if r.readout != "XZinj_plus_YX45" else READOUTS["XZinj_plus_YX45"]
        X = real_features(res, theta)[cols].values
        cv = cv_rmse(X[:1095], y[:1095], r.selected_lambda)[0]
        rows.append(dict(branch=r.seed_candidate_id, r=r.trotter_r, cv_recorded=r.cv_rmse, cv_recomputed=cv,
                         fidelity_recorded=r.process_fidelity,
                         fidelity_recomputed=process_fidelity(edges, J, r.hx, r.hy, 1.6, int(r.trotter_r))))
    out = pd.DataFrame(rows)
    best = out.loc[out.groupby("branch").cv_recomputed.idxmin()]
    print("branches whose best CV depth is r=3:", int((best.r == 3).sum()), "of", len(best))
    print("max |CV diff|", (out.cv_recorded - out.cv_recomputed).abs().max(),
          " max |fidelity diff|", (out.fidelity_recorded - out.fidelity_recomputed).abs().max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = load_data().iloc[:1460].reset_index(drop=True)
    theta = encode(df, 0.75, **training_scales(df))
    y = df.target_property_damage_claim_count.values.astype(float)
    J2 = stage2_couplings()
    if args.search:
        trotter_search(theta, y, J2).to_csv(os.path.join(OUT, "dynamics_trotter_search.csv"), index=False)
        return
    y_scan(theta, y, J2, args.scan).to_csv(os.path.join(OUT, "dynamics_y_scan.csv"), index=False)
    rc = readout_recheck(theta, y, J2)
    rc.to_csv(os.path.join(OUT, "dynamics_readout_recheck.csv"), index=False)
    print(rc.loc[rc.groupby(["graph", "hy"]).cv_rmse.idxmin()].to_string(index=False))
    check_recorded(theta, y).to_csv(os.path.join(OUT, "dynamics_trotter_check.csv"), index=False)


if __name__ == "__main__":
    main()
