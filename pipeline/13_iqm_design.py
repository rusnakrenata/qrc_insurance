"""IQM Emerald design: native embeddings, the H5/H6 coupling search and the five frozen IQM roles.

  python pipeline/13_iqm_design.py [--search]

Default: re-evaluates the recorded selections (data/iqm/design/). --search repeats the
coupling search on H5 and H6 (60 candidates each). The later IQM stages (Y field, Trotter depth,
replay search, rules R1-R5, IQM calibrated-simulator layout screen) are kept as recorded outputs;
the simulator screen needs IQM's Qiskit backend and is not repeated here.

The 2026 forecasts of the five roles are ideal (noiseless) and post hoc: the roles were frozen
before any 2026 data were used, and no IQM 2026 hardware run exists.

Writes results/iqm/.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import INITIAL_STATES, Reservoir, cv_rmse, mc_fast, probe  # noqa: E402
from qrc.layouts import native_embeddings  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
DES = os.path.join(DATA, "iqm", "design")
OUT = os.path.join(HERE, "..", "results", "iqm")
TRAIN, VAL, TEST = np.arange(0, 1095), np.arange(1095, 1460), np.arange(1460, 1692)
IQM_GRAPHS = {"H5": [(0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (0, 1), (4, 5)],
              "H6": [(0, 1), (1, 5), (2, 5), (2, 3), (3, 4), (0, 4), (4, 5)]}
INJ = ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3"]
READOUTS = {"XZ_injection": INJ, "XZinj_dropX3": [f for f in INJ if f != "X3"],
            "XZinj_plus_YX45": INJ + ["YX45"], "XZinj_dropX3_plus_YX45": [f for f in INJ if f != "X3"] + ["YX45"]}
LHS_SEED, N_LHS = 84201, 28


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def embeddings():
    e = pd.read_csv(os.path.join(DES, "emerald_coupling_edges.csv"))
    qubits = sorted(set(e.source) | set(e.target))
    adj = {q: set() for q in qubits}
    for a, b in zip(e.source, e.target):
        adj[a].add(b)
        adj[b].add(a)
    rec = pd.read_csv(os.path.join(DES, "iqm_region_summary.csv"))
    rec = rec[rec.backend == "emerald"].set_index("topology").native_embeddings
    rows = [dict(graph=g, edges=len(ed), native_embeddings=len(native_embeddings(ed, adj, qubits)),
                 recorded=int(rec[g])) for g, ed in {**GRAPHS, **IQM_GRAPHS}.items()]
    return pd.DataFrame(rows)


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, d))
    for k in range(d):
        out[:, k] = (rng.permutation(n) + rng.random(n)) / n
    return out


def search_vectors(d):
    return np.vstack([[[-1] * d, [1] * d, [-.5] * d, [.5] * d], -2 + 4 * lhs(N_LHS, d, LHS_SEED)])


def local_vectors(centre):
    out = []
    for k in range(len(centre)):
        for dlt in (-.5, -.25, .25, .5):
            v = centre.copy()
            v[k] = np.clip(v[k] + dlt, -2.5, 2.5)
            out.append(v)
    return out


def coupling_stage(search):
    df = load_data().iloc[:1460].reset_index(drop=True)
    pr = probe(encode(df, 0.75, **training_scales(df)))
    rec = pd.read_csv(os.path.join(DES, "graph_coupling_winners.csv")).set_index("topology")
    winners, trials = [], []
    for g, edges in IQM_GRAPHS.items():
        def ev(J):
            mc, tw = mc_fast(Reservoir(2, edges, list(J), 0.5, 0.0, 1.6, 2), pr)
            return dict(mc_per_channel=mc, washout=tw, valid=tw is not None and tw <= 300)
        if search:
            rows = [dict(graph=g, stage="global", J=list(map(float, v)), **ev(v)) for v in search_vectors(len(edges))]
            centre = np.array(max([r for r in rows if r["valid"]] or rows, key=lambda r: r["mc_per_channel"])["J"])
            rows += [dict(graph=g, stage="local", J=list(map(float, v)), **ev(v)) for v in local_vectors(centre)]
            trials += rows
            best = max([r for r in rows if r["valid"]] or rows, key=lambda r: r["mc_per_channel"])
        else:
            J = json.loads(rec.loc[g, "J_vector"])
            best = dict(graph=g, stage="recorded winner", J=J, **ev(J))
        best["recorded_mc_per_channel"] = float(rec.loc[g, "MC_per_channel_corrected"])
        best["recorded_washout"] = float(rec.loc[g, "Tw_0.01"])
        winners.append(best)
        print({k: v for k, v in best.items() if k != "J"}, flush=True)
    if trials:
        pd.DataFrame(trials).to_csv(os.path.join(OUT, "coupling_search.csv"), index=False)
    return pd.DataFrame(winners)


def roles():
    five = pd.read_csv(os.path.join(DES, "final_five.csv"))
    h6 = pd.read_csv(os.path.join(DES, "rule_candidates_rwp.csv")).set_index("candidate_uid")
    h2 = pd.read_csv(os.path.join(DATA, "design", "rule_representatives.csv")).set_index("candidate_key")
    out = []
    for r in five.itertuples():
        if r.topology == "H6":
            s = h6.loc[r.candidate_id]
            edges = IQM_GRAPHS["H6"]
            J = [float(s[f"J{min(e)}{max(e)}"]) for e in edges]
            c = dict(alpha=s.alpha, dt=s["dt"], hx=s.hx, hy=s.hy, lam=float(s.selected_lambda))
        else:
            s = h2.loc[r.candidate_id]
            edges = GRAPHS["H2"]
            Jd = json.loads(s.J_json)
            J = [Jd[f"J{min(e)}{max(e)}"] for e in edges]
            c = dict(alpha=s.alpha, dt=s["dt"], hx=s.hx, hy=s.hy, lam=float(s.source_selected_lambda))
        c.update(edges=edges, J=J, role=int(r.role_id), rule=r.selected_rules, candidate_id=r.candidate_id,
                 graph=r.topology, protocol=r.protocol, r=int(r.r), readout=r.readout, window=int(r.window_effective),
                 cv_recorded=float(r.cv_rmse), rmse_2025_recorded=r.validation_rmse_2025_diagnostic)
        out.append(c)
    return out


def washout(S, F, eps=0.01):
    vs = [s.reshape(-1) for s in INITIAL_STATES]
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


def role_stage():
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    scales = training_scales(df)
    rows, preds = [], []
    for c in roles():
        names = READOUTS[c["readout"]]
        theta = encode(df, c["alpha"], **scales)
        A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
        extra = {}
        if c["protocol"] == "CONT":
            Xc = cont_features(A, names)
            Xfit, yfit = Xc[TRAIN], y[TRAIN]
            S, F = Reservoir(2, c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]).maps(theta[:1095], names)
            tw = washout(S, F)
            W = max(tw)
            extra = dict(washout_trace=tw[0], washout_feature=tw[1])
        else:
            W = c["window"]
            idx = np.arange(W - 1, 1095)
            Xfit, yfit = rwp_features(A, idx, W, names), y[idx]
        cv, sd = cv_rmse(Xfit, yfit, c["lam"])
        w, b = fit_ridge(Xfit, yfit, c["lam"])
        ends = np.r_[VAL, TEST]
        p = rwp_features(A, ends, W, names) @ w + b
        e25, e26 = p[:len(VAL)] - y[VAL], p[len(VAL):] - y[TEST]
        h = len(TEST) // 2
        row = dict(role=c["role"], rule=c["rule"], candidate_id=c["candidate_id"], graph=c["graph"],
                   design_protocol=c["protocol"], replay_window=W, r=c["r"], readout=c["readout"], m=len(names),
                   lam=c["lam"], cv_recorded=c["cv_recorded"], cv=cv, cv_sd=sd,
                   rmse_2025_recorded=c["rmse_2025_recorded"], rmse_2025=rmse(e25), rmse_2026=rmse(e26),
                   mae_2026=float(np.abs(e26).mean()), bias_2026=float(e26.mean()), crmse_2026=float(np.std(e26)),
                   rmse_2026_first_half=rmse(e26[:h]), rmse_2026_second_half=rmse(e26[h:]), **extra)
        if c["protocol"] == "CONT":
            pc = Xc[ends] @ w + b
            row["replay_minus_cont_rmse_2025"] = row["rmse_2025"] - rmse(pc[:len(VAL)] - y[VAL])
        rows.append(row)
        preds.append(pd.DataFrame(dict(role=c["role"], target_date=df.target_date.values[TEST], actual=y[TEST],
                                       ideal_prediction=p[len(VAL):])))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return pd.DataFrame(rows), pd.concat(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    emb = embeddings()
    emb.to_csv(os.path.join(OUT, "emerald_embeddings.csv"), index=False)
    print(emb.to_string(index=False))
    coupling_stage(args.search).to_csv(os.path.join(OUT, "coupling_winners.csv"), index=False)
    summ, preds = role_stage()
    summ.to_csv(os.path.join(OUT, "roles_ideal.csv"), index=False)
    preds.to_csv(os.path.join(OUT, "roles_ideal_2026_predictions.csv"), index=False)
    screen = pd.read_csv(os.path.join(DES, "sim_layout_screen.csv"))
    print("IQM simulator layout screen (role 1, recorded):",
          f"{len(screen)} layouts, d = {screen.sim_pred_distortion_rmse.min():.2f}-{screen.sim_pred_distortion_rmse.max():.2f}")


if __name__ == "__main__":
    main()
