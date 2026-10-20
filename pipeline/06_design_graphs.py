"""Design stage 2: Tm, native embeddings on Heron and the coupling search on H0-H4.

  python pipeline/06_design_graphs.py [--search]

Writes results/design/graphs_*.csv.
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
from qrc.design import ONE_BODY, Reservoir, cv_rmse, mc_fast, probe  # noqa: E402
from qrc.layouts import native_embeddings  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "design")
OUT = os.path.join(HERE, "..", "results", "design")
Tm = dict(edges=[(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
          J=[0.3835384679783487, -0.0855701843471267, 0.5020370878759355, -0.907276, -0.444414, -2.0, -1.0])
LHS_SEED, N_LHS = 84201, 28
ANCHORS = [[-1] * 5, [1] * 5, [-.5] * 5, [.5] * 5]


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((n, d))
    for k in range(d):
        out[:, k] = (rng.permutation(n) + rng.random(n)) / n
    return out


def global_vectors():
    return np.vstack([ANCHORS, -2 + 4 * lhs(N_LHS, 5, LHS_SEED)])


def local_vectors(centre):
    out, seen = [], set()
    for k in range(5):
        for d in (-.5, -.25, .25, .5):
            v = centre.copy()
            v[k] = np.clip(v[k] + d, -2.5, 2.5)
            key = tuple(np.round(v, 10))
            if key not in seen:
                seen.add(key)
                out.append(v)
    return out


def embeddings_table():
    edges = pd.read_csv(os.path.join(DATA, "heron_coupling_edges.csv"))
    rows = []
    for backend, g in edges.groupby("backend"):
        qubits = sorted(set(g.source) | set(g.target))
        adj = {q: set() for q in qubits}
        for a, b in zip(g.source, g.target):
            adj[a].add(b)
            adj[b].add(a)
        for name, e in [("Tm", Tm["edges"])] + list(GRAPHS.items()):
            rows.append(dict(backend=backend, graph=name, native_embeddings=len(native_embeddings(e, adj, qubits))))
    return pd.DataFrame(rows)


def evaluate(graph, J, pr, theta, y):
    """MC and washout on the probe; forecast CV on the real chronology (diagnostic)."""
    res = Reservoir(2, graph, J, 0.5, 0.0, 1.6, 2)
    mc, tw = mc_fast(res, pr)
    X = res.cont(*res.maps(theta[:1095], ONE_BODY))
    return dict(mc_per_channel=mc, washout=tw, valid=tw is not None and tw <= 300,
                cv_rmse_xyz=cv_rmse(X, y, 0.01)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    emb = embeddings_table()
    emb.to_csv(os.path.join(OUT, "graphs_embeddings.csv"), index=False)
    print(emb.pivot(index="graph", columns="backend", values="native_embeddings"))

    df = load_data().iloc[:1460].reset_index(drop=True)
    theta = encode(df, 0.75, **training_scales(df))
    pr = probe(theta)
    y = df.target_property_damage_claim_count.values[:1095].astype(float)
    tm = mc_fast(Reservoir(2, Tm["edges"], Tm["J"], 0.5, 0.0, 1.6, 2), pr)
    print(f"Tm: MC/ch = {tm[0]:.4f}, T_w = {tm[1]}")

    winners, trials = [], []
    recorded = pd.read_csv(os.path.join(DATA, "graph_coupling_winners.csv")).set_index("topology")
    for name, edges in GRAPHS.items():
        if args.search:
            rows = []
            for stage, vecs in (("global", global_vectors()), ("local", None)):
                if stage == "local":
                    valid = [r for r in rows if r["valid"]] or rows
                    vecs = local_vectors(np.array(max(valid, key=lambda r: r["mc_per_channel"])["J"]))
                for v in vecs:
                    rows.append(dict(graph=name, stage=stage, J=list(map(float, v)),
                                     **evaluate(edges, list(v), pr, theta, y)))
                    print(name, stage, round(rows[-1]["mc_per_channel"], 4), rows[-1]["washout"], flush=True)
            trials += rows
            best = max([r for r in rows if r["valid"]] or rows, key=lambda r: r["mc_per_channel"])
        else:
            J = json.loads(recorded.loc[name, "J_vector"])
            best = dict(graph=name, stage="recorded winner", J=J, **evaluate(edges, J, pr, theta, y))
            best["recorded_mc_per_channel"] = recorded.loc[name, "MC_per_channel_corrected"]
        best["mc_retained_vs_Tm"] = best["mc_per_channel"] / tm[0]
        winners.append(best)
        print({k: v for k, v in best.items() if k != "J"}, flush=True)
    if trials:
        pd.DataFrame(trials).to_csv(os.path.join(OUT, "graphs_J_search.csv"), index=False)
    pd.DataFrame(winners).to_csv(os.path.join(OUT, "graphs_winners.csv"), index=False)


if __name__ == "__main__":
    main()
