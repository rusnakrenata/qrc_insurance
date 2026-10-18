"""Reconstruct the IQM Emerald probe batch: 25 training dates, two layouts, 512 shots.

The leader reservoir (graph H6, W=6, r=2) is rebuilt from its identity file, its ridge
readout (lambda=1) is fitted on noiseless 2022-2024 features, and the returned counts
are turned into expectation values and forecasts. Writes results/tables/iqm_probe.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import load_data, training_scales  # noqa: E402
from qrc.circuits import expectation_from_counts  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
NAMES = ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3", "YX45"]


def main():
    ident = json.load(open(os.path.join(DATA, "iqm", "iqm_candidate.json")))
    keys = ["01", "15", "52", "23", "34", "40", "45"]
    edges = [(int(k[0]), int(k[1])) for k in keys]
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    theta = encode(df, ident["alpha"], **training_scales(df))
    A = step_operators(trotter_unitary(edges, [ident["J"][k] for k in keys], ident["hx"], ident["hy"],
                                       ident["dt"], ident["r"]), theta)
    fit = np.arange(5, 1095)
    w, b = fit_ridge(rwp_features(A, fit, 6, NAMES), y[fit], 1.0)

    raw = pd.read_csv(os.path.join(DATA, "iqm", "iqm_probe_counts.csv"))
    rows = []
    for (e, lay), g in raw.groupby(["endpoint", "layout_id"]):
        cx = json.loads(g[g.setting == "XXXXYX"].counts_json.iloc[0])
        cz = json.loads(g[g.setting == "ZZZZZZ"].counts_json.iloc[0])
        zq = [expectation_from_counts(cz if n[0] == "Z" else cx, [4, 5] if n == "YX45" else [int(n[1])]) for n in NAMES]
        zi = rwp_features(A, [e], 6, NAMES)[0]
        rows.append(dict(layout=lay, endpoint=e, pred_qpu=np.dot(zq, w) + b, pred_ideal=zi @ w + b))
    d = pd.DataFrame(rows)
    out = []
    for lay, g in d.groupby("layout"):
        delta = g.pred_qpu - g.pred_ideal
        out.append(dict(layout=lay, dates=len(g), d=np.sqrt(np.mean(delta ** 2)), level_shift=delta.mean(),
                        level_share=delta.mean() ** 2 / np.mean(delta ** 2),
                        corr_with_ideal=np.corrcoef(g.pred_qpu, g.pred_ideal)[0, 1]))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(HERE, "..", "results", "tables", "iqm_probe.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
