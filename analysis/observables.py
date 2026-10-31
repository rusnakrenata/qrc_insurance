"""Per-observable fit z_qpu = eta * z_ideal + zeta and contribution w_j * mean(dz_j) to the mean shift.

  python analysis/observables.py

Writes results/observables.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm-final", default=os.path.join(HERE, "..", "data", "ibm_kingston_runs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    runs = pd.read_csv(os.path.join(a.out, "runs.csv"))
    rows = []
    for _, r in runs.iterrows():
        folder = os.path.join(a.ibm_final, r.source)
        plan = json.load(open(os.path.join(folder, "plan.json")))
        ref = np.load(os.path.join(folder, "reference.npz"))
        names, w = plan["features"], ref["weights"]
        d = pd.read_csv(os.path.join(folder, "2026_features_predictions.csv"))
        for j, name in enumerate(names):
            zi, zq = d[name + "_ideal"].values, d[name + "_qpu"].values
            eta, zeta = np.polyfit(zi, zq, 1)
            rows.append(dict(candidate=r.candidate, round=r["round"], approach=r.approach, observable=name,
                             weight=w[j], ideal_sd=zi.std(), eta=eta, zeta=zeta,
                             mean_dz=float(np.mean(zq - zi)), shift_contribution=float(w[j] * np.mean(zq - zi)),
                             corr=float(np.corrcoef(zi, zq)[0, 1])))
        # contributions sum to the level shift
        tot = sum(x["shift_contribution"] for x in rows[-len(names):])
        assert abs(tot - r.level_shift) < 1e-6, (r.candidate, r["round"], r.approach)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "observables.csv"), index=False)
    print(len(rows), "observable rows")


if __name__ == "__main__":
    main()
