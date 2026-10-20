"""Noiseless features, readouts and forecasts of the five candidates.

  python pipeline/11_ideal_reservoir.py

Writes results/ideal/.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "ideal")
TRAIN, VAL, TEST = np.arange(0, 1095), np.arange(1095, 1460), np.arange(1460, 1692)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    scales = training_scales(df)
    rows = []
    for k, c in CANDIDATES.items():
        theta = encode(df, c["alpha"], **scales)
        A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
        if c["protocol"] == "CONT":               # readout designed on continuous operation
            Xfit, yfit = cont_features(A[:1095], c["features"]), y[TRAIN]
        else:
            fit_idx = np.arange(c["window"] - 1, 1095)
            Xfit, yfit = rwp_features(A, fit_idx, c["window"], c["features"]), y[fit_idx]
        w, b = fit_ridge(Xfit, yfit, c["lam"])
        ends = np.r_[VAL, TEST]
        X = rwp_features(A, ends, c["window"], c["features"])     # deployed protocol: replay window W
        pred = X @ w + b
        p25, p26 = pred[:len(VAL)], pred[len(VAL):]
        out = pd.DataFrame(dict(endpoint=ends, target_date=df.target_date.values[ends], actual=y[ends], prediction=pred))
        for j, name in enumerate(c["features"]):
            out[name] = X[:, j]
        out.to_csv(os.path.join(OUT, f"candidate{k}_predictions.csv"), index=False)
        pd.DataFrame(dict(feature=c["features"], weight=w)).assign(intercept=b).to_csv(
            os.path.join(OUT, f"candidate{k}_readout.csv"), index=False)
        rows.append(dict(candidate=k, graph=c["graph"], window=c["window"], r=c["r"],
                         rmse_2025=rmse(p25 - y[VAL]), rmse_2026=rmse(p26 - y[TEST]),
                         bias_2026=float(np.mean(p26 - y[TEST]))))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
