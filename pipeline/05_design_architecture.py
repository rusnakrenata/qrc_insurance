"""Design stage 1: number of memory qubits, CONT vs RWP, exposure encoding.

  python pipeline/05_design_architecture.py

Writes results/design/architecture.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import load_data, training_scales  # noqa: E402
from qrc.design import Reservoir, _zridge, select_lambda  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "design")
WINDOWS = [1, 2, 5, 7, 14, 21, 28]
ENCODINGS = {"clip3": lambda z: np.clip(z / 3, -1, 1), "tanh3": lambda z: np.tanh(z / 3),
             "arctan": lambda z: 2 / np.pi * np.arctan(np.pi * z / 6), "softsign3": lambda z: z / (3 + np.abs(z))}


def angles(df, encoding, alpha=1.0):
    s = training_scales(df)
    zc = (df.property_damage_claim_count_t - s["mean_c"]) / s["std_c"]
    zp = (df.active_policy_count_t - s["mean_p"]) / s["std_p"]
    return np.column_stack([alpha * np.clip(zc, -3, 3) / 3, 2 * np.pi * df.target_date.dt.dayofweek / 7,
                            alpha * ENCODINGS[encoding](zp.values), np.pi * df.is_public_holiday_t_plus_1])


def evaluate(df, n_mem, encoding, J):
    n = 4 + n_mem
    res = Reservoir(n_mem, [(q, q + 1) for q in range(n - 1)], J[: n - 1], hx=0.5, dt=0.8, r=2)
    names = [f"{p}{q}" for q in range(n) for p in "XZ"]
    S, F = res.maps(angles(df, encoding), names)
    y = df.target_property_damage_claim_count.values.astype(float)
    rows = []
    for proto, w in [("CONT", None)] + [("RWP", w) for w in WINDOWS]:
        if proto == "CONT":
            X, ends = res.cont(S, F), np.arange(len(df))
        else:
            ends = np.arange(w - 1, len(df))
            X = res.rwp(S, F, ends, w)
        tr, va = ends < 1095, (ends >= 1095) & (ends < 1460)
        lam, cv, sd = select_lambda(X[tr], y[ends[tr]])
        val = np.sqrt(np.mean((_zridge(X[tr], y[ends[tr]], X[va], lam) - y[ends[va]]) ** 2))
        rows.append(dict(memory_qubits=n_mem, encoding=encoding, protocol=proto, window=w, lam=lam,
                         cv_rmse=cv, cv_sd=sd, rmse_2025=val))
        print(rows[-1], flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data().iloc[:1460].reset_index(drop=True)          # 2026 not needed
    J = pd.read_csv(os.path.join(HERE, "..", "data", "design", "chain_couplings.csv")).J_ij.values
    rows = []
    for m in (1, 2, 3, 4):
        rows += evaluate(df, m, "clip3", J)
    for enc in ("tanh3", "arctan", "softsign3"):
        rows += evaluate(df, 2, enc, J)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "architecture.csv"), index=False)


if __name__ == "__main__":
    main()
