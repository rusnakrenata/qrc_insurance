"""Noiseless diagnostics of the five frozen reservoirs (no hardware involved).

  1. Encoding range: how often the 2025/2026 inputs leave the training range.
  2. Readout refitted on 2022-2025 (same features, same lambda) vs Ridge refitted the same way.
  3. Memory-window ablation for the two long-window candidates (C4, C5).
  4. Shot-noise floor of the transfer distortion, sqrt(kappa/N).
Writes results/tables/ideal_*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import (OBS, cont_features, encode, rwp_features,  # noqa: E402
                           step_operators, trotter_unitary)

OUT = os.path.join(HERE, "..", "results", "tables")
TRAIN, VAL = np.arange(0, 1095), np.arange(1095, 1460)
TEST = np.arange(1460, 1692)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def features(c, A, endpoints, window=None):
    window = window or c["window"]
    return rwp_features(A, endpoints, window, c["features"])


def ridge_classical(df, fit_idx, lam=1.0):
    """The Ridge reference on F4 inputs (standardised on the fitting period)."""
    cols = ["C_t_z", "D_sin", "D_cos", "P_t_z", "is_public_holiday_t_plus_1"]
    X, y = df[cols].values.astype(float), df.target_property_damage_claim_count.values.astype(float)
    Xt, yt = X[fit_idx], y[fit_idx]
    mu, ym = Xt.mean(0), yt.mean()
    w = np.linalg.solve((Xt - mu).T @ (Xt - mu) + lam * np.eye(len(cols)), (Xt - mu).T @ (yt - ym))
    return X @ w + (ym - mu @ w)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    sc = training_scales(df)

    # 1. encoding range
    zc = (df.property_damage_claim_count_t - sc["mean_c"]) / sc["std_c"]
    zp = (df.active_policy_count_t - sc["mean_p"]) / sc["std_p"]
    rows = []
    for name, idx in [("train 2022-24", TRAIN), ("validation 2025", VAL), ("test 2026 first half", TEST[:116]),
                      ("test 2026 second half", TEST[116:]), ("test 2026", TEST)]:
        s = zp.values[idx] / (3 + np.abs(zp.values[idx]))
        rows.append(dict(period=name, n=len(idx), claims_mean=df.property_damage_claim_count_t.values[idx].mean(),
                         claims_clipped_share=float(np.mean(zc.values[idx] > 3)),
                         exposure_min=df.active_policy_count_t.values[idx].min(),
                         exposure_max=df.active_policy_count_t.values[idx].max(),
                         exposure_angle_min=s.min(), exposure_angle_max=s.max(),
                         claim_frequency_per_1000=float(np.mean(df.property_damage_claim_count_t.values[idx]
                                                                / df.active_policy_count_t.values[idx] * 1000))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ideal_encoding_range.csv"), index=False)

    # 2. readout refitted through 2025 and 3. window ablation
    refit, ablation, floors = [], [], []
    for k, c in CANDIDATES.items():
        theta = encode(df, c["alpha"], **sc)
        A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
        ep = np.arange(c["window"] - 1, 1692)
        X = features(c, A, ep)
        Xf = pd.DataFrame(X, index=ep)
        if c["protocol"] == "CONT":     # readout trained on continuous features, deployed with RWP window
            Xc = cont_features(A, c["features"])
            fit_X, fit_y = Xc[TRAIN], y[TRAIN]
            fit_X25, fit_y25 = Xc[np.r_[TRAIN, VAL]], y[np.r_[TRAIN, VAL]]
        else:
            tr = ep[ep < 1095]
            fit_X, fit_y = Xf.loc[tr].values, y[tr]
            tr25 = ep[ep < 1460]
            fit_X25, fit_y25 = Xf.loc[tr25].values, y[tr25]
        w, b = fit_ridge(fit_X, fit_y, c["lam"])
        w25, b25 = fit_ridge(fit_X25, fit_y25, c["lam"])
        p26, p26b = Xf.loc[TEST].values @ w + b, Xf.loc[TEST].values @ w25 + b25
        p25 = Xf.loc[VAL].values @ w + b
        refit.append(dict(candidate=k, rmse_2025=rmse(p25 - y[VAL]), bias_2025=float(np.mean(p25 - y[VAL])),
                          rmse_2026=rmse(p26 - y[TEST]), bias_2026=float(np.mean(p26 - y[TEST])),
                          rmse_2026_refit=rmse(p26b - y[TEST]), bias_2026_refit=float(np.mean(p26b - y[TEST]))))

        # 4. shot-noise floor: kappa = mean_t w' Sigma_t w with single-shot covariances per setting
        kap = []
        for t in TEST:
            rho = state_at(A, t, c["window"])
            cov = np.zeros((len(w), len(w)))
            for i, a in enumerate(c["features"]):
                for j, bb in enumerate(c["features"]):
                    if setting_of(c, a) != setting_of(c, bb):
                        continue
                    ea = np.trace(rho @ OBS[a]).real
                    eb = np.trace(rho @ OBS[bb]).real
                    cov[i, j] = np.trace(rho @ OBS[a] @ OBS[bb]).real - ea * eb
            kap.append(w @ cov @ w)
        kappa = float(np.mean(kap))
        floors.append(dict(candidate=k, kappa=kappa, floor_256=np.sqrt(kappa / 256),
                           floor_eval=np.sqrt(kappa / c["shots"]), shots_eval=c["shots"]))

        if k in (4, 5):
            for W in [1, 2, 4, 8, 16, 32, 64] + ([117] if k == 4 else []):
                eW = np.arange(W - 1, 1692)
                XW = pd.DataFrame(features(c, A, eW, W), index=eW)
                trW = eW[eW < 1095]
                wW, bW = fit_ridge(XW.loc[trW].values, y[trW], c["lam"])
                ablation.append(dict(candidate=k, window=W,
                                     rmse_2025=rmse(XW.loc[VAL].values @ wW + bW - y[VAL]),
                                     rmse_2026=rmse(XW.loc[TEST].values @ wW + bW - y[TEST])))
    ridge = ridge_classical(df, TRAIN)
    ridge25 = ridge_classical(df, np.r_[TRAIN, VAL])
    refit.append(dict(candidate="Ridge", rmse_2025=rmse(ridge[VAL] - y[VAL]), bias_2025=float(np.mean(ridge[VAL] - y[VAL])),
                      rmse_2026=rmse(ridge[TEST] - y[TEST]), bias_2026=float(np.mean(ridge[TEST] - y[TEST])),
                      rmse_2026_refit=rmse(ridge25[TEST] - y[TEST]),
                      bias_2026_refit=float(np.mean(ridge25[TEST] - y[TEST]))))
    pd.DataFrame(refit).to_csv(os.path.join(OUT, "ideal_refit_2025.csv"), index=False)
    pd.DataFrame(ablation).to_csv(os.path.join(OUT, "ideal_window_ablation.csv"), index=False)
    pd.DataFrame(floors).to_csv(os.path.join(OUT, "ideal_shot_floor.csv"), index=False)
    for f in ["ideal_encoding_range", "ideal_refit_2025", "ideal_window_ablation", "ideal_shot_floor"]:
        print(pd.read_csv(os.path.join(OUT, f + ".csv")).round(3).to_string())


def setting_of(c, name):
    """Measurement setting in which an observable is read out (X-type or Z-type basis)."""
    return "Z" if name.startswith("Z") else "X"


def state_at(A, t, window):
    rho_m = np.zeros((4, 4), dtype=complex)
    rho_m[0, 0] = 1
    for s in range(t - window + 1, t + 1):
        rho = np.einsum("amn,nk,bpk->ambp", A[s], rho_m, A[s].conj()).reshape(64, 64)
        rho_m = np.einsum("amn,nk,apk->mp", A[s], rho_m, A[s].conj())
    return rho


if __name__ == "__main__":
    main()
