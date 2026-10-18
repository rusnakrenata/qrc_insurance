"""Predictors of the hardware distortion available before deployment (probe, proxy, Aer).

  python analysis/distortion_model.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def loo_predictions(x, y, groups):
    """Leave-one-group-out ordinary least squares, returns out-of-sample predictions."""
    pred = np.empty_like(y)
    for g in np.unique(groups):
        tr = groups != g
        slope, intercept = np.polyfit(x[tr], y[tr], 1)
        pred[~tr] = slope * x[~tr] + intercept
    return pred


def summarise(name, x, y, groups, scope):
    p = loo_predictions(x, y, groups)
    ss = np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(predictor=name, scope=scope, n=len(y),
                pearson_r=stats.pearsonr(x, y).statistic, spearman_rho=stats.spearmanr(x, y).statistic,
                loo_r2=1 - ss, loo_mae=float(np.mean(np.abs(y - p)))), p


def main():
    runs = pd.read_csv(os.path.join(OUT, "runs.csv"))
    runs["group"] = runs.candidate * 10 + runs["round"]
    rows, preds = [], runs[["candidate", "round", "approach", "transfer_d", "level_shift"]].copy()

    ha = runs[runs.approach == "hardware-aware"]
    r, p = summarise("probe distortion (training endpoints)", ha.probe_d.values, ha.transfer_d.values,
                     ha.group.values, "HA")
    rows.append(r)
    preds.loc[ha.index, "pred_from_probe"] = p
    r, p = summarise("probe level shift -> realised level shift", ha.probe_shift.values, ha.level_shift.values,
                     ha.group.values, "HA")
    rows.append(r)

    ok = runs[np.isfinite(runs.error_proxy_nll)]
    x = np.log10(ok.error_proxy_nll.values)
    r, p = summarise("log10 workload error proxy", x, ok.transfer_d.values, ok.group.values, "all approaches")
    rows.append(r)
    preds.loc[ok.index, "pred_from_proxy"] = p
    ok2 = ok[ok.cal_max_cz < 0.04]
    r, _ = summarise("log10 workload error proxy, no degraded coupler", np.log10(ok2.error_proxy_nll.values),
                     ok2.transfer_d.values, ok2.group.values, "all approaches")
    rows.append(r)

    r, p = summarise("calibrated Aer distortion (exact expectations)", runs.noisy_direct_d.values,
                     runs.transfer_d.values, runs.group.values, "all approaches")
    rows.append(r)
    preds["pred_from_aer"] = p
    ratio = runs.transfer_d / runs.noisy_direct_d
    rows.append(dict(predictor="ratio hardware d / Aer d (median)", scope="all approaches", n=len(runs),
                     pearson_r=np.nan, spearman_rho=np.nan, loo_r2=np.nan, loo_mae=float(ratio.median())))
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "distortion_predictors.csv"), index=False)
    preds.to_csv(os.path.join(OUT, "distortion_predictions.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
