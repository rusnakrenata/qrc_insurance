"""Select one configuration per classical family on 2025, then forecast 2026 with it.

  python pipeline/04_classical_2026.py [--methods esn,rnn] [--compare data/classical_predictions]

Every model is fitted on 2022-2024. The three CV-ranked configurations of each family
(configs/classical_selected.json) are scored on 2025; the one with the lowest 2025 RMSE
(mean over seeds for ESN, RNN and GRU) is written to configs/classical_final.json and is
the only one evaluated on 2026.

Writes results/classical_validation/ and results/classical/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from classical import models as M  # noqa: E402
from qrc.candidates import load_data  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FROZEN = os.path.join(HERE, "..", "configs", "classical_selected.json")
FINAL = os.path.join(HERE, "..", "configs", "classical_final.json")
VAL = np.arange(M.N_TRAIN, 1460)
TEST = np.arange(1460, 1692)
METHODS = ["baseline", "ridge", "poisson", "nb", "esn", "rnn", "gru", "xgboost_squared", "xgboost_poisson"]


def seeds(method):
    return {"esn": M.SEEDS, "rnn": M.NEURAL_SEEDS, "gru": M.NEURAL_SEEDS}.get(
        method, [42] if method.startswith("xgboost") else [None])


def forecast(method, cfg, seed, data, ends):
    train = data.iloc[:M.N_TRAIN]
    if method == "baseline":
        return {"seasonal_naive_7d": data[M.C].values[ends - 6],          # C_{t-6} for C_{t+1}
                "training_mean": np.full(len(ends), train[M.Y].mean()),
                "persistence": data[M.C].values[ends]}[cfg["name"]].astype(float)
    if method == "ridge":
        return M.ridge_fit_predict(train, data.iloc[ends], cfg["feature_set"], cfg["alpha"])[0]
    if method in ("poisson", "nb"):
        return M.glm_fit_predict(train, data.iloc[ends], cfg["feature_set"], cfg["variant"], method)[0]
    if method == "esn":
        # the readout alpha is re-chosen by training CV for every reservoir seed
        arrays, _ = M.esn_prepare(data, cfg["feature_set"], int(cfg["window_length"]))
        W, W_in = M.esn_reservoir(int(cfg["reservoir_size"]), len(M.FEATURES[cfg["feature_set"]]), seed,
                                  cfg["requested_connectivity"])
        cfg = dict(cfg, ridge_alpha=M.esn_cv(arrays, W, W_in, cfg)["ridge_alpha"])
        return M.esn_fit_predict(data, cfg, seed, ends)
    if method in ("rnn", "gru"):
        return M.neural_fit_predict(method, data, cfg, seed, ends)
    cols = M.FEATURES[cfg["feature_set"]]
    model = M.xgb_model(cfg).fit(train[cols].to_numpy(float), train[M.Y].to_numpy(float))
    return model.predict(data.iloc[ends][cols].to_numpy(float)).astype(float)


def scores(p, y):
    e = p - y
    return dict(rmse=M.rmse(p, y), mae=float(np.abs(e).mean()), bias=float(e.mean()))


def validate(methods, data):
    """Score the three ranked configurations of each family on 2025 and keep the best."""
    inputs = data.copy()
    inputs.loc[VAL[0]:, M.Y] = np.nan                  # nothing after 2024 is visible to a fit
    y = data[M.Y].values[VAL].astype(float)
    frozen = json.load(open(FROZEN))
    rows, final = [], {}
    for method in methods:
        for rank, cfg in enumerate(frozen[method], 1):
            for s in seeds(method):
                rows.append(dict(method=method, rank=rank, seed=s,
                                 **scores(forecast(method, cfg, s, inputs, VAL), y)))
            print(method, rank, round(np.mean([r["rmse"] for r in rows if r["method"] == method
                                               and r["rank"] == rank]), 4), flush=True)
    table = pd.DataFrame(rows)
    summary = (table.groupby(["method", "rank"]).agg(rmse_2025=("rmse", "mean"), rmse_2025_sd=("rmse", "std"),
                                                     mae_2025=("mae", "mean"), bias_2025=("bias", "mean"),
                                                     n_seeds=("rmse", "size")).reset_index())
    best = summary.sort_values(["method", "rmse_2025", "rank"]).groupby("method").head(1)
    summary["selected"] = summary.set_index(["method", "rank"]).index.isin(best.set_index(["method", "rank"]).index)
    for r in best.itertuples():
        final[r.method] = dict(frozen[r.method][r.rank - 1], cv_rank=int(r.rank), rmse_2025=float(r.rmse_2025))
    return table, summary, final


def test(final, data, compare=None):
    inputs = data.copy()
    inputs.loc[TEST[0]:, M.Y] = np.nan                 # 2026 targets are never visible to a model
    y = data[M.Y].values[TEST].astype(float)
    out = os.path.join(RES, "classical")
    rows = []
    for method, cfg in final.items():
        for s in seeds(method):
            rid = f"{method}_seed{s}"
            p = forecast(method, cfg, s, inputs, TEST)
            row = dict(id=rid, method=method, cv_rank=cfg["cv_rank"], seed=s, **scores(p, y))
            pd.DataFrame(dict(target_date=data.target_date.values[TEST], actual=y, prediction=p)).to_csv(
                os.path.join(out, f"{rid}_predictions.csv"), index=False)
            if compare:
                ref = pd.read_csv(os.path.join(compare, f"{rid}_predictions.csv")).prediction.values
                row["max_abs_diff_vs_original"] = float(np.max(np.abs(ref - p)))
            rows.append(row)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--compare", help="folder with reference 2026 predictions")
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits
    methods = METHODS if args.methods == "all" else args.methods.split(",")
    vdir = os.path.join(RES, "classical_validation")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(os.path.join(RES, "classical"), exist_ok=True)
    data = load_data()
    with threadpool_limits(limits=1):
        runs, summary, final = validate(methods, data)
        runs.to_csv(os.path.join(vdir, "validation_runs.csv"), index=False)
        summary.to_csv(os.path.join(vdir, "validation_2025.csv"), index=False)
        print(summary.round(4).to_string(index=False))
        if args.methods == "all":
            M.dump({m: final[m] for m in METHODS}, FINAL)
        else:
            final = {m: c for m, c in json.load(open(FINAL)).items() if m in methods} | final
        metrics = test(final, data, args.compare)
    metrics.to_csv(os.path.join(RES, "classical", "metrics.csv"), index=False)
    print(metrics.groupby("method")[["rmse"]].agg(["mean", "std", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
