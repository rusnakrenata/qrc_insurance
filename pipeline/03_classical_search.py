"""Hyperparameter search of the classical models on 2022-2024.

  python pipeline/03_classical_search.py --family ridge|glm|xgboost|esn|rnn|gru
  python pipeline/03_classical_search.py --verify

Writes results/classical_search/. The selected configurations are in configs/classical_selected.json.
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
from classical import models as M  # noqa: E402
from qrc.candidates import load_data  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "classical_search")
FROZEN = os.path.join(HERE, "..", "configs", "classical_selected.json")
RIDGE_ALPHAS = [1, 2, 3, 5, 6, 6.25, 6.5, 6.6, 6.75, 7, 7.5, 8, 9, 10, 15, 20, 25, 30, 50, 75, 100]


def ridge_cv(train, feature, alpha):
    errs = []
    for tr, va in M.folds(len(train)):
        p, _ = M.ridge_fit_predict(train.iloc[tr], train.iloc[va], feature, alpha)
        errs.append(M.rmse(p, train[M.Y].iloc[va]))
    return dict(feature_set=feature, alpha=alpha, cv_rmse=float(np.mean(errs)), cv_std=float(np.std(errs)))


def glm_cv(train, feature, variant, family):
    errs = []
    for tr, va in M.folds(len(train)):
        p, _ = M.glm_fit_predict(train.iloc[tr], train.iloc[va], feature, variant, family)
        errs.append(M.rmse(p, train[M.Y].iloc[va]))
    return dict(family=family, feature_set=feature, variant=variant, cv_rmse=float(np.mean(errs)),
                cv_std=float(np.std(errs)))


def xgb_cv(train, base, trees=(50, 150, 400)):
    """One fit with the largest tree count per fold; smaller counts read from the same ensemble."""
    X, y = train[M.FEATURES[base["feature_set"]]].to_numpy(float), train[M.Y].to_numpy(float)
    errs = {n: [] for n in trees}
    for tr, va in M.folds(len(X)):
        model = M.xgb_model(dict(base, n_estimators=max(trees))).fit(X[tr], y[tr])
        for n in trees:
            errs[n].append(M.rmse(model.predict(X[va], iteration_range=(0, n)), y[va]))
    return [dict(base, n_estimators=n, cv_rmse=float(np.mean(e)), cv_std=float(np.std(e))) for n, e in errs.items()]


def esn_cv(data, cfg, seed=42):
    arrays, _ = M.esn_prepare(data, cfg["feature_set"], int(cfg["window_length"]))
    W, W_in = M.esn_reservoir(int(cfg["reservoir_size"]), len(M.FEATURES[cfg["feature_set"]]), seed,
                              cfg["requested_connectivity"])
    return dict(cfg, **M.esn_cv(arrays, W, W_in, cfg))


def search(family, data):
    train = data.iloc[:M.N_TRAIN]
    rows = []
    if family == "ridge":
        rows = [ridge_cv(train, f, a) for f in M.FEATURES for a in RIDGE_ALPHAS]
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["feature_set"], r["alpha"])
    elif family == "glm":
        rows = [glm_cv(train, f, v, fam) for fam in ("poisson", "nb") for f, v in M.GLM_SPECS]
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["feature_set"], r["variant"])
    elif family == "xgboost":
        grid = itertools.product(M.FEATURES, ["reg:squarederror", "count:poisson"], [1, 2, 3], [.03, .1], [1., 10.], [1., 10.])
        for f, o, d, lr, mcw, lam in grid:
            rows += xgb_cv(train, dict(feature_set=f, objective=o, max_depth=d, learning_rate=lr,
                                       min_child_weight=mcw, reg_lambda=lam))
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["max_depth"], r["n_estimators"], r["feature_set"],
                         r["objective"], r["learning_rate"], r["min_child_weight"], r["reg_lambda"])
    elif family == "esn":
        for f, w in itertools.product(M.FEATURES, M.WINDOWS):
            arrays, _ = M.esn_prepare(data, f, w)
            for size, p in itertools.product([25, 50, 100, 200], [.05, .1, .5]):
                W, W_in = M.esn_reservoir(size, len(M.FEATURES[f]), 42, p)
                for rho, sc, leak in itertools.product([.3, .5, .7, .9], [.25, .5, .75, 1., 2.], [.25, .5, .75, 1.]):
                    cfg = dict(feature_set=f, window_length=w, reservoir_size=size, requested_connectivity=p,
                               spectral_radius=rho, input_scaling=sc, leak_rate=leak)
                    rows.append(dict(cfg, **M.esn_cv(arrays, W, W_in, cfg)))
            print(f"ESN {f} W={w} done", flush=True)
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["reservoir_size"], r["feature_set"], r["window_length"],
                         r["requested_connectivity"], r["spectral_radius"], r["input_scaling"], r["leak_rate"],
                         r["ridge_alpha"])
    elif family == "rnn":
        # screening: one-layer networks, fixed optimizer; then tune the three best feature/window branches
        screen = [M.neural_cv("rnn", data, dict(feature_set=f, window=w, hidden_size=h, learning_rate=.001,
                                                 weight_decay=1e-4, layers=1))
                  for f, w, h in itertools.product(M.FEATURES, M.WINDOWS, [8, 16, 32, 64])]
        pd.DataFrame(screen).to_csv(os.path.join(OUT, "rnn_screening.csv"), index=False)
        branches = sorted(screen, key=lambda r: (r["cv_rmse"], r["cv_std"]))[:3]
        for b in branches:
            for h, lr, wd, layers in itertools.product([8, 32, 64], [3e-4, 1e-3, 3e-3], [1e-5, 1e-4, 1e-3], [1, 2]):
                rows.append(M.neural_cv("rnn", data, dict(feature_set=b["feature_set"], window=b["window"],
                                                          hidden_size=h, learning_rate=lr, weight_decay=wd,
                                                          layers=layers)))
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["hidden_size"], r["layers"], r["learning_rate"],
                         r["weight_decay"])
    elif family == "gru":
        rows = [M.neural_cv("gru", data, dict(feature_set=f, window=w, hidden_size=h, layers=1,
                                              learning_rate=.001, weight_decay=1e-4))
                for f, w, h in itertools.product(M.FEATURES, M.WINDOWS, [1, 2, 4, 8, 16, 32])]
        key = lambda r: (r["cv_rmse"], r["cv_std"], r["feature_set"], r["window"], r["hidden_size"])
    rows = sorted(rows, key=key)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{family}.csv"), index=False)
    if family == "glm":
        top = {fam: [r for r in rows if r["family"] == fam][:3] for fam in ("poisson", "nb")}
    elif family == "xgboost":
        top = {f"xgboost_{o.split(':')[1].replace('squarederror', 'squared')}": [r for r in rows if r["objective"] == o][:3]
               for o in ("reg:squarederror", "count:poisson")}
    else:
        top = {family: rows[:3]}
    M.dump(top, os.path.join(OUT, f"{family}_top3.json"))
    print(json.dumps(top, indent=1, default=float))


def verify(data):
    """Recompute the CV score of every frozen configuration and compare with the recorded value."""
    frozen = json.load(open(FROZEN))
    train = data.iloc[:M.N_TRAIN]
    rows = []
    for c in frozen["ridge"]:
        rows.append(("ridge", c["cv_rmse"], ridge_cv(train, c["feature_set"], c["alpha"])["cv_rmse"]))
    for fam in ("poisson", "nb"):
        for c in frozen[fam]:
            rows.append((fam, c["cv_rmse"], glm_cv(train, c["feature_set"], c["variant"], fam)["cv_rmse"]))
    for fam in ("xgboost_squared", "xgboost_poisson"):
        for c in frozen[fam]:
            base = {k: c[k] for k in ("feature_set", "objective", "max_depth", "learning_rate", "min_child_weight",
                                      "reg_lambda")}
            got = next(r for r in xgb_cv(train, base) if r["n_estimators"] == c["n_estimators"])
            rows.append((fam, c["cv_rmse"], got["cv_rmse"]))
    for c in frozen["esn"]:
        rows.append(("esn", c["cv_rmse"], esn_cv(data, c)["cv_rmse"]))
    for kind in ("gru", "rnn"):
        for c in frozen[kind][:1]:                                   # rank 1 only: neural CV is slow
            cfg = {k: c[k] for k in ("feature_set", "window", "hidden_size", "layers", "learning_rate", "weight_decay")}
            got = M.neural_cv(kind, data, cfg)
            rows.append((kind, c["cv_rmse"], got["cv_rmse"]))
            print(kind, "epochs recorded/recomputed", c["epochs"], got["epochs"], flush=True)
    out = pd.DataFrame(rows, columns=["family", "recorded", "recomputed"])
    out["abs_diff"] = (out.recorded - out.recomputed).abs()
    out.to_csv(os.path.join(OUT, "verification.csv"), index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["ridge", "glm", "xgboost", "esn", "rnn", "gru"])
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from threadpoolctl import threadpool_limits
    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
    data = load_data().iloc[:1460].reset_index(drop=True)        # 2026 is not loaded at all
    with threadpool_limits(limits=1):
        if args.verify:
            verify(data)
        elif args.family:
            search(args.family, data)
        else:
            ap.error("choose --family or --verify")


if __name__ == "__main__":
    main()
