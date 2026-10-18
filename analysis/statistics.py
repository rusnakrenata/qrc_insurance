"""Forecasting comparison, hardware summaries and paired bootstrap tests.

  python analysis/statistics.py

Writes results/tables/.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
BLOCK = 14
B = 4000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")


def rmse(e, axis=-1):
    return np.sqrt(np.mean(np.square(e), axis=axis))


def block_indices(n, block=BLOCK):
    """One circular-block bootstrap sample of date indices."""
    starts = RNG.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def ci(values):
    return np.percentile(values, [2.5, 97.5])


def classical_table(classical_dir, ideal):
    """The configuration of every classical family selected on 2025, on the 232 test dates."""
    m = pd.read_csv(os.path.join(classical_dir, "metrics.csv"))
    rows, preds = [], {}
    for method, g in m.groupby("method"):
        P = np.array([pd.read_csv(os.path.join(classical_dir, f"{i}_predictions.csv")).prediction.values for i in g.id])
        preds[method] = P
        rows.append(dict(method=method, cv_rank=int(g.cv_rank.iloc[0]), n_seeds=len(g), rmse_mean=g.rmse.mean(),
                         rmse_sd=g.rmse.std(ddof=1) if len(g) > 1 else np.nan, mae=g.mae.mean(), bias=g.bias.mean(),
                         crmse=float(np.mean([np.std(p - ideal.actual.values) for p in P]))))
    return pd.DataFrame(rows).sort_values("rmse_mean"), preds


def paired_vs_reference(P_model, P_ref, y, block=BLOCK):
    """Mean RMSE difference (model - reference) with a circular-block bootstrap CI over dates."""
    n = len(y)
    base = rmse(P_model - y).mean() - rmse(P_ref - y).mean()
    boot = []
    for _ in range(B):
        i = block_indices(n, block)
        boot.append(rmse(P_model[:, i] - y[i]).mean() - rmse(P_ref[:, i] - y[i]).mean())
    lo, hi = ci(boot)
    return base, lo, hi


def half_split(P, y):
    h = len(y) // 2
    return rmse(P[:, :h] - y[:h]).mean(), rmse(P[:, h:] - y[h:]).mean()


def paired_rounds(daily, a, b, metric, block=BLOCK):
    """Two-level bootstrap of the mean paired difference a - b over rounds.

    metric: 'rmse' (forecast error vs observations) or 'd' (distance to ideal).
    Rounds are resampled within candidate, dates in circular blocks.
    """
    keys = sorted(set(map(tuple, daily[["candidate", "round"]].drop_duplicates().values)))
    arr = {}
    for (c, r) in keys:
        g = daily[(daily.candidate == c) & (daily["round"] == r)]
        y = g[g.approach == a].actual.values
        if metric == "rmse":
            ea = g[g.approach == a].pred_qpu.values - y
            eb = g[g.approach == b].pred_qpu.values - y
        else:
            ea = g[g.approach == a].delta.values
            eb = g[g.approach == b].delta.values
        arr[(c, r)] = (ea, eb)
    n = len(next(iter(arr.values()))[0])
    diffs = np.array([rmse(ea) - rmse(eb) for ea, eb in arr.values()])
    by_cand = {}
    for (c, r) in keys:
        by_cand.setdefault(c, []).append((c, r))
    boot = []
    for _ in range(B):
        i = block_indices(n, block)
        vals = []
        for c, ks in by_cand.items():
            pick = RNG.integers(0, len(ks), len(ks))
            for p in pick:
                ea, eb = arr[ks[p]]
                vals.append(rmse(ea[i]) - rmse(eb[i]))
        boot.append(np.mean(vals))
    lo, hi = ci(boot)
    wins = int(np.sum(diffs < 0))
    # exact two-sided sign test on rounds (rounds treated as independent replications)
    from math import comb
    k, m = min(wins, len(diffs) - wins), len(diffs)
    p_sign = min(1.0, 2 * sum(comb(m, j) for j in range(k + 1)) / 2 ** m)
    return dict(comparison=f"{a} vs {b}", metric=metric, rounds=len(diffs), a_better=wins,
                median_diff=float(np.median(diffs)), mean_diff=float(diffs.mean()),
                ci_low=lo, ci_high=hi, sign_test_p=p_sign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", default=os.path.join(HERE, "..", "data", "classical_predictions"))
    ap.add_argument("--ibm-final", default=os.path.join(HERE, "..", "data", "ibm_kingston_runs"))
    a = ap.parse_args()
    tdir = os.path.join(OUT, "tables")
    os.makedirs(tdir, exist_ok=True)
    runs = pd.read_csv(os.path.join(OUT, "runs.csv"))
    daily = pd.read_csv(os.path.join(OUT, "daily.csv"))

    # ---- Objective 1: forecasting
    ideal_files = {c: sorted(glob.glob(os.path.join(a.ibm_final, f"candidate{c}", "*", "ideal_predictions.csv")))[-1]
                   for c in range(1, 6)}
    ideal = {c: pd.read_csv(f) for c, f in ideal_files.items()}
    y = ideal[1].actual.values
    for c in ideal:
        assert np.allclose(ideal[c].actual.values, y)
    ctab, cpred = classical_table(a.classical, ideal[1])
    ctab.to_csv(os.path.join(tdir, "classical.csv"), index=False)

    models = {f"QRC-C{c} ideal": ideal[c].prediction.values[None, :] for c in ideal}
    for c in range(1, 6):   # hardware runs with HA layouts (one row per round)
        g = daily[(daily.candidate == c) & (daily.approach == "hardware-aware")]
        models[f"QRC-C{c} HA hardware"] = np.array([x.pred_qpu.values for _, x in g.groupby("round")])
    for meth in ["ridge", "nb", "esn", "gru", "rnn", "xgboost_squared", "poisson", "xgboost_poisson"]:
        models[meth] = cpred[meth]
    rows = []
    for name, P in models.items():
        e = P - y
        h1, h2 = half_split(P, y)
        row = dict(model=name, replicates=len(P), rmse=rmse(e).mean(), mae=np.abs(e).mean(),
                   bias=e.mean(), crmse=np.std(e, axis=1).mean(), rmse_first_half=h1, rmse_second_half=h2)
        for ref in ["ridge", "xgboost_squared"]:
            d, lo, hi = paired_vs_reference(P, cpred[ref], y)
            row.update({f"diff_vs_{ref}": d, f"diff_vs_{ref}_lo": lo, f"diff_vs_{ref}_hi": hi})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "forecasting_comparison.csv"), index=False)

    # ---- Objective 2: hardware
    summ = runs.groupby(["candidate", "approach"]).agg(
        rounds=("round", "count"), qpu_rmse_mean=("qpu_rmse", "mean"), qpu_rmse_sd=("qpu_rmse", "std"),
        qpu_rmse_max=("qpu_rmse", "max"), d_mean=("transfer_d", "mean"), d_sd=("transfer_d", "std"),
        d_median=("transfer_d", "median"), d_max=("transfer_d", "max"), bias_mean=("qpu_bias", "mean"),
        level_share_median=("level_share", "median"), gap_median=("gap_median", "median"),
        ideal_rmse=("ideal_rmse", "first")).reset_index()
    summ.to_csv(os.path.join(tdir, "hardware_by_candidate.csv"), index=False)

    pooled = runs.groupby("approach").agg(
        runs=("round", "count"), qpu_rmse_median=("qpu_rmse", "median"), qpu_rmse_mean=("qpu_rmse", "mean"),
        qpu_rmse_max=("qpu_rmse", "max"), d_median=("transfer_d", "median"), d_mean=("transfer_d", "mean"),
        d_p90=("transfer_d", lambda s: s.quantile(0.9)), d_max=("transfer_d", "max"),
        severe_d_over_8=("transfer_d", lambda s: int((s > 8).sum())),
        d_over_6=("transfer_d", lambda s: int((s > 6).sum())),
        level_share_median=("level_share", "median"),
        noisy_direct_d_median=("noisy_direct_d", "median"), noisy_sampled_d_median=("noisy_sampled_d", "median"),
        flagged_coupler=("cal_max_cz", lambda s: int((s > 0.04).sum()))).reset_index()
    # within-candidate spread of the transfer distortion, averaged over candidates with >=3 rounds
    spread = runs.groupby(["approach", "candidate"]).transfer_d.agg(lambda s: s.max() - s.min()).groupby("approach").median()
    pooled["d_range_within_candidate_median"] = pooled.approach.map(spread)
    pooled.to_csv(os.path.join(tdir, "hardware_pooled.csv"), index=False)

    pairs = [paired_rounds(daily, "hardware-aware", b, m) for b in ("random", "qiskit") for m in ("rmse", "d")]
    pairs += [paired_rounds(daily, "random", "qiskit", m) for m in ("rmse", "d")]
    pd.DataFrame(pairs).to_csv(os.path.join(tdir, "paired_layouts.csv"), index=False)

    # sensitivity of the intervals to the block length
    sens = []
    for blk in (7, 14, 28):
        for name in ["QRC-C%d ideal" % c for c in range(1, 6)]:
            d, lo, hi = paired_vs_reference(models[name], cpred["ridge"], y, blk)
            sens.append(dict(block=blk, comparison=name + " vs ridge", diff=d, ci_low=lo, ci_high=hi))
        for b in ("random", "qiskit"):
            r = paired_rounds(daily, "hardware-aware", b, "rmse", blk)
            sens.append(dict(block=blk, comparison="HA vs " + b + " (rmse)", diff=r["mean_diff"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"]))
    pd.DataFrame(sens).to_csv(os.path.join(tdir, "block_sensitivity.csv"), index=False)

    # post-hoc but outcome-blind: random layouts with and without a degraded coupler (>4 % CZ error)
    rnd = runs[runs.approach == "random"].copy()
    rnd["flag"] = rnd.cal_max_cz > 0.04
    rnd.groupby("flag").agg(n=("round", "count"), d_median=("transfer_d", "median"), d_max=("transfer_d", "max"),
                            rmse_median=("qpu_rmse", "median")).to_csv(os.path.join(tdir, "random_by_coupler_flag.csv"))
    print(pd.read_csv(os.path.join(tdir, "hardware_pooled.csv")).T)
    print(pd.DataFrame(pairs))


if __name__ == "__main__":
    main()
