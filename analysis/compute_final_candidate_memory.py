"""Delayed-input memory of the five frozen candidates for k = 1..120 (post-hoc).

  python analysis/compute_final_candidate_memory.py [--kmax 120] [--check]

Writes results/memory/.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import ONE_BODY, Reservoir, _zridge  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "memory")
SEEDS = [79001, 42, 101, 505, 707]
N_TRAIN, N_VAL, ALPHA = 1095, 365, 1e-6
NULL_DELAYS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 117]
NULL_REPS = 20


def probe(theta, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(theta)
    for j in range(theta.shape[1]):
        out[:, j] = theta[rng.permutation(len(theta)), j]
    return out


def r2(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def replay_features(c, u):
    res = Reservoir(2, GRAPHS[c["graph"]], c["J"], c["hx"], c["hy"], c["dt"], c["r"])
    S, F = res.maps(u, ONE_BODY)
    ends = np.arange(c["window"] - 1, len(u))
    return ends, res.rwp(S, F, ends, c["window"])


def mc_delay(ends, X, target, k):
    """Corrected MC for one channel and delay. target: the probe channel (or a permuted copy)."""
    ok = ends >= k
    e, Xk = ends[ok], X[ok]
    tr, va = e < N_TRAIN, e >= N_TRAIN
    pred = _zridge(Xk[tr], target[e[tr] - k], Xk[va], ALPHA)
    raw = r2(target[e[va] - k], pred)
    return raw, max(raw - 1 / (N_VAL - 1), 0.0)


def check_recorded():
    """Rule-2 representative RWP_H4_R2 (W=14): recorded MC_ch^{1:20} 1.1634 (SD 0.0861)."""
    rec = pd.read_csv(os.path.join(HERE, "..", "data", "design", "rule_representatives.csv"))
    r = rec[rec.candidate_key == "RWP_H4_R2"].iloc[0]
    J = {tuple(int(x) for x in k[1:]): v for k, v in json.loads(r.J_json).items()}
    c = dict(graph="H4", J=[J[e] for e in GRAPHS["H4"]], hx=r["hx"], hy=r["hy"], dt=r["dt"], r=int(r["original_r"]),
             window=int(r.window))
    df = load_data().iloc[:N_TRAIN + N_VAL]
    theta = encode(df, r.alpha, **training_scales(df))
    vals = []
    for s in SEEDS:
        u = probe(theta, s)
        ends, X = replay_features(c, u)
        vals.append(sum(mc_delay(ends, X, u[:, j], k)[1] for j in range(4) for k in range(1, 21)) / 4)
    print(f"RWP_H4_R2: recomputed {np.mean(vals):.4f} (SD {np.std(vals, ddof=1):.4f}), "
          f"recorded {r.MC_ch_mean:.4f} (SD {r.MC_ch_std:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmax", type=int, default=120)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        check_recorded()
    df = load_data().iloc[:N_TRAIN + N_VAL]            # 2026 is not used
    rows, null_rows, ident = [], [], {}
    for cand, c in CANDIDATES.items():
        ident[cand] = hashlib.sha256(json.dumps({k: c[k] for k in ("graph", "J", "hx", "hy", "dt", "r", "alpha",
                                                                   "window")}, sort_keys=True).encode()).hexdigest()[:12]
        theta = encode(df, c["alpha"], **training_scales(df))
        rng = np.random.default_rng(2026)
        for s in SEEDS:
            u = probe(theta, s)
            ends, X = replay_features(c, u)
            for j in range(4):
                for k in range(1, args.kmax + 1):
                    raw, corr = mc_delay(ends, X, u[:, j], k)
                    rows.append(dict(candidate=cand, window=c["window"], seed=s, channel=j, delay=k,
                                     mc_raw=raw, mc_corr=corr))
                for rep in range(NULL_REPS):
                    shuffled = u[rng.permutation(len(u)), j]
                    for k in NULL_DELAYS:
                        null_rows.append(dict(candidate=cand, seed=s, channel=j, rep=rep, delay=k,
                                              mc_corr=mc_delay(ends, X, shuffled, k)[1]))
        print(f"C{cand} W={c['window']} done", flush=True)

    by = pd.DataFrame(rows)
    by.to_csv(os.path.join(OUT, "memory_by_seed.csv"), index=False)
    per_seed = by.groupby(["candidate", "window", "seed", "delay"]).mc_corr.mean().rename("mc_k").reset_index()
    curve = per_seed.groupby(["candidate", "window", "delay"]).mc_k.agg(mc_mean="mean", mc_sd="std").reset_index()
    curve.to_csv(os.path.join(OUT, "memory_curve.csv"), index=False)

    null = pd.DataFrame(null_rows)
    null.to_csv(os.path.join(OUT, "memory_null.csv"), index=False)
    null_stat = null.groupby(["candidate", "rep", "delay"]).mc_corr.mean()      # mean over seeds and channels

    def horizon(cv, tau):
        above = cv.mc_mean.values > tau
        return int(np.argmin(above)) if not above.all() else int(len(above))

    summ = []
    for cand, c in CANDIDATES.items():
        cv = curve[curve.candidate == cand].sort_values("delay")
        ns = null_stat.loc[cand]
        ch20 = per_seed[(per_seed.candidate == cand) & (per_seed.delay <= 20)].groupby("seed").mc_k.sum()
        t99, t95 = float(np.quantile(ns, .99)), float(np.quantile(ns, .95))
        summ.append(dict(candidate=cand, window=c["window"], mc_ch_1_20=ch20.mean(), mc_ch_1_20_sd=ch20.std(ddof=1),
                         mc_ch_1_kmax=per_seed[per_seed.candidate == cand].groupby("seed").mc_k.sum().mean(),
                         tau99=t99, k_eff=horizon(cv, t99), tau95=t95, k_eff_95=horizon(cv, t95),
                         k_eff_001=horizon(cv, 0.01), mc_at_kmax=float(cv.mc_mean.iloc[-1]), config_id=ident[cand]))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "memory_summary.csv"), index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
