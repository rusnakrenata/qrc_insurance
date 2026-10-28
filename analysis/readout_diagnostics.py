"""Frozen versus refitted readouts (Sec. II-C, VI-B, VI-C).

For every stored run, the 2026 features are read out by
  frozen     the deployed ridge readout, fitted on ideal features of 2022-2024
  affine     the frozen readout after a per-observable linear map from QPU to ideal features
             (fitted on the probe dates for HA runs, and on the 2026 days as an upper bound)
  refitted   ridge readouts fitted on QPU, ideal and classical-input features of the same
             2026 days (4 contiguous blocks, each predicted by a readout trained on the others)
It also reports standardized readout weights (claims/day per SD of the observable),
drop-column importance, and the memory test in simulation (readouts refitted on W = 1 features).

Usage:  python analysis/readout_diagnostics.py
Writes results/readout_diagnostics/; headline.json holds the numbers quoted in the article.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG = ROOT
sys.path.insert(0, ROOT)
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

RUNS = os.path.join(PKG, "data", "ibm_kingston_runs")
OUT = os.path.join(ROOT, "results", "readout_diagnostics")
TRAIN, VAL, TEST = np.arange(0, 1095), np.arange(1095, 1460), np.arange(1460, 1692)
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
N_BLOCKS = 4
HIGH = 0.20
MEMORY_OBS = {"YX45"}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def ridge(X, y, lam):
    """Closed-form ridge on standardised columns (as qrc.readout.fit_ridge); constant columns get weight 0.
    Returns raw weights w, intercept b and standardised coefficients beta (claims/day per SD)."""
    mu, sd = X.mean(0), X.std(0)
    ok = sd > 1e-12
    Z = (X[:, ok] - mu[ok]) / sd[ok]
    beta_ok = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    beta = np.zeros(X.shape[1])
    beta[ok] = beta_ok
    w = np.where(ok, beta / np.where(ok, sd, 1), 0.0)
    return w, y.mean() - mu @ w, beta


def blocks(n, k=N_BLOCKS):
    edges = np.linspace(0, n, k + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def pick_lambda(X, y, idx_blocks):
    """Leave-one-block-out over the given blocks."""
    best, best_err = None, np.inf
    for lam in LAMBDAS:
        errs = []
        for i, te in enumerate(idx_blocks):
            tr = np.concatenate([b for j, b in enumerate(idx_blocks) if j != i])
            w, b, _ = ridge(X[tr], y[tr], lam)
            errs.append(np.mean((X[te] @ w + b - y[te]) ** 2))
        if np.mean(errs) < best_err:
            best, best_err = lam, np.mean(errs)
    return best


def blocked_cv_predict(X, y):
    """Out-of-block predictions over 2026 with nested penalty choice."""
    bl = blocks(len(y))
    pred = np.empty(len(y))
    for i, te in enumerate(bl):
        inner = [b for j, b in enumerate(bl) if j != i]
        lam = pick_lambda(X, y, inner)
        tr = np.concatenate(inner)
        w, b, _ = ridge(X[tr], y[tr], lam)
        pred[te] = X[te] @ w + b
    return pred


def drop_column_importance(X, y, names):
    base = rmse(blocked_cv_predict(X, y) - y)
    return {n: rmse(blocked_cv_predict(np.delete(X, j, axis=1), y) - y) - base for j, n in enumerate(names)}


def full_fit_beta(X, y):
    lam = pick_lambda(X, y, blocks(len(y)))
    return ridge(X, y, lam)[2], lam


def classical_inputs(df):
    """F4 inputs in plain form (claims and policies z-scored, weekday on the circle, holiday flag)."""
    s = training_scales(df)
    return np.column_stack([(df.property_damage_claim_count_t - s["mean_c"]) / s["std_c"],
                            (df.active_policy_count_t - s["mean_p"]) / s["std_p"],
                            np.sin(2 * np.pi * df.target_date.dt.dayofweek / 7),
                            np.cos(2 * np.pi * df.target_date.dt.dayofweek / 7),
                            df.is_public_holiday_t_plus_1.astype(float)])


# simulation: frozen readouts and memory test

def simulation(df, y):
    rows, frozen, mem = {}, {}, []
    for k, c in CANDIDATES.items():
        theta = encode(df, c["alpha"], **training_scales(df))
        A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
        names = c["features"]
        if c["protocol"] == "CONT":
            Xtr, ytr = cont_features(A[:1095], names), y[TRAIN]
        else:
            fit = np.arange(c["window"] - 1, 1095)
            Xtr, ytr = rwp_features(A, fit, c["window"], names), y[fit]
        w, b, beta = ridge(Xtr, ytr, c["lam"])
        ideal = pd.read_csv(os.path.join(PKG, "results", "ideal", f"candidate{k}_predictions.csv"))
        Xdep = ideal[names].values                              # deployed features, 2025 + 2026
        assert np.allclose(Xdep @ w + b, ideal.prediction.values, atol=1e-6), f"C{k} frozen readout mismatch"
        frozen[k] = dict(names=names, w=w, b=b, beta=beta, lam=c["lam"], train_X=Xtr)
        yv, yt = y[VAL], y[TEST]
        pv, pt = Xdep[:365] @ w + b, Xdep[365:] @ w + b
        # W = 1: no input history, same penalty
        X1 = rwp_features(A, np.arange(len(y)), 1, names)
        w1, b1, _ = ridge(X1[TRAIN], y[TRAIN], c["lam"])
        mem.append(dict(candidate=k, variant=f"deployed (W={c['window']}, {c['protocol']} readout)",
                        rmse_2025=rmse(pv - yv), rmse_2026=rmse(pt - yt)))
        mem.append(dict(candidate=k, variant="W=1, readout refitted",
                        rmse_2025=rmse(X1[VAL] @ w1 + b1 - yv), rmse_2026=rmse(X1[TEST] @ w1 + b1 - yt)))
        for obs in MEMORY_OBS & set(names):
            j = names.index(obs)
            keep = [i for i in range(len(names)) if i != j]
            wd, bd, _ = ridge(Xtr[:, keep], ytr, c["lam"])
            mem.append(dict(candidate=k, variant=f"deployed without {obs}, readout refitted",
                            rmse_2025=rmse(Xdep[:365, keep] @ wd + bd - yv),
                            rmse_2026=rmse(Xdep[365:, keep] @ wd + bd - yt)))
        rows[k] = A
    m = pd.DataFrame(mem)
    m["delta_2025_vs_deployed"] = m.rmse_2025 - m.groupby("candidate").rmse_2025.transform("first")
    m["delta_2026_vs_deployed"] = m.rmse_2026 - m.groupby("candidate").rmse_2026.transform("first")
    return frozen, m


# one hardware run

def affine_map(zq_fit, zi_fit):
    """Per observable z_ideal ~ a z_qpu + b (least squares)."""
    a = np.ones(zq_fit.shape[1])
    b = np.zeros(zq_fit.shape[1])
    for j in range(zq_fit.shape[1]):
        if zq_fit[:, j].std() > 1e-12:
            a[j], b[j] = np.polyfit(zq_fit[:, j], zi_fit[:, j], 1)
    return a, b


def analyse_run(k, policy, rnd, d26, probe, fr, y, Xclass, label):
    names = fr["names"]
    Zi = d26[[n + "_ideal" for n in names]].values
    Zq = d26[[n + "_qpu" for n in names]].values
    ends = d26.endpoint.values
    yt = y[ends]
    w, b = fr["w"], fr["b"]
    p_ideal, p_q = Zi @ w + b, Zq @ w + b
    out = dict(label=label, candidate=k, policy=policy, round=rnd, n=len(yt),
               frozen_rmse_ideal=rmse(p_ideal - yt), frozen_rmse_qpu=rmse(p_q - yt),
               frozen_d=rmse(p_q - p_ideal), frozen_shift=float(np.mean(p_q - p_ideal)))
    out["frozen_d_noshift"] = float(np.std(p_q - p_ideal))

    # per-observable linear map: on 2026 (upper bound) and on the probe dates
    a, c0 = affine_map(Zq, Zi)
    p_aff = (Zq * a + c0) @ w + b
    out.update(affine_oracle_rmse=rmse(p_aff - yt), affine_oracle_d=rmse(p_aff - p_ideal))
    if probe is not None:
        Pi = probe[[n + "_ideal" for n in names]].values
        Pq = probe[[n + "_qpu" for n in names]].values
        a, c0 = affine_map(Pq, Pi)
        p_affp = (Zq * a + c0) @ w + b
        out.update(affine_probe_rmse=rmse(p_affp - yt), affine_probe_d=rmse(p_affp - p_ideal))
        yp = y[probe.endpoint.values]
        lam = CANDIDATES[k]["lam"]
        wq, bq, _ = ridge(Pq, yp, lam)
        wi, bi, _ = ridge(Pi, yp, lam)
        pq_probe, pi_probe = Zq @ wq + bq, Zi @ wi + bi
        out.update(probe_ridge_rmse_qpu=rmse(pq_probe - yt), probe_ridge_rmse_ideal_control=rmse(pi_probe - yt),
                   probe_ridge_d=rmse(pq_probe - pi_probe))

    # readouts refitted on 2026 (blocked CV), QPU and controls
    pq_cv = blocked_cv_predict(Zq, yt)
    pi_cv = blocked_cv_predict(Zi, yt)
    pc_cv = blocked_cv_predict(Xclass[ends], yt)
    out.update(qpu_cv_rmse=rmse(pq_cv - yt), ideal_cv_rmse=rmse(pi_cv - yt), classical_cv_rmse=rmse(pc_cv - yt),
               qpu_cv_d=rmse(pq_cv - pi_cv), qpu_cv_shift=float(np.mean(pq_cv - pi_cv)))
    out["hardware_effect_cv"] = out["qpu_cv_rmse"] - out["ideal_cv_rmse"]
    out["hardware_effect_frozen"] = out["frozen_rmse_qpu"] - out["frozen_rmse_ideal"]

    # weights and importance
    bq_full, lam_q = full_fit_beta(Zq, yt)
    bi_full, lam_i = full_fit_beta(Zi, yt)
    imp_q = drop_column_importance(Zq, yt, names)
    imp_i = drop_column_importance(Zi, yt, names)
    eta = [np.polyfit(Zi[:, j], Zq[:, j], 1)[0] if Zi[:, j].std() > 1e-12 else np.nan for j in range(len(names))]
    wrows = [dict(label=label, candidate=k, policy=policy, round=rnd, observable=n, memory=n in MEMORY_OBS,
                  eta=eta[j], beta_frozen=fr["beta"][j], beta_ideal2026=bi_full[j], beta_qpu2026=bq_full[j],
                  importance_ideal2026=imp_i[n], importance_qpu2026=imp_q[n])
             for j, n in enumerate(names)]
    return out, wrows


def load_probe(folder, layout):
    f = os.path.join(folder, "probe_daily_features.csv")
    if not os.path.exists(f):
        return None
    p = pd.read_csv(f)
    key = json.dumps([int(q) for q in layout]).replace(",", ", ")
    p = p[p.layout.str.replace(" ", "") == key.replace(" ", "")]
    return p if len(p) else None


def case(frozen_rel, qpu_rel):
    hf, hq = frozen_rel >= HIGH, qpu_rel >= HIGH
    return {(True, True): "A", (False, False): "B", (True, False): "C", (False, True): "D"}[(hf, hq)]


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="16_01 round folder with C4 full and w1 arms (default: latest complete)")
    ap.add_argument("--no-session", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    Xclass = classical_inputs(df)
    print("simulation: frozen readouts and memory test ...", flush=True)
    frozen, mem = simulation(df, y)
    mem.to_csv(os.path.join(args.out, "memory_in_simulation.csv"), index=False)

    runs = pd.read_csv(os.path.join(PKG, "results", "runs.csv"))
    results, weights = [], []
    for _, r in runs.iterrows():
        folder = os.path.join(RUNS, r.source)
        d26 = pd.read_csv(os.path.join(folder, "2026_features_predictions.csv"))
        layout = json.loads(r.layout)
        probe = load_probe(folder, layout) if r.approach == "hardware-aware" else None
        o, wr = analyse_run(r.candidate, r.approach, int(r["round"]), d26, probe, frozen[r.candidate], y, Xclass,
                            f"C{r.candidate}")
        assert abs(o["frozen_rmse_qpu"] - r.qpu_rmse) < 1e-6 and abs(o["frozen_d"] - r.transfer_d) < 1e-6, r.source
        results.append(o)
        weights += wr
        print(f"  C{r.candidate} r{r['round']} {r.approach:14s} frozen d {o['frozen_d']:.2f} -> qpu-cv d "
              f"{o['qpu_cv_d']:.2f}", flush=True)

    if not args.no_session:
        session = args.session
        if session is None:
            cands = sorted(glob.glob(os.path.join(ROOT, "data", "ibm_kingston_ablation", "candidate4", "*")))
            cands = [c for c in cands if all(os.path.exists(os.path.join(c, a, "2026_features_predictions.csv"))
                                             for a in ("full", "w1"))]
            session = cands[-1] if cands else None
        if session:
            for arm in ("full", "w1"):
                d26 = pd.read_csv(os.path.join(session, arm, "2026_features_predictions.csv"))
                o, wr = analyse_run(4, f"session-{arm}", 0, d26, None, frozen[4], y, Xclass, f"C4 {arm} (session)")
                results.append(o)
                weights += wr
                print(f"  C4 session {arm}: frozen d {o['frozen_d']:.2f} -> qpu-cv d {o['qpu_cv_d']:.2f}", flush=True)

    res = pd.DataFrame(results)
    W = pd.DataFrame(weights)
    res.to_csv(os.path.join(args.out, "per_run.csv"), index=False)
    W.to_csv(os.path.join(args.out, "weights_per_run.csv"), index=False)

    # medians per candidate and policy
    keep = ["frozen_rmse_ideal", "frozen_rmse_qpu", "frozen_d", "frozen_d_noshift", "affine_probe_d", "affine_oracle_d",
            "probe_ridge_rmse_qpu", "probe_ridge_rmse_ideal_control", "qpu_cv_rmse", "ideal_cv_rmse",
            "classical_cv_rmse", "qpu_cv_d", "hardware_effect_cv"]
    summ = res.groupby(["label", "policy"])[keep].median().reset_index()
    summ.insert(2, "runs", res.groupby(["label", "policy"]).size().values)
    summ.to_csv(os.path.join(args.out, "summary_by_candidate_policy.csv"), index=False)

    # numbers quoted in the article (stored runs)
    st = res[~res.policy.str.startswith("session")]
    per_cand = {}
    for c, g in st.groupby("candidate"):
        per_cand[f"C{c}"] = dict(
            frozen_d_median=float(g.frozen_d.median()), affine_2026_d_median=float(g.affine_oracle_d.median()),
            affine_removed_pct=float(100 * (1 - g.affine_oracle_d.median() / g.frozen_d.median())),
            retrained_d_median=float(g.qpu_cv_d.median()), hardware_cost_rmse_median=float(g.hardware_effect_cv.median()))
    head = dict(
        runs=int(len(st)),
        frozen_d_median=float(st.frozen_d.median()), retrained_d_median=float(st.qpu_cv_d.median()),
        frozen_d_max=float(st.frozen_d.max()), retrained_d_max=float(st.qpu_cv_d.max()),
        level_share_median=float(np.median(1 - (st.frozen_d_noshift / st.frozen_d) ** 2)),
        qpu_trained_worse_than_ideal_trained=int((st.qpu_cv_rmse > st.ideal_cv_rmse).sum()),
        qpu_trained_better_than_classical=int((st.qpu_cv_rmse < st.classical_cv_rmse).sum()),
        per_candidate=per_cand,
        standardized_frozen_weights={f"C{k}": dict(zip(f["names"], map(float, f["beta"]))) for k, f in frozen.items()})
    with open(os.path.join(args.out, "headline.json"), "w") as fh:
        json.dump(head, fh, indent=1)

    # weights: medians over runs, relative magnitudes, case labels
    g = W.groupby(["label", "observable"], sort=False)
    wsum = g.agg(memory=("memory", "first"), eta_median=("eta", "median"), beta_frozen=("beta_frozen", "first"),
                 beta_ideal2026=("beta_ideal2026", "first"), beta_qpu2026_median=("beta_qpu2026", "median"),
                 importance_ideal2026=("importance_ideal2026", "first"),
                 importance_qpu2026_median=("importance_qpu2026", "median")).reset_index()
    for col in ("beta_frozen", "beta_ideal2026", "beta_qpu2026_median"):
        wsum[col + "_rel"] = wsum[col].abs() / wsum.groupby("label")[col].transform(lambda s: s.abs().max())
    wsum["case_frozen_vs_qpu"] = [case(a, b) for a, b in zip(wsum.beta_frozen_rel, wsum.beta_qpu2026_median_rel)]
    wsum["case_ideal2026_vs_qpu"] = [case(a, b) for a, b in zip(wsum.beta_ideal2026_rel, wsum.beta_qpu2026_median_rel)]
    wsum.to_csv(os.path.join(args.out, "weights_summary.csv"), index=False)

    # report
    ha = summ[summ.policy.isin(["hardware-aware", "session-full", "session-w1"])]
    rep = ["# Frozen versus hardware-trained readouts\n",
           "All RMSE and d values in claims/day on the 2026 test days. Medians over rounds.\n",
           "## Concern 1: does d vanish when the readout is trained on hardware data?\n",
           md(summ[["label", "policy", "runs", "frozen_rmse_qpu", "frozen_d", "affine_probe_d", "affine_oracle_d",
                    "qpu_cv_rmse", "ideal_cv_rmse", "classical_cv_rmse", "qpu_cv_d", "hardware_effect_cv"]]),
           "\n## Concern 2: standardised readout weights (claims/day per SD) and case labels\n",
           md(wsum[["label", "observable", "eta_median", "beta_frozen", "beta_ideal2026", "beta_qpu2026_median",
                    "importance_qpu2026_median", "case_frozen_vs_qpu", "case_ideal2026_vs_qpu"]]),
           "\n## Memory in simulation\n", md(mem)]
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
