"""Linear and nonlinear information-processing capacity of the five frozen candidates.

  python analysis/analyze_nonlinear_capacity.py

Needs results/memory/. Writes results/capacity/.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from compute_final_candidate_memory import ALPHA, N_TRAIN, N_VAL, SEEDS, probe, r2, replay_features  # noqa: E402
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.design import _zridge  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "capacity")
DELAYS = [0, 1, 2, 3, 5, 10, 18, 20, 25]       # 18 and 25 are the K_eff of C4 and C5
CHANNELS = ["C", "D", "P", "H"]
SMOOTH = (0, 1, 2)     # H: binary, linear targets only
NULL_TARGETS, NULL_SEED = 300, 2027


def orthonormal_basis(x, degree=3):
    """Orthonormal polynomials on the empirical distribution of x."""
    powers = np.vstack([x ** d for d in range(degree + 1)]).T
    basis, coefs = [], []
    for d in range(degree + 1):
        v, c = powers[:, d].copy(), np.eye(degree + 1)[d]
        for b, cb in zip(basis, coefs):
            if b is None:
                continue
            proj = np.mean(v * b)
            v, c = v - proj * b, c - proj * cb
        norm = np.sqrt(np.mean(v ** 2))
        if norm < 1e-8 * max(1.0, np.sqrt(np.mean(powers[:, d] ** 2))):
            basis.append(None)
            coefs.append(None)
        else:
            basis.append(v / norm)
            coefs.append(c / norm)
    return coefs


def evaluate(coefs, x):
    return None if coefs is None else np.polyval(coefs[::-1], x)


def target_list(polys):
    """(family, degree, ((channel, delay, order), ...)) for every target."""
    items = [(j, k) for j in range(4) for k in DELAYS]
    smooth = [(j, k) for j, k in items if j in SMOOTH]
    out = [("linear", 1, ((j, k, 1),)) for j, k in items]
    out += [("nonlinear_2", 2, ((j, k, 2),)) for j, k in smooth if polys[j][2] is not None]
    out += [("cross", 2, ((i, k1, 1), (j, k2, 1))) for (i, k1), (j, k2) in itertools.combinations(smooth, 2)]
    out += [("nonlinear_3", 3, ((j, k, 3),)) for j, k in smooth if polys[j][3] is not None]
    return out


def target_values(spec, phi, ends):
    v = np.ones(len(ends))
    for j, k, order in spec:
        v = v * phi[(j, order)][ends - k]
    return v


def capacity(ends, X, y, kmax):
    """Raw squared correlation on 2025 endpoints of a ridge reconstruction trained on 2022-2024."""
    ok = ends >= kmax
    e, Xk, yk = ends[ok], X[ok], y[ok]
    tr, va = e < N_TRAIN, e >= N_TRAIN
    return r2(yk[va], _zridge(Xk[tr], yk[tr], Xk[va], ALPHA))


def ranks(X, ends):
    Z = X[ends < N_TRAIN]
    Z = Z[:, Z.std(0) > 1e-12]
    s = np.linalg.svd((Z - Z.mean(0)) / Z.std(0), compute_uv=False)
    p = s ** 2 / np.sum(s ** 2)
    return int(np.sum(s > 1e-8 * s[0])), float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data().iloc[:N_TRAIN + N_VAL]
    rows, nulls, info = [], [], {}
    for cand, c in CANDIDATES.items():
        theta = encode(df, c["alpha"], **training_scales(df))
        for s in SEEDS:
            u = probe(theta, s)
            polys = [orthonormal_basis(u[:N_TRAIN, j]) for j in range(4)]
            phi = {(j, d): evaluate(polys[j][d], u[:, j]) for j in range(4) for d in (1, 2, 3)}
            ends, X = replay_features(c, u)
            if s == SEEDS[0]:
                info[cand] = dict(zip(("feature_rank", "effective_rank"), ranks(X, ends)))
                specs = target_list(polys)
                rng = np.random.default_rng(NULL_SEED)
                for q in rng.choice(len(specs), NULL_TARGETS, replace=False):      # permutation null
                    spec = specs[q]
                    kmax = max(k for _, k, _ in spec[2])
                    y = target_values(spec[2], phi, ends)
                    nulls.append(dict(candidate=cand, family=spec[0],
                                      raw_capacity=capacity(ends, X, rng.permutation(y), kmax)))
            for fam, deg, spec in target_list(polys):
                kmax = max(k for _, k, _ in spec)
                raw = capacity(ends, X, target_values(spec, phi, ends), kmax)
                a, b = spec[0], spec[-1]
                rows.append(dict(candidate_id=cand, window=c["window"], probe_seed=s, capacity_family=fam,
                                 degree=deg, channel_i=CHANNELS[a[0]], channel_j=CHANNELS[b[0]],
                                 delay_k1=a[1], delay_k2=b[1], raw_capacity=raw))
        print(f"C{cand} done", flush=True)

    tab = pd.DataFrame(rows)
    null = pd.DataFrame(nulls)
    tau = null.groupby("candidate").raw_capacity.quantile(0.99)
    floor = 1 / (N_VAL - 1)
    tab["null_threshold"] = tab.candidate_id.map(tau)
    tab["corrected_capacity"] = np.where(tab.raw_capacity > tab.null_threshold,
                                         np.maximum(tab.raw_capacity - floor, 0), 0.0)
    null.to_csv(os.path.join(OUT, "capacity_null.csv"), index=False)

    tab["family"] = np.where((tab.capacity_family == "linear") & (tab.delay_k1 == 0), "linear_k0", tab.capacity_family)
    past = tab[["delay_k1", "delay_k2"]].max(axis=1) > 0
    linear = tab.capacity_family == "linear"
    tab["part"] = np.select([~past & linear, ~past, linear], ["current_linear", "current_nonlinear", "past_linear"],
                            "past_nonlinear")
    reach = tab[["delay_k1", "delay_k2"]].max(axis=1) <= tab.window - 1
    tab["part_unreachable"] = np.where(reach, 0.0, tab.corrected_capacity)
    per_seed = tab.pivot_table(index=["candidate_id", "probe_seed"], columns="family", values="corrected_capacity",
                               aggfunc="sum", fill_value=0.0)
    per_seed["total"] = per_seed.sum(axis=1)
    per_seed["nonlinear_fraction"] = (per_seed[["nonlinear_2", "cross", "nonlinear_3"]].sum(axis=1) / per_seed.total)
    parts = tab.pivot_table(index=["candidate_id", "probe_seed"], columns="part", values="corrected_capacity",
                            aggfunc="sum", fill_value=0.0)
    parts["unreachable"] = tab.groupby(["candidate_id", "probe_seed"]).part_unreachable.sum()
    per_seed = per_seed.join(parts)
    mean, sd = per_seed.groupby("candidate_id").mean(), per_seed.groupby("candidate_id").std(ddof=1)
    mem = pd.read_csv(os.path.join(HERE, "..", "results", "memory", "memory_vs_forecast.csv")).set_index("candidate")
    summ = pd.DataFrame({
        "candidate_id": mean.index, "window": [CANDIDATES[k]["window"] for k in mean.index],
        "C_linear_k0": mean.linear_k0, "C_linear": mean.linear, "C_nonlinear_degree2": mean.nonlinear_2,
        "C_cross": mean.cross, "C_degree3": mean.nonlinear_3, "C_total": mean.total, "C_total_sd": sd.total,
        "nonlinear_fraction": mean.nonlinear_fraction,
        "C_current_linear": mean.current_linear, "C_current_nonlinear": mean.current_nonlinear,
        "C_past_linear": mean.past_linear, "C_past_nonlinear": mean.past_nonlinear,
        "C_past_nonlinear_sd": sd.past_nonlinear, "C_unreachable_delays": mean.unreachable,
        "feature_rank": [info[k]["feature_rank"] for k in mean.index],
        "effective_rank": [info[k]["effective_rank"] for k in mean.index],
        "null_threshold": tau.reindex(mean.index).values,
        "MC_ch_1_20": mem.mc_ch_1_20.reindex(mean.index).values, "K_eff": mem.k_eff.reindex(mean.index).values,
        "ideal_RMSE_2026": mem.ideal_rmse_2026.reindex(mean.index).values,
        "HA_RMSE_2026": mem.ha_qpu_rmse_mean.reindex(mean.index).values,
        "HA_distortion": mem.ha_d_median.reindex(mean.index).values})
    cols = ["candidate_id", "window", "probe_seed", "capacity_family", "degree", "channel_i", "channel_j", "delay_k1",
            "delay_k2", "raw_capacity", "null_threshold", "corrected_capacity", "part"]
    tab[cols].to_csv(os.path.join(OUT, "capacity_by_target.csv"), index=False)
    tab[cols].to_json(os.path.join(OUT, "capacity_by_target.json"), orient="records")
    summ.to_csv(os.path.join(OUT, "capacity_summary.csv"), index=False)
    summ.to_json(os.path.join(OUT, "capacity_summary.json"), orient="records", indent=1)
    meta = dict(probe_seeds=SEEDS, delays=DELAYS, channels=CHANNELS, readout="ridge on standardized 18 one-body "
                "Pauli expectations, alpha=1e-6, intercept", train="endpoints in 2022-2024 (index < 1095)",
                evaluation="endpoints in 2025 (365)", features="reset-window replay with each frozen W",
                basis="per channel, polynomials orthonormal on the empirical distribution of the probe training "
                      "rows (Gram-Schmidt on 1, x, x^2, x^3); degrees with zero residual are skipped; the binary holiday channel H enters only linear targets",
                families={"linear": "phi1(u_j(t-k))", "nonlinear_2": "phi2(u_j(t-k))",
                          "cross": "phi1(u_i(t-k1)) phi1(u_j(t-k2)), all pairs of (channel, delay)",
                          "nonlinear_3": "phi3(u_j(t-k))"},
                correction=f"corrected = max(raw - 1/(365-1), 0) if raw > 99th percentile of a permutation null "
                           f"({NULL_TARGETS} random targets, seed {NULL_SEED}, per candidate) else 0",
                n_targets=int(tab.groupby(["candidate_id", "probe_seed"]).size().iloc[0]),
                frozen_candidates={k: {x: CANDIDATES[k][x] for x in ("graph", "window", "r", "alpha", "hx", "hy")}
                                   for k in CANDIDATES})
    json.dump(meta, open(os.path.join(OUT, "capacity_metadata.json"), "w"), indent=1, default=float)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
