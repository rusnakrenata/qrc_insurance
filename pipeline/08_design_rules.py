"""Design stage 4: replay search and selection rules R1-R4.

  python pipeline/08_design_rules.py [--search H3]

Writes results/design/rules_*.csv.
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
from qrc.candidates import GRAPHS, load_data, training_scales  # noqa: E402
from qrc.design import ONE_BODY, Reservoir, _zridge, cv_rmse, select_lambda  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "design")
OUT = os.path.join(HERE, "..", "results", "design")
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0]
INJ = ["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3"]
READOUTS = {"XZ_injection": INJ,
            "XZinj_dropX3": [f for f in INJ if f != "X3"],
            "XZinj_dropX3_plus_YX45": [f for f in INJ if f != "X3"] + ["YX45"],
            "XZinj_plus_YX45": INJ + ["YX45"],
            "XYZ_all": ONE_BODY}
SETTINGS = {k: (3 if k == "XYZ_all" else 2) for k in READOUTS}
SHOTS = 1024


def features(c, theta, names):
    """Feature matrix and the target-row index for CONT or RWP operation."""
    res = Reservoir(2, GRAPHS[c["graph"]], c["J"], c["hx"], c["hy"], c["dt"], c["r"])
    S, F = res.maps(theta, names)
    if c["protocol"] == "CONT":
        return res.cont(S, F), np.arange(len(theta))
    ends = np.arange(c["window"] - 1, len(theta))
    return res.rwp(S, F, ends, c["window"]), ends


def shot_metrics(X, y, lam):
    """Forecast shot-noise proxy and resolvability of every feature at 1024 shots."""
    m, s = X.mean(0), X.std(0)
    keep = s > 1e-12
    from sklearn.linear_model import Ridge
    w = Ridge(alpha=lam).fit((X[:, keep] - m[keep]) / s[keep], y).coef_
    shot_sd = np.sqrt(np.clip(1 - np.clip(m[keep], -1, 1) ** 2, 0, None) / SHOTS)
    return dict(forecast_shot_sd_1024=float(np.sqrt(np.sum((w / s[keep] * shot_sd) ** 2))),
                max_shot_to_train_sd=float(np.max(shot_sd / s[keep])),
                all_resolvable=bool(np.all(shot_sd <= s[keep])))


def evaluate(c, theta, y, lam=None):
    cols = READOUTS[c["readout"]]
    X, ends = features(c, theta, cols)
    tr, va = ends < 1095, (ends >= 1095) & (ends < 1460)
    if lam is None or np.isnan(lam):
        lam = select_lambda(X[tr], y[ends[tr]], LAMBDAS)[0]
    cv, sd = cv_rmse(X[tr], y[ends[tr]], lam)
    pred = _zridge(X[tr], y[ends[tr]], X[va], lam)
    w = 1 if c["protocol"] == "CONT" else c["window"]
    return dict(lam=lam, cv_rmse=cv, cv_sd=sd, rmse_2025=float(np.sqrt(np.mean((pred - y[ends[va]]) ** 2))),
                cz_per_feature_vector=2 * 5 * c["r"] * w * SETTINGS[c["readout"]], settings=SETTINGS[c["readout"]],
                **shot_metrics(X[tr], y[ends[tr]], lam))


def rule_r1(t):
    return t.sort_values(["cv_rmse", "settings", "cz_per_feature_vector"]).iloc[0]


def rule_r2(t):
    v = t[t.washout.notna()]
    return v.sort_values("mc_per_channel", ascending=False).iloc[0]


def rule_r3(t):
    """Lowest CV among shot-resolvable reservoirs; the shot-noise proxy breaks ties."""
    ok = t[t.all_resolvable]
    if len(ok):
        return ok.sort_values(["cv_rmse", "forecast_shot_sd_1024", "settings", "cz_per_feature_vector"]).iloc[0]
    return t.sort_values(["max_shot_to_train_sd", "forecast_shot_sd_1024", "cv_rmse"]).iloc[0]


def rule_r4(t):
    best = rule_r1(t)
    near = t[t.cv_rmse <= best.cv_rmse + best.cv_sd / np.sqrt(5)]
    return near.sort_values(["swaps", "cz_per_feature_vector", "settings", "depth", "duration_us"]).iloc[0]


def neighbourhood(c):
    """Fixed stage-C moves around a parent reservoir."""
    out = [dict(c)]
    out += [dict(c, alpha=a) for a in (.25, .5, .75, 1.0) if a != c["alpha"]]
    out += [dict(c, dt=d) for d in (.8, 1.2, 1.6, 2.0) if d != c["dt"]]
    out += [dict(c, hx=c["hx"] * s) for s in (.75, 1.25)]
    out += ([dict(c, hy=h) for h in (-.3, .3)] if c["hy"] == 0 else
            [dict(c, hy=c["hy"] * s) for s in (.5, 1.5, -1.0)] + [dict(c, hy=0.0)])
    out += [dict(c, J=[j * s for j in c["J"]]) for s in (.75, 1.25)]
    out += [dict(c, J=[j * (s if k == e else 1) for k, j in enumerate(c["J"])]) for e in range(5) for s in (.85, 1.15)]
    for a, hs, js, ds in ((.75, .9, .9, .9), (1., 1.1, 1.1, 1.1), (1., .9, .9, 1.), (.5, 1.1, 1.1, 1.)):
        out.append(dict(c, alpha=a, hx=c["hx"] * hs, hy=c["hy"] * hs, dt=c["dt"] * ds, J=[j * js for j in c["J"]]))
    return out


def trotter_moves(c, r, seed=42, n=8):
    rng = np.random.default_rng(seed + 1009 * r + 37 * c["window"])
    out = [dict(c, r=r)]
    out += [dict(c, r=r, hx=c["hx"] * s, hy=c["hy"] * s, J=[j * s for j in c["J"]]) for s in (.9, 1.1)]
    out += [dict(c, r=r, dt=c["dt"] * s) for s in (.9, 1.1)]
    for _ in range(n):
        d = dict(c, r=r, hx=c["hx"] * float(np.clip(rng.normal(1, .08), .8, 1.2)))
        if d["hy"] != 0:
            d["hy"] = d["hy"] * float(np.clip(rng.normal(1, .1), .75, 1.25))
        d["dt"] = d["dt"] * float(np.clip(rng.normal(1, .06), .85, 1.15))
        d["J"] = [j * float(np.clip(rng.normal(1, .1), .75, 1.25)) for j in d["J"]]
        out.append(d)
    return out


def search(graph, df, y):
    """Full RWP search for one graph, starting from its best stage-3 reservoir."""
    rec = pd.read_csv(os.path.join(DATA, "trotter_search_best.csv"))
    b = rec[rec.topology == graph].sort_values(["cv_rmse", "validation_rmse"]).iloc[0]
    base = dict(graph=graph, protocol="RWP", alpha=.75, dt=1.6, r=int(b.trotter_r), hx=b.hx, hy=b.hy,
                J=[b[f"J{min(e)}{max(e)}"] for e in GRAPHS[graph]])
    rows = []

    def run(c):
        theta = encode(df, c["alpha"], **training_scales(df))
        for ro in READOUTS:
            rows.append(dict(c, readout=ro, J=json.dumps(c["J"]), **evaluate(dict(c, readout=ro), theta, y)))
        return rows[-len(READOUTS):]

    for w in range(1, 29):
        best = min(run(dict(base, window=w)), key=lambda r: (not r["all_resolvable"], r["cv_rmse"]))
        parent = dict(base, window=w, alpha=best["alpha"])
        for c in neighbourhood(parent)[1:]:
            run(c)
        for r in (1, 2, 3, 4):
            for c in trotter_moves(parent, r):
                run(c)
        print(graph, "W =", w, "done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", choices=list(GRAPHS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = load_data().iloc[:1460].reset_index(drop=True)
    y = df.target_property_damage_claim_count.values.astype(float)
    if args.search:
        search(args.search, df, y).to_csv(os.path.join(OUT, f"rules_search_{args.search}.csv"), index=False)
        return

    reps = pd.read_csv(os.path.join(DATA, "rule_representatives.csv"))
    rows = []
    for _, r in reps.iterrows():
        J = json.loads(r.J_json)
        edges = GRAPHS[r.topology]
        c = dict(graph=r.topology, protocol=r.protocol, window=int(r.window) if r.window == r.window else 1,
                 alpha=r["alpha"], dt=r["dt"], hx=r["hx"], hy=r["hy"], r=int(r["original_r"]), readout=r["test_readout"],
                 J=[J[f"J{min(e)}{max(e)}"] if f"J{min(e)}{max(e)}" in J else J[f"J{e[0]}{e[1]}"] for e in edges])
        theta = encode(df, c["alpha"], **training_scales(df))
        res = evaluate(c, theta, y, r.source_selected_lambda)
        rows.append(dict(key=r.candidate_key, rules=r.selected_by_rules, protocol=r.protocol, graph=r.topology,
                         window=c["window"], r=c["r"], readout=c["readout"], cv_recorded=r.source_readout_cv_rmse,
                         rmse_2025_recorded=r.source_validation_rmse, **res))
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "rules_representatives.csv"), index=False)
    print("max |CV - recorded| =", (out.cv_rmse - out.cv_recorded).abs().max())

    # rule checks
    checks = []
    for (proto, g), t in out.groupby(["protocol", "graph"]):
        r1 = t[t.rules.str.contains("R1")].iloc[0]
        for _, row in t.iterrows():
            checks.append(dict(protocol=proto, graph=g, key=row.key, rules=row.rules,
                               cv_not_below_R1=row.cv_rmse >= r1.cv_rmse - 1e-9,
                               R3_resolvable=(not "R3" in row.rules) or row.all_resolvable,
                               R4_within_1SE=(not "R4" in row.rules) or row.cv_rmse <= r1.cv_rmse + r1.cv_sd / np.sqrt(5) + 1e-9,
                               R4_not_costlier=(not "R4" in row.rules) or row.cz_per_feature_vector <= r1.cz_per_feature_vector))
    ck = pd.DataFrame(checks)
    ck.to_csv(os.path.join(OUT, "rules_checks.csv"), index=False)
    print(ck[~ck[["cv_not_below_R1", "R3_resolvable", "R4_within_1SE", "R4_not_costlier"]].all(axis=1)].to_string())
    rmap = out.assign(rule=out.rules.str.split(";")).explode("rule").pivot(index=["protocol", "graph"], columns="rule",
                                                                           values="key")
    rmap.to_csv(os.path.join(OUT, "rules_map.csv"))
    print(rmap.to_string(), "\nunique reservoirs:", out.key.nunique(),
          "(CONT", (out.protocol == "CONT").sum(), ", RWP", (out.protocol == "RWP").sum(), ")")


if __name__ == "__main__":
    main()
