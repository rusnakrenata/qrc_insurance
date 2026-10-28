"""Contraction of the observables against coherence and gate error (Sec. VI-C, VI-D).

For the 63 stored runs and the C4 memory-ablation session, T1 and T2 of the layout qubits and
the CZ errors of the layout couplers come from the calibration snapshot of the round, and eta_j
is refitted from the 2026 features (z_qpu = eta_j * z_ideal + zeta_j). Spearman correlations of
  1  eta(Y4X5) of C1 and C4 with T2 and with circuit duration / memory-qubit T2
  2  eta of each X_j, Z_j with T1, T2 and CZ error of its own qubit (within candidate and observable)
  3  the mean eta of a run with the median T2 and CZ errors of its layout (within candidate)

Usage:  python analysis/coherence_contraction.py
Writes results/coherence/.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG = ROOT
RUNS = os.path.join(PKG, "data", "ibm_kingston_runs")
OUT = os.path.join(ROOT, "results", "coherence")
EDGES = {"H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
         "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)], "H3": [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)],
         "H4": [(1, 2), (0, 1), (0, 4), (3, 4), (4, 5)]}
TOPO = {1: "H3", 2: "H3", 3: "H2", 4: "H0", 5: "H2"}
POLICY_MARK = {"hardware-aware": "o", "random": "^", "qiskit": "s", "session-full": "*", "session-w1": "P"}
CAND_COLOR = {1: "#2a6fdb", 2: "#7a5195", 3: "#ef5675", 4: "#e08b00", 5: "#2f9e75"}


def calibration(props_path):
    p = json.load(open(props_path))
    q = {i: {e["name"]: e["value"] for e in entries} for i, entries in enumerate(p["qubits"])}
    cz = {}
    for g in p["gates"]:
        if g["gate"] == "cz":
            e = [x["value"] for x in g["parameters"] if x["name"] == "gate_error"]
            if e:
                cz[tuple(sorted(g["qubits"]))] = e[0]
    return q, cz


def contraction(zi, zq):
    if np.std(zi) < 1e-12:
        return np.nan, np.nan
    eta, _ = np.polyfit(zi, zq, 1)
    return float(eta), float(np.corrcoef(zi, zq)[0, 1])


def runs_table():
    runs = pd.read_csv(os.path.join(PKG, "results", "runs.csv"))
    rows = []
    for _, r in runs.iterrows():
        rows.append(dict(candidate=int(r.candidate), round=int(r["round"]), policy=r.approach,
                         layout=json.loads(r.layout), duration_us=float(r.max_duration_us),
                         proxy=float(r.error_proxy_nll), transfer_d=float(r.transfer_d),
                         features=os.path.join(RUNS, r.source, "2026_features_predictions.csv"),
                         props=os.path.join(RUNS, os.path.dirname(r.source), "backend_properties.json")))
    sessions = sorted(glob.glob(os.path.join(ROOT, "data", "ibm_kingston_ablation", "candidate4", "*")))
    sessions = [s for s in sessions if all(os.path.exists(os.path.join(s, a, "2026_metrics.json")) for a in ("full", "w1"))]
    if sessions:
        s = sessions[-1]
        for arm in ("full", "w1"):
            m = json.load(open(os.path.join(s, arm, "2026_metrics.json")))
            rows.append(dict(candidate=4, round=0, policy=f"session-{arm}", layout=m["layout"],
                             duration_us=float(m["resources"]["max_duration_us"]), proxy=np.nan,
                             transfer_d=float(m["transfer_d"]),
                             features=os.path.join(s, arm, "2026_features_predictions.csv"),
                             props=os.path.join(s, "backend_properties.json")))
    return rows


def build():
    run_rows, obs_rows = [], []
    for r in runs_table():
        q, cz = calibration(r["props"])
        L = r["layout"]
        t1 = np.array([q[p].get("T1", np.nan) for p in L])
        t2 = np.array([q[p].get("T2", np.nan) for p in L])
        edges = EDGES[TOPO[r["candidate"]]]
        cze = np.array([cz.get(tuple(sorted((L[a], L[b]))), np.nan) for a, b in edges])
        mem_cz = np.array([cz.get(tuple(sorted((L[a], L[b]))), np.nan) for a, b in edges if a >= 4 or b >= 4])
        d = pd.read_csv(r["features"])
        names = [c[:-6] for c in d.columns if c.endswith("_ideal") and c != "pred_ideal"]
        etas = {}
        for n in names:
            eta, corr = contraction(d[n + "_ideal"].values, d[n + "_qpu"].values)
            etas[n] = eta
            qubits = [4, 5] if n == "YX45" else [int(n[1])]
            obs_rows.append(dict(candidate=r["candidate"], round=r["round"], policy=r["policy"], observable=n,
                                 memory=n == "YX45", eta=eta, corr=corr,
                                 qubit_T1=float(np.nanmedian(t1[qubits])), qubit_T2=float(np.nanmedian(t2[qubits])),
                                 qubit_cz_mean=float(np.nanmean([cz.get(tuple(sorted((L[a], L[b]))), np.nan)
                                                                 for a, b in edges if a in qubits or b in qubits])),
                                 duration_us=r["duration_us"]))
        run_rows.append(dict(candidate=r["candidate"], round=r["round"], policy=r["policy"], layout=json.dumps(L),
                             duration_us=r["duration_us"], T1_median=float(np.nanmedian(t1)),
                             T2_median=float(np.nanmedian(t2)), T2_min=float(np.nanmin(t2)),
                             T1_mem_median=float(np.nanmedian(t1[4:])), T2_mem_median=float(np.nanmedian(t2[4:])),
                             T2_mem_min=float(np.nanmin(t2[4:])), cz_mean=float(np.nanmean(cze)),
                             cz_max=float(np.nanmax(cze)), cz_mem_mean=float(np.nanmean(mem_cz)), proxy=r["proxy"],
                             transfer_d=r["transfer_d"], eta_YX45=etas.get("YX45", np.nan),
                             eta_mean=float(np.nanmean([v for k, v in etas.items() if k != "YX45"]))))
    R = pd.DataFrame(run_rows)
    R["time_over_T2mem"] = R.duration_us / R.T2_mem_median
    return R, pd.DataFrame(obs_rows)


def corr_row(test, x, y, label, df):
    m = df[[x, y]].dropna()
    if len(m) < 4:
        return None
    rs, ps = spearmanr(m[x], m[y])
    rp, pp = pearsonr(m[x], m[y])
    return dict(test=test, subset=label, x=x, y=y, n=len(m), spearman=rs, spearman_p=ps, pearson=rp, pearson_p=pp)


def within_group_spearman(df, x, y, keys):
    """Spearman of x and y after ranking within each group: correlation pooled over groups."""
    rows = []
    for _, g in df.groupby(keys):
        g = g[[x, y]].dropna()
        if len(g) >= 3 and g[x].nunique() > 1 and g[y].nunique() > 1:
            rows.append(pd.DataFrame(dict(rx=g[x].rank() - g[x].rank().mean(), ry=g[y].rank() - g[y].rank().mean())))
    if not rows:
        return np.nan, 0, np.nan
    a = pd.concat(rows)
    rho = float((a.rx * a.ry).sum() / np.sqrt((a.rx ** 2).sum() * (a.ry ** 2).sum()))
    n = len(a)
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    from scipy.stats import t as tdist
    return rho, n, float(2 * tdist.sf(abs(t), n - 2))


def plots(R, O, out):
    stored = R[~R.policy.str.startswith("session")]
    mem = R.dropna(subset=["eta_YX45"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for (c, p), g in mem.groupby(["candidate", "policy"]):
        for a, x in zip(ax[:2], ("T2_median", "T2_mem_median")):
            a.scatter(g[x], g.eta_YX45, color=CAND_COLOR[c], marker=POLICY_MARK[p], s=70 if "session" in p else 45,
                      edgecolor="k", linewidth=0.4, label=f"C{c} {p}")
        ax[2].scatter(g.time_over_T2mem, g.eta_YX45, color=CAND_COLOR[c], marker=POLICY_MARK[p],
                      s=70 if "session" in p else 45, edgecolor="k", linewidth=0.4)
    ax[0].set_xlabel("median T2 of the six layout qubits (us)")
    ax[1].set_xlabel("median T2 of the two memory qubits (us)")
    ax[2].set_xlabel("circuit duration / memory-qubit T2")
    ax[2].set_xscale("log")
    for a in ax:
        a.set_ylabel(r"contraction $\eta$ of $Y_4X_5$")
        a.axhline(0, color="0.6", lw=0.8)
        a.grid(alpha=0.25)
    h, l = ax[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Memory correlator Y4X5: contraction against coherence (C1: W=4, ~16 us; C4: W=117, ~499 us)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "fig_memory_correlator_eta_vs_T2.png"), dpi=150)
    plt.close(fig)

    single = O[(~O.memory) & (~O.policy.str.startswith("session"))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for c, g in single.groupby("candidate"):
        ax[0].scatter(g.qubit_T2, g.eta, s=12, alpha=0.6, color=CAND_COLOR[c], label=f"C{c}")
        ax[1].scatter(g.qubit_cz_mean * 100, g.eta, s=12, alpha=0.6, color=CAND_COLOR[c])
    ax[0].set_xlabel("T2 of the qubit carrying the observable (us)")
    ax[1].set_xlabel("mean CZ error of the couplers of that qubit (%)")
    for a in ax:
        a.set_ylabel(r"contraction $\eta$ of $X_j$ / $Z_j$")
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.suptitle("Single-qubit observables, 63 runs: T2 of their qubit vs CZ error of their couplers")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_single_qubit_eta_vs_T2_and_CZ.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.4))
    for (c, p), g in stored.groupby(["candidate", "policy"]):
        ax.scatter(g.T2_median, g.eta_mean, color=CAND_COLOR[c], marker=POLICY_MARK[p], s=40, edgecolor="k",
                   linewidth=0.4)
    ax.set_xlabel("median T2 of the layout (us)")
    ax.set_ylabel(r"mean $\eta$ over the injection observables")
    ax.grid(alpha=0.25)
    ax.set_title("Run level, 63 runs (colour: candidate, marker: policy)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_run_mean_eta_vs_T2.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    R, O = build()
    R.to_csv(os.path.join(OUT, "per_run.csv"), index=False)
    O.to_csv(os.path.join(OUT, "per_observable.csv"), index=False)
    stored = R[~R.policy.str.startswith("session")]
    tests = []

    # 1 memory correlator
    for label, sub in (("C1 (W=4)", stored[stored.candidate == 1]), ("C4 (W=117)", stored[stored.candidate == 4]),
                       ("C1+C4 stored", stored[stored.candidate.isin([1, 4])]),
                       ("C1+C4 stored + session", R[R.candidate.isin([1, 4])])):
        for x in ("T2_median", "T2_mem_median", "T2_mem_min", "T1_mem_median", "cz_mem_mean", "proxy", "time_over_T2mem"):
            row = corr_row("1 memory correlator", x, "eta_YX45", label, sub)
            if row:
                tests.append(row)

    # 2 single-qubit observables against their own qubit
    single = O[(~O.memory) & (~O.policy.str.startswith("session"))]
    for x in ("qubit_T2", "qubit_T1", "qubit_cz_mean"):
        rho, n, p = within_group_spearman(single, x, "eta", ["candidate", "observable"])
        tests.append(dict(test="2 single-qubit observables", subset="within candidate x observable", x=x, y="eta",
                          n=n, spearman=rho, spearman_p=p, pearson=np.nan, pearson_p=np.nan))

    # 3 run level, within candidate
    for x in ("T2_median", "T2_min", "T1_median", "cz_mean", "cz_max", "proxy"):
        rho, n, p = within_group_spearman(stored, x, "eta_mean", ["candidate"])
        tests.append(dict(test="3 run mean eta", subset="within candidate", x=x, y="eta_mean", n=n, spearman=rho,
                          spearman_p=p, pearson=np.nan, pearson_p=np.nan))
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(OUT, "correlations.csv"), index=False)

    pol = stored.groupby("policy")[["T2_median", "T2_min", "T2_mem_median", "T1_median", "cz_mean", "cz_max"]].median()
    pol.to_csv(os.path.join(OUT, "hardware_by_policy.csv"))

    plots(R, O, OUT)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    print(pol.round(4).to_string())
    print(R[R.candidate.isin([1, 4])][["candidate", "round", "policy", "duration_us", "T2_median", "T2_mem_median",
                                      "time_over_T2mem", "eta_YX45"]].round(3).to_string(index=False))
    print(f"results: {OUT}")


if __name__ == "__main__":
    main()
