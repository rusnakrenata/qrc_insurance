"""Publication figures (vector PDF) for the article. Run after the other analysis scripts."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import load_data, training_scales  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
ONE, TWO = 3.5, 7.16      # IEEE column widths (inches)
COL = {"hardware-aware": "#00ABE5", "random": "#2B3335", "qiskit": "#E21A22"}
MARK = {"hardware-aware": "o", "random": "^", "qiskit": "s"}
LABEL = {"hardware-aware": "HA", "random": "Random", "qiskit": "Qiskit L1"}
GRAY, INK = "#8a8986", "#2b2b2a"
APPROACHES = ["hardware-aware", "random", "qiskit"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def fig_data():
    """Daily claims, exposure and the encoding's clipping threshold over the three periods."""
    df = load_data()
    sc = training_scales(df)
    clip = sc["mean_c"] + 3 * sc["std_c"]
    fig, ax = plt.subplots(2, 1, figsize=(ONE, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    t = df.input_date
    ax[0].plot(t, df.property_damage_claim_count_t, color=GRAY, lw=0.4)
    roll = df.property_damage_claim_count_t.rolling(28, center=True).mean()
    ax[0].plot(t, roll, color=INK, lw=1.0, label="28-day mean")
    ax[0].axhline(clip, color="#F7941D", lw=0.9, ls="--", label=r"clip level $\bar C+3\sigma_C$")
    ax[0].set_ylabel("Claims per day")
    ax[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax[1].plot(t, df.active_policy_count_t / 1000, color=INK, lw=1.0)
    ax[1].set_ylabel(r"Policies ($10^3$)")
    for a in ax:
        for x0, lab in [(pd.Timestamp("2025-01-01"), "validation"), (pd.Timestamp("2026-01-01"), "test")]:
            a.axvline(x0, color=GRAY, lw=0.6, ls=":")
    ax[0].text(pd.Timestamp("2023-07-01"), 31, "train 2022–24", ha="center", fontsize=7)
    ax[0].text(pd.Timestamp("2025-07-01"), 31, "2025", ha="center", fontsize=7)
    ax[0].text(pd.Timestamp("2026-04-25"), 31, "2026", ha="center", fontsize=7)
    ax[0].set_ylim(0, 34)
    fig.align_ylabels()
    save(fig, "fig_data")


def fig_forecasting():
    """Test RMSE of the classical models and the ideal reservoirs, on all test days and on each half."""
    t = pd.read_csv(os.path.join(RES, "tables", "forecasting_comparison.csv"))
    names = {"ridge": "Ridge", "nb": "NB2 GLM", "esn": "ESN", "gru": "GRU", "rnn": "Vanilla RNN",
             "xgboost_squared": "XGBoost (sq.)", "poisson": "Poisson GLM", "xgboost_poisson": "XGBoost (Poi.)"}
    rows = []
    for _, r in t.iterrows():
        if r.model in names:
            rows.append((names[r.model], r.rmse, "classical", r.rmse_first_half, r.rmse_second_half))
        elif "ideal" in r.model:
            rows.append((r.model.replace("QRC-", "").replace(" ideal", " ideal"), r.rmse, "ideal",
                         r.rmse_first_half, r.rmse_second_half))
    d = pd.DataFrame(rows, columns=["name", "rmse", "kind", "h1", "h2"]).sort_values("rmse")
    fig, ax = plt.subplots(figsize=(ONE, 2.5))
    y = np.arange(len(d))[::-1]
    sty = {"classical": ("#238037", "D", "Classical"), "ideal": ("#AA54A0", "o", "QRC, ideal")}
    for k, (c, m, lab) in sty.items():
        s = d.kind.values == k
        ax.scatter(d.h1.values[s], y[s], marker=m, facecolor="white", edgecolor=c, s=14, lw=0.8, zorder=3)
        ax.scatter(d.rmse.values[s], y[s], marker=m, color=c, s=18, zorder=4, label=lab)
        for yy, a, b in zip(y[s], d.h1.values[s], d.h2.values[s]):
            ax.plot([a, b], [yy, yy], color=c, lw=0.6, zorder=2)
        ax.scatter(d.h2.values[s], y[s], marker="|", color=c, s=30, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.name)
    ax.axvline(d[d.name == "Ridge"].rmse.iloc[0], color=GRAY, lw=0.6, ls="--")
    ax.set_xlabel("RMSE on 2026 test dates (claims/day)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=2, fontsize=6.5,
              handletextpad=0.2, columnspacing=0.8)
    ax.set_xlabel("RMSE on 2026 test dates (claims/day)\nfilled: all 232 days; open: first 116 days; bar: last 116 days")
    save(fig, "fig_forecasting")


def fig_hardware():
    """Transfer distortion and forecast RMSE of every QPU run, by candidate and layout policy."""
    r = pd.read_csv(os.path.join(RES, "runs.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(TWO, 2.3))
    for j, (col, lab) in enumerate([("transfer_d", r"Transfer distortion $d$ (claims/day)"),
                                    ("qpu_rmse", "Forecast RMSE on QPU (claims/day)")]):
        a = ax[j]
        for i, ap in enumerate(APPROACHES):
            g = r[r.approach == ap]
            x = g.candidate + (i - 1) * 0.22 + np.random.default_rng(i).uniform(-0.04, 0.04, len(g))
            a.scatter(x, g[col], marker=MARK[ap], facecolor="white" if ap != "hardware-aware" else COL[ap],
                      edgecolor=COL[ap], s=16, lw=0.8, label=LABEL[ap], zorder=3)
        if col == "qpu_rmse":
            ideal = r.groupby("candidate").ideal_rmse.first()
            for c, v in ideal.items():
                a.plot([c - 0.4, c + 0.4], [v, v], color=INK, lw=0.8, zorder=2)
            a.plot([], [], color=INK, lw=0.8, label="ideal reservoir")
        a.set_yscale("log")
        a.set_yticks([1, 2, 5, 10, 20, 35] if col == "transfer_d" else [5, 6, 8, 10, 15, 20, 30])
        a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xticks(range(1, 6))
        a.set_xticklabels([f"C{c}" for c in range(1, 6)])
        a.set_ylabel(lab)
        a.set_title("(a)" if j == 0 else "(b)", loc="left")
    # gap between panels
    fig.tight_layout(w_pad=4.4)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    gap = (ax[1].get_tightbbox(rend).x0 - ax[0].get_tightbbox(rend).x1) / fig.dpi
    shift = (1.0 / 2.54 / 0.8 - gap) / TWO
    p0, p1 = ax[0].get_position(), ax[1].get_position()
    ax[0].set_position([p0.x0, p0.y0, p0.width - shift / 2, p0.height])
    ax[1].set_position([p1.x0 + shift / 2, p1.y0, p1.width - shift / 2, p1.height])
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.canvas.draw()
    gap = (ax[1].get_tightbbox(rend).x0 - ax[0].get_tightbbox(rend).x1) / fig.dpi
    print(f"fig_hardware panel gap {gap * 2.54 * 0.8:.2f} cm at 0.8 textwidth")
    save(fig, "fig_hardware")


def fig_reproducibility():
    """Median daily gap |hardware - ideal| of each round: spread across repeated rounds."""
    r = pd.read_csv(os.path.join(RES, "runs.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.0), sharey=True)
    for j, ap in enumerate(APPROACHES):
        a = ax[j]
        for c in range(1, 6):
            g = r[(r.approach == ap) & (r.candidate == c)].sort_values("round")
            a.plot(g["round"], g.gap_median, marker=MARK[ap], ms=3, lw=0.8, color=COL[ap],
                   alpha=0.35 + 0.13 * c)
            a.text(g["round"].iloc[-1] + 0.12, g.gap_median.iloc[-1] * (1.12 if c == 1 else 1.0), f"C{c}",
                   fontsize=5.5, va="center", color=INK)
        a.set_yscale("log")
        a.set_yticks([0.5, 1, 2, 5, 10, 20, 35])
        a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xticks(range(1, 6))
        a.set_xlim(0.7, 5.6)
        a.set_xlabel("Round")
        a.set_title(LABEL[ap])
    ax[0].set_ylabel("Median daily |QPU − ideal| (claims)")
    save(fig, "fig_reproducibility")


def fig_mechanism():
    """(a) level shift vs distortion; (b) contraction of each observable class; (c) shift attribution."""
    r = pd.read_csv(os.path.join(RES, "runs.csv"))
    o = pd.read_csv(os.path.join(RES, "observables.csv"))
    o["kind"] = np.where(o.observable == "YX45", r"$Y_4X_5$", np.where(o.observable.str[0] == "X", r"$X_j$", r"$Z_j$"))
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.1))
    a = ax[0]
    for ap in APPROACHES:
        g = r[r.approach == ap]
        a.scatter(g.transfer_d ** 2, g.level_shift ** 2, marker=MARK[ap], s=12, lw=0.7, edgecolor=COL[ap],
                  facecolor=COL[ap] if ap == "hardware-aware" else "white", label=LABEL[ap])
    lim = [0.3, 2000]
    a.plot(lim, lim, color=GRAY, lw=0.6, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lim)
    a.set_ylim(1e-3, 2000)
    a.set_xlabel(r"$d^2$")
    a.set_ylabel(r"$\bar\delta^{\,2}$ (squared level shift)")
    a.legend(frameon=False, fontsize=6)
    a = ax[1]
    kinds = [r"$X_j$", r"$Z_j$", r"$Y_4X_5$"]
    data = [o[o.kind == k].eta.values for k in kinds]
    bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for p in bp["boxes"]:
        p.set(facecolor="white", edgecolor=INK, lw=0.7)
    for k in ("whiskers", "caps", "medians"):
        for p in bp[k]:
            p.set(color=INK, lw=0.7)
    a.axhline(1, color=GRAY, lw=0.6, ls="--")
    a.set_xticks([1, 2, 3])
    a.set_xticklabels(kinds)
    a.set_ylabel(r"Contraction $\eta_j$")
    a = ax[2]
    c1 = o[(o.candidate == 1)].groupby(["approach", "observable"]).shift_contribution.mean().unstack()
    obs = ["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3", "YX45"]
    lab = [r"$X_0$", r"$X_1$", r"$X_2$", r"$Z_0$", r"$Z_1$", r"$Z_2$", r"$Z_3$", r"$Y_4X_5$"]
    x = np.arange(len(obs))
    for i, ap in enumerate(APPROACHES):
        a.bar(x + (i - 1) * 0.27, c1.loc[ap, obs], width=0.25, color=COL[ap], label=LABEL[ap])
    a.axhline(0, color=INK, lw=0.5)
    a.set_xticks(x)
    a.set_xticklabels(lab, fontsize=6)
    a.set_ylabel(r"$w_j\,\overline{\Delta z_j}$ for C1 (claims/day)")
    a.legend(frameon=False, fontsize=6, loc="upper left")
    for i, a in enumerate(ax):
        a.set_title(f"({'abc'[i]})", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_mechanism")


def fig_predictors():
    """Pre-deployment indicators of the realised distortion."""
    r = pd.read_csv(os.path.join(RES, "runs.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.1))
    ha = r[r.approach == "hardware-aware"]
    a = ax[0]
    a.scatter(ha.probe_d, ha.transfer_d, marker="o", s=14, color=COL["hardware-aware"])
    a.plot([0, 9], [0, 9], color=GRAY, lw=0.6, ls="--")
    a.set_xlabel(r"Probe $d$, 100 training dates")
    a.set_ylabel(r"Realised $d$ on 2026 (claims/day)")
    a.set_xlim(0, 9.5)
    a.set_ylim(0, 9.5)
    a = ax[1]
    for ap in APPROACHES:
        g = r[r.approach == ap]
        a.scatter(g.noisy_direct_d, g.transfer_d, marker=MARK[ap], s=12, lw=0.7, edgecolor=COL[ap],
                  facecolor=COL[ap] if ap == "hardware-aware" else "white", label=LABEL[ap])
    a.plot([0.1, 40], [0.1, 40], color=GRAY, lw=0.6, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"Aer $d$ (calibrated noise model)")
    a.set_ylabel(r"QPU $d$")
    a.legend(frameon=False, fontsize=6)
    a = ax[2]
    for ap in APPROACHES:
        g = r[r.approach == ap]
        a.scatter(100 * g.cal_max_cz, g.transfer_d, marker=MARK[ap], s=12, lw=0.7, edgecolor=COL[ap],
                  facecolor=COL[ap] if ap == "hardware-aware" else "white")
    a.axvline(4, color="#F7941D", lw=0.6, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Worst CZ error on layout (%)")
    a.set_ylabel(r"QPU $d$")
    for i, a in enumerate(ax):
        a.set_title(f"({'abc'[i]})", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_predictors")


if __name__ == "__main__":
    fig_data()
    fig_forecasting()
    fig_hardware()
    fig_reproducibility()
    fig_mechanism()
    fig_predictors()
    print("figures written to", os.path.abspath(FIG))
