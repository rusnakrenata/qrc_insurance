"""Memory figure of the frozen candidates from results/memory/.

  python analysis/plot_final_candidate_memory.py

Writes figures/fig_memory.pdf/.png and results/memory/memory_vs_forecast.csv.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = {1: "#00ABE5", 2: "#E21A22", 3: "#2B3335", 4: "#238037", 5: "#AA54A0"}


def main():
    curve = pd.read_csv(os.path.join(RES, "memory", "memory_curve.csv"))
    summ = pd.read_csv(os.path.join(RES, "memory", "memory_summary.csv"))
    hw = pd.read_csv(os.path.join(RES, "tables", "hardware_by_candidate.csv"))
    ha = hw[hw.approach == "hardware-aware"].set_index("candidate")
    runs = pd.read_csv(os.path.join(RES, "runs.csv"))
    table = summ.set_index("candidate")[["window", "mc_ch_1_20", "mc_ch_1_20_sd", "k_eff", "k_eff_95", "k_eff_001"]]
    table["ideal_rmse_2026"] = ha.ideal_rmse
    table["ha_qpu_rmse_mean"] = ha.qpu_rmse_mean
    table["ha_d_median"] = ha.d_median
    table["duration_us"] = runs.groupby("candidate").max_duration_us.median()
    table.reset_index().to_csv(os.path.join(RES, "memory", "memory_vs_forecast.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[2.2, 1]))
    tau = summ.tau99.max()
    for cand, g in curve.groupby("candidate"):
        g = g.sort_values("delay")
        w = int(g.window.iloc[0])
        k = int(summ.loc[summ.candidate == cand, "k_eff"].iloc[0])
        ax[0].plot(g.delay, g.mc_mean, color=COL[cand], label=f"C{cand} ($W$={w}, $K_{{\\rm eff}}$={k})")
        ax[0].fill_between(g.delay, (g.mc_mean - g.mc_sd).clip(lower=1e-4), g.mc_mean + g.mc_sd,
                           color=COL[cand], alpha=0.2, lw=0)
    ax[0].axhline(tau, color="#8a8986", ls="--", lw=0.7)
    ax[0].text(119, tau * 1.15, "null 99 %", ha="right", va="bottom", fontsize=6.5, color="#8a8986")
    ax[0].axvline(20, color="#8a8986", ls=":", lw=0.7)
    ax[0].text(21, 0.3, "Rule-2 sum\nends at $k$=20", fontsize=6.5, color="#8a8986", va="top")
    ax[0].set_yscale("log")
    ax[0].set_ylim(5e-4, 0.5)
    ax[0].set_xlim(1, 120)
    ax[0].set_xlabel("delay $k$ (days)")
    ax[0].set_ylabel(r"$\overline{MC}_k$ (mean of 4 input sequences)")
    ax[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    ax[0].text(-0.1, 1.02, "(a)", transform=ax[0].transAxes)

    for cand, r in table.iterrows():
        ax[1].scatter(r.k_eff, r.ideal_rmse_2026, color=COL[cand], marker="o", s=18, zorder=3)
        ax[1].scatter(r.k_eff, r.ha_qpu_rmse_mean, facecolor="white", edgecolor=COL[cand], marker="o", s=18, zorder=3)
        ax[1].annotate(f"C{cand}", (r.k_eff, r.ha_qpu_rmse_mean), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax[1].scatter([], [], color="#2B3335", s=18, label="ideal")
    ax[1].scatter([], [], facecolor="white", edgecolor="#2B3335", s=18, label="IBM, HA")
    ax[1].set_xscale("symlog", linthresh=2)
    ax[1].set_xticks([1, 3, 10, 25])
    ax[1].set_xticklabels(["1", "3", "10", "25"])
    ax[1].set_xlabel(r"$K_{\rm eff}$ (days)")
    ax[1].set_ylabel("2026 RMSE (claims/day)")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].text(-0.25, 1.02, "(b)", transform=ax[1].transAxes)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_memory.{ext}"), dpi=300)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
