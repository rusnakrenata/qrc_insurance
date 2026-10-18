"""Capacity-decomposition figure from results/capacity (no reservoir runs).

  python analysis/plot_nonlinear_capacity.py

Writes figures/fig_capacity.pdf/.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "capacity")
FIG = os.path.join(HERE, "..", "figures")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
PARTS = [("C_current_linear", "current input, linear", "#C9CDD0"),
         ("C_current_nonlinear", "current input, nonlinear", "#8a8986"),
         ("C_past_linear", "past inputs, linear (memory)", "#00ABE5"),
         ("C_past_nonlinear", "past inputs, nonlinear", "#AA54A0")]
COL = {1: "#00ABE5", 2: "#E21A22", 3: "#2B3335", 4: "#238037", 5: "#AA54A0"}


def main():
    s = pd.read_csv(os.path.join(RES, "capacity_summary.csv")).sort_values("candidate_id")
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    x = range(len(s))
    bottom = [0.0] * len(s)
    for col, label, color in PARTS:
        ax[0].bar(x, s[col], bottom=bottom, color=color, width=0.62, label=label, edgecolor="white", lw=0.6)
        bottom = [b + v for b, v in zip(bottom, s[col])]
    ax[0].errorbar(x, s.C_total, yerr=s.C_total_sd, fmt="none", ecolor="#2B3335", lw=0.7, capsize=2)
    ax[0].scatter(x, s.feature_rank, marker="_", s=120, color="#2B3335", zorder=3, label="feature rank")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([f"C{c}\n$W$={w}" for c, w in zip(s.candidate_id, s.window)])
    ax[0].set_ylabel("capacity (sum over targets)")
    ax[0].set_ylim(0, 19.5)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    ax[0].text(-0.12, 1.02, "(a)", transform=ax[0].transAxes)

    for _, r in s.iterrows():
        c = int(r.candidate_id)
        ax[1].errorbar(r.C_past_linear, r.C_past_nonlinear, yerr=r.C_past_nonlinear_sd, fmt="o", ms=4,
                       color=COL[c], lw=0.7, capsize=1.5)
        ax[1].annotate(f"C{c}", (r.C_past_linear, r.C_past_nonlinear), xytext=(4, 2), textcoords="offset points",
                       fontsize=6.5)
    ax[1].set_xlabel("past inputs, linear capacity")
    ax[1].set_ylabel("past inputs, nonlinear capacity")
    ax[1].set_xlim(0.6, 2.7)
    ax[1].set_ylim(-0.1, 2.3)
    ax[1].text(-0.02, 1.03, "(b)", transform=ax[1].transAxes, ha="right")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_capacity.{ext}"), dpi=300)


if __name__ == "__main__":
    main()
