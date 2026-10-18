"""Write the LaTeX tables of the article from the result CSVs (numbers are never typed by hand)."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(RES, "tables")
OUT = os.path.join(HERE, "..", "article", "tables")
os.makedirs(OUT, exist_ok=True)
POL = {"hardware-aware": "HA", "random": "Random", "qiskit": "Qiskit"}


def write(name, body):
    import re
    body = re.sub(r"(?<![\w${^-])-(?=\d)", "$-$", body)   # typographic minus for negative numbers
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)


def f2(x):
    return f"{x:.2f}"


def ci(d, lo, hi):
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def forecasting():
    t = pd.read_csv(os.path.join(TAB, "forecasting_comparison.csv"))
    names = {"ridge": "Ridge", "nb": "NB2 GLM", "esn": "ESN$^\\dagger$", "gru": "GRU$^\\dagger$",
             "rnn": "Vanilla RNN$^\\dagger$", "xgboost_squared": "XGBoost (squared)",
             "poisson": "Poisson GLM", "xgboost_poisson": "XGBoost (Poisson)"}
    rows = []
    for _, r in t.iterrows():
        if r.model in names:
            n = names[r.model]
        elif "ideal" in r.model:
            n = r.model.replace("QRC-", "").replace(" ideal", ", ideal")
        else:
            n = r.model.replace("QRC-", "").replace(" HA hardware", ", QPU (HA)$^\\ddagger$")
        rows.append((n, r))
    order = ["Ridge", "NB2 GLM", "ESN$^\\dagger$", "GRU$^\\dagger$", "Vanilla RNN$^\\dagger$", "XGBoost (squared)",
             "Poisson GLM", "XGBoost (Poisson)"]
    cls = [x for o in order for x in rows if x[0] == o]
    qrc = [x for x in rows if x[0].startswith("C")]
    qrc = sorted(qrc, key=lambda x: (x[0][:2], "QPU" in x[0]))
    lines = []
    for block in (cls, qrc):
        for n, r in block:
            lines.append(f"{n} & {f2(r.rmse)} & {f2(r.mae)} & {r.bias:+.2f} & {f2(r.crmse)} & "
                         f"{ci(r.diff_vs_ridge, r.diff_vs_ridge_lo, r.diff_vs_ridge_hi)} & "
                         f"{ci(r.diff_vs_xgboost_squared, r.diff_vs_xgboost_squared_lo, r.diff_vs_xgboost_squared_hi)} \\\\")
        lines.append("\\midrule")
    body = ("\\begin{tabular}{@{}lccccc c@{}}\n\\toprule\n"
            "Model & RMSE & MAE & Bias & cRMSE & $\\Delta$ vs Ridge & $\\Delta$ vs XGBoost (sq.) \\\\\n\\midrule\n"
            + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_forecast", body)


def candidates():
    s = pd.read_csv("/dev/stdin" if False else os.path.join(HERE, "..", "configs", "fold_local_cv_summary.csv"))
    runs = pd.read_csv(os.path.join(RES, "runs.csv"))
    res = runs.groupby("candidate").agg(cz=("cz_per_feature_vector", "first"), depth=("max_depth", "first"),
                                        dur=("max_duration_us", "first"))
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from qrc.candidates import CANDIDATES
    lines = []
    for k, c in CANDIDATES.items():
        r = s[s.candidate == k].iloc[0]
        w = f"{c['window']}" + ("$^{*}$" if c["protocol"] == "CONT" else "")
        lines.append(f"C{k} & {c['graph']} & {w} & {c['r']} & {c['alpha']:.2f} & {c['hx']:.3f} & {c['hy']:+.3f} & "
                     f"$10^{{{int(round(__import__('math').log10(c['lam'])))}}}$ & {len(c['features'])} & "
                     f"{r.corrected_cv:.2f} & {r.corrected_rmse:.2f} & {int(res.loc[k, 'cz'])} & {res.loc[k, 'dur']:.0f} \\\\")
    body = ("\\begin{tabular}{@{}lcccccccccccc@{}}\n\\toprule\n"
            "& Graph & $W$ & $r$ & $a$ & $h_x$ & $h_y$ & $\\lambda$ & $m$ & CV & 2025 & CZ & $t_{\\rm c}$ ($\\mu$s) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_candidates", body)


def hardware():
    t = pd.read_csv(os.path.join(TAB, "hardware_by_candidate.csv"))
    lines = []
    for c in range(1, 6):
        g = t[t.candidate == c].set_index("approach")
        first = True
        for ap in ["hardware-aware", "random", "qiskit"]:
            r = g.loc[ap]
            lead = f"\\multirow{{3}}{{*}}{{C{c} ({r.ideal_rmse:.2f})}}" if first else ""
            first = False
            lines.append(f"{lead} & {POL[ap]} & {int(r.rounds)} & {r.qpu_rmse_mean:.2f} $\\pm$ {r.qpu_rmse_sd:.2f} & "
                         f"{r.bias_mean:+.2f} & {r.d_median:.2f} & {r.d_max:.2f} & {r.level_share_median:.2f} \\\\")
        lines.append("\\midrule" if c < 5 else "")
    body = ("\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
            "Cand. (ideal) & Policy & $n$ & RMSE & Bias & $d$ median & $d$ max & $\\rho_{\\rm lvl}$ \\\\\n\\midrule\n"
            + "\n".join(x for x in lines if x) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_hardware", body)


def pooled():
    p = pd.read_csv(os.path.join(TAB, "hardware_pooled.csv")).set_index("approach")
    rows = [("Runs", "runs", "{:.0f}"), ("Forecast RMSE, median", "qpu_rmse_median", "{:.2f}"),
            ("Forecast RMSE, worst run", "qpu_rmse_max", "{:.2f}"),
            ("$d$, median", "d_median", "{:.2f}"), ("$d$, 90th percentile", "d_p90", "{:.2f}"),
            ("$d$, worst run", "d_max", "{:.2f}"), ("Runs with $d>6$", "d_over_6", "{:.0f}"),
            ("Runs with $d>8$", "severe_d_over_8", "{:.0f}"),
            ("Layouts with a CZ error $>4$\\,\\%", "flagged_coupler", "{:.0f}"),
            ("Within-candidate range of $d$, median", "d_range_within_candidate_median", "{:.2f}"),
            ("Level-shift share $\\rho_{\\rm lvl}$, median", "level_share_median", "{:.2f}"),
            ("Aer $d$ (exact expectations), median", "noisy_direct_d_median", "{:.2f}")]
    lines = [f"{lab} & " + " & ".join(fmt.format(p.loc[a, col]) for a in ["hardware-aware", "random", "qiskit"]) + " \\\\"
             for lab, col, fmt in rows]
    body = ("\\begin{tabular}{@{}lccc@{}}\n\\toprule\n & HA & Random & Qiskit \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_pooled", body)


def paired():
    p = pd.read_csv(os.path.join(TAB, "paired_layouts.csv"))
    lab = {"hardware-aware vs random": "HA $-$ Random", "hardware-aware vs qiskit": "HA $-$ Qiskit",
           "random vs qiskit": "Random $-$ Qiskit"}
    lines = []
    for _, r in p.iterrows():
        m = "RMSE" if r.metric == "rmse" else "$d$"
        lines.append(f"{lab[r.comparison]} & {m} & {int(r.a_better)}/{int(r.rounds)} & {r.median_diff:+.2f} & "
                     f"{r.mean_diff:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}] & {r.sign_test_p:.3f} \\\\")
    body = ("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
            "Comparison & Metric & First lower & Median $\\Delta$ & Mean $\\Delta$ [95\\% CI] & Sign test $p$ \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_paired", body)


def predictors():
    p = pd.read_csv(os.path.join(TAB, "distortion_predictors.csv"))
    names = ["HA probe $d$ (100 training dates)", "HA probe level shift $\\to$ realised level shift",
             "$\\log_{10}$ workload error proxy $L$", "$\\log_{10} L$, layouts without degraded coupler",
             "Calibrated Aer simulation $d$"]
    lines = []
    for n, (_, r) in zip(names, p.iloc[:5].iterrows()):
        lines.append(f"{n} & {r.scope.replace('all approaches', 'all')} & {int(r.n)} & {r.pearson_r:.2f} & "
                     f"{r.spearman_rho:.2f} & {r.loo_r2:.2f} & {r.loo_mae:.2f} \\\\")
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Pre-deployment indicator & Runs & $n$ & $r$ & $\\rho_s$ & LOO $R^2$ & LOO MAE \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_predictors", body)


def encoding():
    e = pd.read_csv(os.path.join(TAB, "ideal_encoding_range.csv"))
    lines = []
    for _, r in e.iterrows():
        if r.period == "test 2026":
            continue
        lines.append(f"{r.period} & {r.n} & {r.claims_mean:.2f} & {100 * r.claims_clipped_share:.1f} & "
                     f"{r.exposure_min/1000:.1f}--{r.exposure_max/1000:.1f} & {r.exposure_angle_min:.2f}--{r.exposure_angle_max:.2f} & "
                     f"{r.claim_frequency_per_1000:.3f} \\\\")
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            "Period & $n$ & Mean $C$ & Clipped (\\%) & Policies ($10^3$) & $\\theta_P/a$ & Claims per $10^3$ pol. \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_encoding", body)


def refit():
    r = pd.read_csv(os.path.join(TAB, "ideal_refit_2025.csv"))
    lines = []
    for _, x in r.iterrows():
        n = f"C{x.candidate}" if str(x.candidate) != "Ridge" else "Ridge"
        lines.append(f"{n} & {x.rmse_2025:.2f} & {x.rmse_2026:.2f} & {x.bias_2026:+.2f} & {x.rmse_2026_refit:.2f} & {x.bias_2026_refit:+.2f} \\\\")
    body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Readout fitted on 2022--24} & \\multicolumn{2}{c}{Refitted on 2022--25} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-6}\n"
            "Model & RMSE 2025 & RMSE 2026 & Bias 2026 & RMSE 2026 & Bias 2026 \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_refit", body)


def runs_appendix():
    r = pd.read_csv(os.path.join(RES, "runs.csv"))
    lines = []
    for _, x in r.iterrows():
        lay = ",".join(map(str, json.loads(x.layout)))
        lines.append(f"C{x.candidate} & {x['round']} & {POL[x.approach][:2]} & [{lay}] & "
                     f"{100 * x.cal_max_cz:.2f} & {x.qpu_rmse:.2f} & {x.qpu_bias:+.2f} & {x.transfer_d:.2f} & {x.level_share:.2f} \\\\")
    half = (len(lines) + 1) // 2
    head = ("C & R & Pol. & Physical layout & CZ\\,\\% & RMSE & Bias & $d$ & $\\rho_{\\rm lvl}$ \\\\\n\\midrule\n")
    def block(ls):
        return "\\begin{tabular}[t]{@{}lclp{2.75cm}ccccc@{}}\n\\toprule\n" + head + "\n".join(ls) + "\n\\bottomrule\n\\end{tabular}"
    body = block(lines[:half]) + "\\hfill\n" + block(lines[half:]) + "\n"
    write("tab_allruns", body)


def memory():
    m = pd.read_csv(os.path.join(RES, "memory", "memory_vs_forecast.csv"))
    rows = [f"C{int(r.candidate)} & {int(r.window)} & {r.mc_ch_1_20:.2f} ({r.mc_ch_1_20_sd:.2f}) & {int(r.k_eff)} & "
            f"{r.ideal_rmse_2026:.2f} & {r.ha_qpu_rmse_mean:.2f} & {r.ha_d_median:.2f} & {r.duration_us:.0f} \\\\"
            for r in m.itertuples()]
    write("tab_memory", "\\begin{tabular}{@{}lrcrcccr@{}}\n\\toprule\n"
          " & $W$ & $\\mathrm{MC}_{\\rm ch}^{1:20}$ & $K_{\\rm eff}$ & Ideal & HA & HA $d$ & $\\mu$s \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def classical_cfg():
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from classical import models as M
    from qrc.candidates import load_data
    cfg = json.load(open(os.path.join(HERE, "..", "configs", "classical_selected.json")))
    train = load_data().iloc[:1095]
    val = pd.read_csv(os.path.join(RES, "classical_validation", "validation_2025.csv"))
    names = [("baseline", "Baseline"), ("ridge", "Ridge"), ("poisson", "Poisson"), ("nb", "NB2"), ("esn", "ESN"),
             ("rnn", "RNN"), ("gru", "GRU"), ("xgboost_squared", "XGB squared"), ("xgboost_poisson", "XGB Poisson")]
    keys = {"ridge": ["alpha"],
            "esn": ["window_length", "reservoir_size", "requested_connectivity", "spectral_radius", "input_scaling",
                    "leak_rate", "ridge_alpha"],
            "rnn": ["window", "hidden_size", "layers", "learning_rate", "weight_decay", "epochs"],
            "gru": ["window", "hidden_size", "layers", "learning_rate", "weight_decay", "epochs"],
            "xgboost_squared": ["max_depth", "learning_rate", "min_child_weight", "reg_lambda", "n_estimators"],
            "xgboost_poisson": ["max_depth", "learning_rate", "min_child_weight", "reg_lambda", "n_estimators"]}
    rows = []
    for key, label in names:
        for rank, c in enumerate(cfg[key], 1):
            if key == "baseline":
                text = c["name"]
            elif key in ("poisson", "nb"):
                _, fit = M.glm_fit_predict(train, train, c["feature_set"], c["variant"], key)
                parts = [f"variant={c['variant']}"]
                if key == "nb":
                    parts.append(f"alpha_dispersion={float(fit.params['alpha']):.4f}")
                if c["variant"] == "fixed_log_policy_offset":
                    parts.append("gamma=1.0")
                elif c["variant"] == "learned_log_policy":
                    parts.append(f"gamma={float(fit.params['log_P_centered']):.3f}")
                text = "; ".join(parts)
            else:
                text = "; ".join(f"{k}={c[k]}" for k in keys[key])
            v = val[(val.method == key) & (val["rank"] == rank)].iloc[0]
            cv = f"{c['cv_rmse']:.3f}" if "cv_rmse" in c else "--"
            mark = "$\\checkmark$" if v.selected else ""
            rows.append(f"{label} & {rank if key != 'baseline' else '--'} & {cv} & {v.rmse_2025:.3f} & {mark} & {text.replace('_', chr(92) + '_')} \\\\")
    write("tab_classical_cfg", "\\begin{tabular}{@{}llcccp{11.4cm}@{}}\n\\toprule\nMethod & CV rank & CV & 2025 & Sel. & "
          "Configuration \\\\ \\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def capacity():
    c = pd.read_csv(os.path.join(RES, "capacity", "capacity_summary.csv"))
    rows = [f"C{int(r.candidate_id)} & {int(r.feature_rank)} / {r.effective_rank:.1f} & {r.C_current_linear:.2f} & "
            f"{r.C_current_nonlinear:.2f} & {r.C_past_linear:.2f} & {r.C_past_nonlinear:.2f} & "
            f"{r.C_total:.1f} ({r.C_total_sd:.1f}) \\\\" for r in c.itertuples()]
    write("tab_capacity", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
          " & Rank & \\multicolumn{2}{c}{Current} & \\multicolumn{2}{c}{Past} & Total \\\\\n"
          " & & lin. & nonlin. & lin. & nonlin. & \\\\\n\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    forecasting()
    candidates()
    hardware()
    pooled()
    paired()
    predictors()
    encoding()
    refit()
    runs_appendix()
    memory()
    classical_cfg()
    capacity()
    print("tables written to", os.path.abspath(OUT))
