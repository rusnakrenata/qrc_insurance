"""Training-period analysis behind the choice of inputs (F2 -> F3 -> F4).

  python pipeline/02_data_analysis.py

Writes results/eda/*.csv and figures/eda_target.png.
"""
import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, f_oneway, kruskal, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from classical import models as M  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "eda")
FIG = os.path.join(HERE, "..", "figures")
C, P, D, H = "property_damage_claim_count_t", "active_policy_count_t", "day_of_week_t_plus_1", "is_public_holiday_t_plus_1"
Y = "target_property_damage_claim_count"
NAMES = {"C_t": C, "D_t+1": D, "P_t": P, "H_t+1": H}
DISCRETE = {"C_t": True, "D_t+1": True, "P_t": False, "H_t+1": True}
SEED, N_PERM = 42, 500


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"\n{name}\n{df.round(4).to_string(index=False)}")

# ordinary least squares regression with a constant, given a target and one or more blocks of covariates
def ols(y, *blocks):
    X = sm.add_constant(pd.concat(blocks, axis=1), has_constant="add")
    return sm.OLS(y, X).fit()


def audit(df):
    """Structural checks over the whole series: one-day horizon, continuity, alignment."""
    t, i = pd.to_datetime(df.target_date), pd.to_datetime(df.input_date)
    checks = [("one-day horizon", ((t - i).dt.days == 1).all()),
              ("unique target dates", t.is_unique),
              ("consecutive days", (t.diff().dropna().dt.days == 1).all()),
              ("C_t equals previous target", (df[C].values[1:] == df[Y].values[:-1]).all()),
              ("weekday belongs to target day", (df[D] == t.dt.dayofweek + 1).all()),
              ("no missing values", df[[C, P, D, H, Y]].notna().all().all()),
              ("splits chronological", df.split.tolist() == sorted(df.split, key=["train", "validation", "test"].index))]
    return pd.DataFrame([dict(check=k, status="PASS" if v else "FAIL") for k, v in checks])


def screen_raw(path):
    """Relevance of every numeric column of day t for C_{t+1}, and its correlation with active policies.
    Training rows only (target dates 2022-01-02 .. 2024-12-31)."""
    raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw["target"] = raw["claim_count_maj"].shift(-1)
    tr = raw[(raw.date >= "2022-01-01") & (raw.date <= "2024-12-30")].dropna(subset=["target"])
    rows = []
    for col in raw.columns.drop(["date", "target"]):
        if not pd.api.types.is_numeric_dtype(raw[col]) or tr[col].nunique() < 2:
            continue
        rows.append(dict(column=col, pearson_with_target=pearsonr(tr[col], tr.target)[0],
                         spearman_with_target=spearmanr(tr[col], tr.target)[0],
                         pearson_with_active_policies=pearsonr(tr[col], tr["policy_live_total"])[0]))
    return pd.DataFrame(rows).sort_values("pearson_with_target", key=abs, ascending=False)


def r2_gain_permutation(y, base, extra, strata, n_perm, seed):
    """Delta R^2 of `extra` over `base`, and a permutation p-value with `extra` shuffled within strata."""
    r0 = ols(y, base).rsquared
    obs = ols(y, base, extra).rsquared - r0
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(n_perm):
        shuffled = extra.copy()
        for _, idx in strata.groupby(strata).groups.items():
            shuffled.loc[idx] = rng.permutation(extra.loc[idx].values)
        hits += ols(y, base, shuffled).rsquared - r0 >= obs
    return obs, (1 + hits) / (n_perm + 1)


def feature_set_gate(df):
    """2025 RMSE of Ridge (CV-chosen penalty), Poisson and NB2 GLMs on F2, F3, F4 (exposure as standardized covariate)."""
    tr, va = df.iloc[:1095], df.iloc[1095:1460]
    alphas = [1, 2, 3, 5, 6, 6.25, 6.5, 6.6, 6.75, 7, 7.5, 8, 9, 10, 15, 20, 25, 30, 50, 75, 100]
    rows = []
    for f in ("F2", "F3", "F4"):
        cv = {a: np.mean([M.rmse(M.ridge_fit_predict(tr.iloc[i], tr.iloc[j], f, a)[0], tr[Y].iloc[j])
                          for i, j in M.folds(len(tr))]) for a in alphas}
        a = min(cv, key=cv.get)
        rows.append(dict(model="Ridge", feature_set=f, rmse_2025=M.rmse(M.ridge_fit_predict(tr, va, f, a)[0], va[Y])))
        v = "no_policy" if f == "F2" else "standardized_policy"
        for fam in ("poisson", "nb"):
            rows.append(dict(model=fam, feature_set=f, rmse_2025=M.rmse(M.glm_fit_predict(tr, va, f, v, fam)[0], va[Y])))
    out = pd.DataFrame(rows).pivot(index="model", columns="feature_set", values="rmse_2025").reset_index()
    out["gain_F2_F3"], out["gain_F3_F4"] = out.F2 - out.F3, out.F3 - out.F4
    return out


def mutual_information(x, y, discrete, seed):
    return float(mutual_info_regression(np.asarray(x).reshape(-1, 1), y, discrete_features=[discrete],
                                        random_state=seed)[0])


def relevance(tr):
    """Linear and nonlinear relevance of each input for C_{t+1}, with a permutation baseline for MI."""
    y, rng, rows = tr[Y].values.astype(float), np.random.default_rng(SEED), []
    for name, col in NAMES.items():
        x = tr[col].values
        mi = mutual_information(x, y, DISCRETE[name], SEED)
        perm = np.array([mutual_information(x, rng.permutation(y), DISCRETE[name], SEED + k + 1) for k in range(N_PERM)])
        rows.append(dict(feature=name, pearson=pearsonr(x, y)[0], spearman=spearmanr(x, y)[0], mutual_information=mi,
                         permutation_95pct=np.quantile(perm, .95),
                         permutation_pvalue=(1 + np.sum(perm >= mi)) / (N_PERM + 1)))
    return pd.DataFrame(rows).sort_values("mutual_information", ascending=False)


def main():
    matplotlib.use("Agg") 
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "data", "raw", "qrc_daily_dataset.csv"))
    ap.add_argument("--permutations", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    if os.path.exists(args.raw):
        save(screen_raw(args.raw), "raw_column_screening.csv")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "daily_claims.csv")
    df = pd.read_csv(path, parse_dates=["input_date", "target_date"])
    tr = df.iloc[:1095].reset_index(drop=True)
    y = tr[Y].astype(float)
    save(audit(df), "leakage_audit.csv")

    # target: level, overdispersion, smoothness
    save(pd.DataFrame([dict(n=len(y), mean=y.mean(), median=y.median(), sd=y.std(), variance=y.var(),
                            dispersion_ratio=y.var() / y.mean(), zero_share=(y == 0).mean(), max=y.max())]),
         "target_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    dates = pd.to_datetime(tr.target_date)
    ax.plot(dates, y, color="#B0B7BB", lw=.6, label="daily")
    ax.plot(dates, y.rolling(7).mean(), color="#00ABE5", lw=1.5, label="7-day mean")
    ax.plot(dates, y.rolling(30).mean(), color="#2B3335", lw=1.5, label="30-day mean")
    ax.set_ylabel("claims per day"); ax.legend(frameon=False, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eda_target.png"), dpi=200); plt.close(fig)

    # memory: autocorrelation of the target
    save(pd.DataFrame([dict(lag=k, acf=y.autocorr(k)) for k in range(1, 31)]), "autocorrelation.csv")

    # calendar: weekday profile and holiday effect
    save(tr.groupby(D)[Y].agg(n="count", mean="mean", variance="var").reset_index(), "weekday.csv")
    groups = [g[Y].values for _, g in tr.groupby(D)]
    ss_b = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
    save(pd.DataFrame([dict(anova_F=f_oneway(*groups)[0], anova_p=f_oneway(*groups)[1],
                            eta_squared=ss_b / ((y - y.mean()) ** 2).sum(), kruskal_p=kruskal(*groups)[1])]),
         "weekday_anova.csv")
    hol, work = y[tr[H] == 1], y[tr[H] == 0]
    pooled = np.sqrt(((len(hol) - 1) * hol.var() + (len(work) - 1) * work.var()) / (len(hol) + len(work) - 2))
    save(pd.DataFrame([dict(holidays=len(hol), mean_holiday=hol.mean(), mean_other=work.mean(),
                            reduction_pct=100 * (1 - hol.mean() / work.mean()),
                            cohens_d=(hol.mean() - work.mean()) / pooled)]), "holiday.csv")
    dow = pd.get_dummies(tr[D].astype(str), prefix="dow", drop_first=True, dtype=float)
    gain, p = r2_gain_permutation(y, dow, tr[[H]].astype(float), tr[D], args.permutations, SEED)
    save(pd.DataFrame([dict(r2_weekday=ols(y, dow).rsquared, delta_r2_holiday=gain, stratified_perm_p=p,
                            permutations=args.permutations)]), "holiday_beyond_weekday.csv")

    week = pd.get_dummies(tr[D].astype(str), prefix="dow", drop_first=True, dtype=float)
    hold = tr[[H]].astype(float)

    # exposure: claim frequency per year, and value of P_t beyond the calendar and a time trend
    allyears = df.assign(year=pd.to_datetime(df.target_date).dt.year).groupby("year").agg(
        n_days=(Y, "size"), total_claims=(Y, "sum"), total_policy_days=(P, "sum"), mean_active_policies=(P, "mean"))
    allyears["claims_per_100k_policy_days"] = 1e5 * allyears.total_claims / allyears.total_policy_days
    save(allyears.reset_index(), "yearly_claim_rates.csv")
    pz = ((tr[P] - tr[P].mean()) / tr[P].std()).rename("P_t").to_frame()
    trend = pd.Series(np.arange(len(tr)) / (len(tr) - 1), name="time").to_frame()
    models = {"Exposure only": [pz], "Calendar": [week, hold], "Calendar + P_t": [week, hold, pz],
              "Calendar + time": [week, hold, trend], "Calendar + time + P_t": [week, hold, trend, pz]}
    save(pd.DataFrame([dict(model=k, r_squared=(m := ols(y, *v)).rsquared, adjusted_r_squared=m.rsquared_adj,
                            aic=m.aic, bic=m.bic) for k, v in models.items()]), "exposure_models.csv")

    # lagged claims: incremental R^2 of one older lag beyond calendar + P_t + C_t
    base_blocks = [week, hold, pz, tr[[C]].astype(float)]
    base = ols(y, *base_blocks).rsquared
    hist = df[Y].shift(1).iloc[:1095].reset_index(drop=True)          # C_t, indexed like the training rows
    rows = []
    for k in [2, 3, 7, 14, 21, 28]:                                   # lag k = C_{(t+1)-k}
        lag = hist.shift(k - 1).rename(f"lag{k}")
        ok = lag.notna()
        r2_base = ols(y[ok], *[b[ok] for b in base_blocks]).rsquared
        r2 = ols(y[ok], *[b[ok] for b in base_blocks], lag[ok].to_frame()).rsquared
        rows.append(dict(lag=k, n=int(ok.sum()), delta_r_squared=r2 - r2_base))
    save(pd.DataFrame(rows).assign(base_r_squared=base), "lag_relevance.csv")

    # nonlinear relevance (mutual information with permutation baseline)
    save(relevance(tr), "nonlinear_relevance.csv")

    # redundancy among the F4 inputs
    Xv = pd.concat([((tr[C] - tr[C].mean()) / tr[C].std()).rename("C_t"), pz,
                    tr[H].astype(float).rename("H_t+1"),
                    pd.get_dummies(tr[D].astype(int).astype(str), prefix="D", drop_first=True, dtype=float)], axis=1)
    Xc = sm.add_constant(Xv)
    save(pd.DataFrame([dict(variable=c, vif=variance_inflation_factor(Xc.values, i))
                       for i, c in enumerate(Xc.columns) if c != "const"]).sort_values("vif", ascending=False), "vif.csv")
    table = pd.crosstab(tr[D], tr[H])
    chi2 = chi2_contingency(table, correction=False)[0]
    save(pd.DataFrame([dict(max_abs_pairwise_pearson=tr[[C, P, D, H]].corr().where(~np.eye(4, dtype=bool)).abs().max().max(),
                            cramers_v_D_vs_H=np.sqrt(chi2 / (len(tr) * (min(table.shape) - 1))))]), "redundancy_summary.csv")

    # stability of relevance across training years
    rows = []
    for year, g in tr.groupby(pd.to_datetime(tr.target_date).dt.year):
        for name, col in NAMES.items():
            rows.append(dict(year=year, feature=name, pearson=pearsonr(g[col], g[Y])[0],
                             spearman=spearmanr(g[col], g[Y])[0],
                             mutual_information=mutual_information(g[col].values, g[Y].values.astype(float),
                                                                   DISCRETE[name], SEED)))
    save(pd.DataFrame(rows), "regime_by_year.csv")

    # the gate that fixed the input set before the nonlinear benchmark (2025 used, 2026 not)
    save(feature_set_gate(df), "feature_set_gate.csv")


if __name__ == "__main__":
    main()
