"""Build data/daily_claims.csv from the insurer's daily export data/raw/qrc_daily_dataset.csv.

  python pipeline/00_prepare_data.py
"""
import argparse
import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "qrc_daily_dataset.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "daily_claims.csv")
START, END = "2022-01-01", "2026-08-20"


def build(raw):
    raw = raw.rename(columns={"claim_count_maj": "property_damage_claim_count",
                              "policy_live_total": "active_policy_count"})
    raw["date"] = pd.to_datetime(raw["date"])
    d = raw[(raw.date >= START) & (raw.date <= END)].sort_values("date").reset_index(drop=True)
    assert (d.date.diff().dropna().dt.days == 1).all(), "days must be consecutive"
    t, t1 = d.iloc[:-1].reset_index(drop=True), d.iloc[1:].reset_index(drop=True)
    df = pd.DataFrame(dict(
        input_date=t.date.dt.date, target_date=t1.date.dt.date,
        property_damage_claim_count_t=t.property_damage_claim_count.astype(int),
        active_policy_count_t=t.active_policy_count.astype(int),
        day_of_week_t_plus_1=t1.day_of_week.astype(int),                 # 1 = Monday ... 7 = Sunday
        is_public_holiday_t_plus_1=t1.is_public_holiday.astype(int),
        target_property_damage_claim_count=t1.property_damage_claim_count.astype(int)))
    target = pd.to_datetime(df.target_date)
    df["split"] = np.where(target <= "2024-12-31", "train", np.where(target <= "2025-12-31", "validation", "test"))
    tr = df[df.split == "train"]
    for col, name in (("property_damage_claim_count_t", "C_t_z"), ("active_policy_count_t", "P_t_z")):
        df[name] = (df[col] - tr[col].mean()) / tr[col].std(ddof=0)
    df["weekday_angle"] = 2 * np.pi * (df.day_of_week_t_plus_1 - 1) / 7
    df["D_sin"], df["D_cos"] = np.sin(df.weekday_angle), np.cos(df.weekday_angle)
    df["weekday_radius"] = np.hypot(df.D_sin, df.D_cos)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW, help="daily export with columns date, day_of_week, "
                    "is_public_holiday, claim_count_maj, policy_live_total")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    df = build(pd.read_csv(args.raw))
    df.to_csv(args.out, index=False)
    print(df.split.value_counts().to_string(), "\nwritten", args.out)


if __name__ == "__main__":
    main()
