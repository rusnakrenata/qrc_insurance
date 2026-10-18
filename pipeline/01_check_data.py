"""Check chronology, one-day-ahead alignment and the train/validation/test split.

  python pipeline/01_check_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qrc.candidates import load_data  # noqa: E402

df = load_data()
C, Y = "property_damage_claim_count_t", "target_property_damage_claim_count"

# consecutive days, and the target of day t is the claim count of day t+1
assert df.target_date.is_unique and (df.target_date.diff().dropna().dt.days == 1).all()
assert ((df.target_date - df.input_date).dt.days == 1).all()
assert (df[C].values[1:] == df[Y].values[:-1]).all()
# calendar inputs refer to the target day and are known one day ahead
assert (df.day_of_week_t_plus_1 == df.target_date.dt.dayofweek + 1).all()

split = df.split.value_counts()
assert (split.train, split.validation, split.test) == (1095, 365, 232)
assert df.iloc[:1095].split.eq("train").all() and df.iloc[1095:1460].split.eq("validation").all()
assert np.isfinite(df[[C, "active_policy_count_t", Y]].values).all()

for name, part in df.groupby("split", sort=False):
    print(f"{name:10s} {part.target_date.min().date()} .. {part.target_date.max().date()}  n={len(part):4d}  "
          f"mean claims={part[Y].mean():.2f}  policies={part.active_policy_count_t.min()}-{part.active_policy_count_t.max()}")
print("data checks passed")
