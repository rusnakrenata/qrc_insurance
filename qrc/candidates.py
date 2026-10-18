"""The five reservoirs frozen before the final IBM Kingston evaluation, and data loading."""
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "daily_claims.csv")

# Logical coupling graphs (qubit 0..3 = C, D, P, H; 4,5 = memory M1, M2).
GRAPHS = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H1": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(2, 3), (1, 2), (0, 1), (0, 4), (4, 5)],
    "H4": [(1, 2), (0, 1), (0, 4), (3, 4), (4, 5)],
}

_H3 = dict(graph="H3", alpha=0.25, dt=1.6, hx=0.777492389747048, hy=-0.293484785757933,
           J={(2, 3): 0.2399121685505184, (1, 2): 0.2776392540871816, (0, 1): 1.5177767277422014,
              (0, 4): 0.839147304059147, (4, 5): -0.4389458357764761}, r=3)
_H2 = dict(graph="H2", alpha=0.75, dt=1.6, hx=0.490662724735852, hy=0.397549311143186,
           J={(0, 1): -0.08344036148982822, (1, 2): 0.2472585102744692, (2, 3): 1.928732883596244,
              (2, 4): 0.5130876835350422, (4, 5): -0.3864902188635002}, r=2)

RAW = {
    1: dict(_H3, window=4, protocol="RWP", lam=1e-4, settings=["XXXZYX", "ZZZZZZ"],
            features=["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3", "YX45"]),
    2: dict(_H3, window=2, protocol="RWP", lam=0.1, settings=["XXXZ", "ZZZZ"],
            features=["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]),
    3: dict(_H2, window=2, protocol="RWP", lam=100.0, settings=["XXXZ", "ZZZZ"],
            features=["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"]),
    4: dict(graph="H0", alpha=0.75, dt=1.6, hx=0.580092560570867, hy=-0.694330375116467,
            J={(0, 1): -1.6178834565099725, (1, 2): -0.1998267640484581, (2, 3): -1.6305688408636787,
               (3, 4): -0.489627102594124, (4, 5): 0.5392332533499583}, r=4,
            window=117, protocol="CONT", lam=100.0, settings=["XXXXYX", "ZZZZZZ"],
            features=["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3", "YX45"]),
    5: dict(_H2, J={**_H2["J"], (0, 1): -0.0725568360781115}, window=64, protocol="CONT", lam=100.0,
            settings=["XXXX", "ZZZZ"], features=["X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3"]),
}


def _expand(c):
    edges = GRAPHS[c["graph"]]
    return dict(c, edges=edges, J=[c["J"][e] for e in edges], shots=2048)


CANDIDATES = {k: _expand(v) for k, v in RAW.items()}


def load_data(path=DATA):
    df = pd.read_csv(path, parse_dates=["input_date", "target_date"])
    return df.sort_values("target_date").reset_index(drop=True)


def training_scales(df, n_train=1095):
    """Mean and population SD of claims and exposure on 2022-2024 (used by the encoding)."""
    tr = df.iloc[:n_train]
    return dict(mean_c=tr.property_damage_claim_count_t.mean(), std_c=tr.property_damage_claim_count_t.std(ddof=0),
                mean_p=tr.active_policy_count_t.mean(), std_p=tr.active_policy_count_t.std(ddof=0))
