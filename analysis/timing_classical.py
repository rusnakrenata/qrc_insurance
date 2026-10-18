"""Wall-clock time to fit each selected classical model on 2022-2024 and forecast the 232 test days.

  python analysis/timing_classical.py

Single thread, median of three repetitions (one seed for ESN, RNN and GRU).
Writes results/timing/classical_timing.csv.
"""
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from classical import models as M  # noqa: E402
from qrc.candidates import load_data  # noqa: E402

spec = importlib.util.spec_from_file_location("p04", os.path.join(HERE, "..", "pipeline", "04_classical_2026.py"))
p04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p04)


def main():
    from threadpoolctl import threadpool_limits
    data = load_data()
    data.loc[p04.TEST[0]:, M.Y] = np.nan
    final = json.load(open(os.path.join(HERE, "..", "configs", "classical_final.json")))
    rows = []
    with threadpool_limits(limits=1):
        for method, cfg in final.items():
            seed = p04.seeds(method)[0]
            t = []
            for _ in range(3):
                t0 = time.perf_counter()
                p04.forecast(method, cfg, seed, data, p04.TEST)
                t.append(time.perf_counter() - t0)
            rows.append(dict(method=method, seconds_fit_and_forecast=float(np.median(t)),
                             seeds_in_article=len(p04.seeds(method))))
            print(rows[-1], flush=True)
    out = os.path.join(HERE, "..", "results", "timing")
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df["cpu"] = platform.processor() or platform.machine()
    df.to_csv(os.path.join(out, "classical_timing.csv"), index=False)


if __name__ == "__main__":
    main()
