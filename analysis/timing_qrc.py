"""Time of the ideal QRC (readout fit and test forecasts) and QPU time of every run.

  python analysis/timing_qrc.py

Writes results/timing/qrc_timing.csv and results/timing/qpu_time_by_run.csv.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "timing")
RUNS = os.path.join(HERE, "..", "data", "ibm_kingston_runs")
TEST = np.arange(1460, 1692)


def ideal(c, df, y, scales):
    theta = encode(df, c["alpha"], **scales)
    A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
    if c["protocol"] == "CONT":
        Xfit, yfit = cont_features(A[:1095], c["features"]), y[:1095]
    else:
        idx = np.arange(c["window"] - 1, 1095)
        Xfit, yfit = rwp_features(A, idx, c["window"], c["features"]), y[idx]
    w, b = fit_ridge(Xfit, yfit, c["lam"])
    return rwp_features(A, TEST, c["window"], c["features"]) @ w + b


def main():
    from threadpoolctl import threadpool_limits
    os.makedirs(OUT, exist_ok=True)
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    scales = training_scales(df)
    rows = []
    with threadpool_limits(limits=1):
        for k, c in CANDIDATES.items():
            t = []
            for _ in range(3):
                t0 = time.perf_counter()
                ideal(c, df, y, scales)
                t.append(time.perf_counter() - t0)
            rows.append(dict(candidate=k, window=c["window"], ideal_seconds=float(np.median(t))))
            print(rows[-1], flush=True)
    jobs = []
    for f in sorted(glob.glob(os.path.join(RUNS, "candidate*", "*", "*", "chunk_*", "job_metrics.json"))):
        parts = os.path.relpath(f, RUNS).split(os.sep)
        j = json.load(open(f))
        ts = j["timestamps"]
        jobs.append(dict(candidate=int(parts[0][-1]), run_id=parts[1], approach=parts[2],
                         qpu_charged_s=j["usage"]["qpu_charge_time_seconds"],
                         circuit_execution_s=j.get("circuits_execution_time_ns", np.nan) / 1e9,
                         running_to_finished_s=(pd.Timestamp(ts["finished"]) - pd.Timestamp(ts["running"])).total_seconds()))
    jobs = pd.DataFrame(jobs).groupby(["candidate", "run_id", "approach"]).sum().reset_index()
    runs = pd.read_csv(os.path.join(HERE, "..", "results", "runs.csv"))[["candidate", "run_id", "approach", "round",
                                                                          "ha_prep_s", "probe_charged_s"]]
    jobs = jobs.merge(runs, on=["candidate", "run_id", "approach"], how="inner")
    jobs.to_csv(os.path.join(OUT, "qpu_time_by_run.csv"), index=False)
    s = jobs.groupby("candidate")[["qpu_charged_s", "circuit_execution_s", "running_to_finished_s"]].median()
    out = pd.DataFrame(rows).set_index("candidate").join(s.add_suffix("_median"))
    out["qpu_s_per_forecast"] = out.qpu_charged_s_median / len(TEST)
    out.reset_index().to_csv(os.path.join(OUT, "qrc_timing.csv"), index=False)
    print(out.round(3).to_string())
    print(len(jobs), "runs;", jobs[jobs.approach == "hardware-aware"][["ha_prep_s", "probe_charged_s"]].median().to_dict())


if __name__ == "__main__":
    main()
