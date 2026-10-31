"""Run-level timing of the 63 QPU evaluations (layout, Pre, QPU, Post, probe, calibration change).

  python analysis/run_timing.py

Pre and Post come from results/timing/pre_post_by_policy.csv (analysis/timing_pre_post.py).
Writes results/tables/run_timing.csv and results/tables/tab_run_timing.tex.
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUNS = os.path.join(ROOT, "data", "ibm_kingston_runs")
POL = {"hardware-aware": "HA", "random": "Ra", "qiskit": "Qi"}


def ts(s):
    return pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")


PRE = ("authentication", "backend_loading", "chunk_compilation", "chunk_submission")


def stage_seconds(folder):
    """Pre- and post-processing seconds of the evaluation segments (those with chunk_* stages) of all timing logs."""
    segs = []
    for f in sorted(glob.glob(os.path.join(folder, "timings_*.json"))):
        for ev in json.load(open(f))["events"]:
            if ev.get("stage") == "execute_total":
                segs.append([])
            elif segs and "elapsed_seconds" in ev:
                segs[-1].append((ev["stage"], ev["elapsed_seconds"]))
    main = [s for s in segs if any(k.startswith("chunk_") for k, _ in s)]
    pre = sum(v for s in main for k, v in s if k in PRE)
    post = sum(v for s in main for k, v in s if k == "chunk_scoring")
    return pre, post


def main():
    runs = pd.read_csv(os.path.join(ROOT, "results", "runs.csv"))
    clean = pd.read_csv(os.path.join(ROOT, "results", "timing", "pre_post_by_policy.csv"))
    clean["policy"] = clean.policy.map({"ha": "HA", "random": "Ra", "qiskit": "Qi"})
    clean = clean.set_index(["candidate", "policy"])
    rows = []
    for _, r in runs.iterrows():
        folder = os.path.join(RUNS, r.source)
        jobs = []
        for ch in sorted(glob.glob(os.path.join(folder, "chunk_*"))):
            m = json.load(open(os.path.join(ch, "job_metrics.json")))
            c = json.load(open(os.path.join(ch, "calibration_execution.json")))
            jobs.append(dict(submitted=ts(m["timestamps"]["created"]),                 # job created at IBM
                             running=ts(m["timestamps"]["running"]), finished=ts(m["timestamps"]["finished"]),
                             cal_selection=ts(c["selection"]), cal_execution=ts(c["execution"]["last_update_date_utc"]),
                             drift=bool(c["differs_from_selection"])))
        j = pd.DataFrame(jobs)
        pre, post = stage_seconds(folder)
        probe = os.path.join(folder, "probe_job_metrics.json")
        probe_s = json.load(open(probe))["usage"]["qpu_charge_time_seconds"] if os.path.exists(probe) else None
        probe_n = len(pd.read_csv(os.path.join(folder, "probe_resources.csv"))) if probe_s is not None else None
        probe_shots = json.load(open(os.path.join(folder, "plan.json")))["probe_shots"] if probe_s is not None else None
        rows.append(dict(candidate=r.candidate, round=r["round"], policy=POL[r.approach], run_id=r.run_id,
                         layout=r.layout, n_jobs=len(j),
                         submitted_utc=j.submitted.min(), started_utc=j.running.min(), finished_utc=j.finished.max(),
                         calibration_selection_utc=j.cal_selection.iloc[0],
                         calibration_execution_utc=j.cal_execution.max(),
                         calibration_changed=j.drift.any(),
                         hours_snapshot_to_start=(j.running.min() - j.cal_selection.iloc[0]).total_seconds() / 3600,
                         hours_submit_to_start=(j.running.min() - j.submitted.min()).total_seconds() / 3600,
                         preprocessing_s=clean.loc[(r.candidate, POL[r.approach]), "pre_s"],
                         qpu_charged_s=r.qpu_charged_s,
                         postprocessing_s=clean.loc[(r.candidate, POL[r.approach]), "post_s"],
                         logged_pre_s=pre, logged_post_s=post,
                         probe_qpu_s=probe_s, probe_circuits=probe_n, probe_shots_per_circuit=probe_shots,
                         calibration_changed_runs_csv=bool(r.calibration_changed_before_execution)))
    t = pd.DataFrame(rows).sort_values(["candidate", "round", "policy"],
                                       key=lambda s: s.map({"HA": 0, "Ra": 1, "Qi": 2}) if s.name == "policy" else s)
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    t.to_csv(os.path.join(ROOT, "results", "tables", "run_timing.csv"), index=False)

    assert (t.calibration_changed == t.calibration_changed_runs_csv).all()
    lines = [f"C{r.candidate} & {r['round']} & {r.policy} & [{','.join(map(str, json.loads(r.layout)))}] & "
             f"{r.preprocessing_s:.0f} & {r.qpu_charged_s:.0f} & {r.postprocessing_s:.2f} & "
             f"{'--' if pd.isna(r.probe_qpu_s) else f'{r.probe_qpu_s:.0f}'} & "
             f"{'Y' if r.calibration_changed else 'N'} \\\\" for _, r in t.iterrows()]
    open(os.path.join(ROOT, "results", "tables", "tab_run_timing.tex"), "w").write("\n".join(lines) + "\n")
    print(t.groupby("policy")[["preprocessing_s", "qpu_charged_s", "postprocessing_s"]].median().round(1).to_string())
    print(t[["hours_snapshot_to_start", "hours_submit_to_start"]].describe().round(2).to_string())
    print("calibration changed before execution:", int(t.calibration_changed.sum()), "of", len(t))
    print(t.groupby(["candidate", "round"]).started_utc.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
          .describe().round(2).rename("start spread within round (h)").to_string())


if __name__ == "__main__":
    main()
