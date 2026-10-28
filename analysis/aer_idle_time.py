"""Idle time in the calibrated Aer simulation of the HA runs (Sec. V, VI-B).

For the 21 HA runs: delay instructions in the simulated circuits (noisy_circuits.qpy), the
time each layout qubit spends in its own gates (calibrated gate lengths) against the circuit
duration, and the contraction eta of every observable in Aer and on the QPU. A round trip
through qrc/noise.py shows that the saved noise model loses the relaxation of idle periods.

Usage:  python analysis/aer_idle_time.py
Writes results/aer_idle/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "data", "ibm_kingston_runs")
OUT = os.path.join(ROOT, "results", "aer_idle")


def gate_lengths(props):
    gl = {}
    for g in props["gates"]:
        for x in g["parameters"]:
            if x["name"] == "gate_length":
                gl[(g["gate"], tuple(g["qubits"]))] = x["value"] * 1e-3      # ns -> us
    ro = {i: {e["name"]: e["value"] for e in es}.get("readout_length", 0) * 1e-3 for i, es in enumerate(props["qubits"])}
    return gl, ro


def busy_and_delays(circuit, gl, ro):
    busy, delays = {}, 0
    for inst in circuit.data:
        name = inst.operation.name
        qs = tuple(circuit.find_bit(b).index for b in inst.qubits)
        if name == "delay":
            delays += 1
            continue
        if name == "barrier":
            continue
        length = ro.get(qs[0], 0) if name == "measure" else (gl.get((name, qs)) or gl.get((name, qs[::-1])) or 0)
        for q in qs:
            busy[q] = busy.get(q, 0) + length
    return busy, delays


def eta(zi, zq):
    return float(np.polyfit(zi, zq, 1)[0]) if np.std(zi) > 1e-12 else np.nan


def analyse(source):
    from qiskit import qpy
    folder = os.path.join(RUNS, source)
    props = json.load(open(os.path.join(os.path.dirname(folder), "backend_properties.json")))
    met = json.load(open(os.path.join(folder, "2026_metrics.json")))
    L = [int(q) for q in met["layout"]]
    qd = {i: {e["name"]: e["value"] for e in es} for i, es in enumerate(props["qubits"])}
    t2 = np.array([qd[q]["T2"] for q in L])
    gl, ro = gate_lengths(props)
    with open(os.path.join(folder, "noisy_circuits.qpy"), "rb") as f:
        circuits = qpy.load(f)
    busy, delays = busy_and_delays(circuits[0], gl, ro)
    duration = float(json.load(open(os.path.join(folder, "layout_pairing.json")))["resource"]["max_setting_duration_us"])
    q = pd.read_csv(os.path.join(folder, "2026_features_predictions.csv"))
    a = pd.read_csv(os.path.join(folder, "noisy_direct_predictions.csv"))
    assert (q.endpoint.values == a.endpoint.values).all()
    names = [c[:-6] for c in q.columns if c.endswith("_ideal") and c != "pred_ideal"]
    obs = [dict(source=source, observable=n, eta_aer=eta(q[n + "_ideal"].values, a[n + "_noisy"].values),
                eta_qpu=eta(q[n + "_ideal"].values, q[n + "_qpu"].values)) for n in names]
    ob = pd.DataFrame(obs)
    inj = ob[ob.observable != "YX45"]
    mem_busy = float(np.mean([busy.get(p, 0) for p in L[4:]]))
    row = dict(source=source, layout=json.dumps(L), duration_us=duration, delays_in_aer_circuit=delays,
               T2_mem_median=float(np.median(t2[4:])), mem_busy_us=mem_busy, mem_busy_fraction=mem_busy / duration,
               exposure_real_T2=duration / float(np.median(t2[4:])), exposure_aer_T2=mem_busy / float(np.median(t2[4:])),
               eta_YX45_aer=float(ob.loc[ob.observable == "YX45", "eta_aer"].iloc[0]) if "YX45" in names else np.nan,
               eta_YX45_qpu=float(ob.loc[ob.observable == "YX45", "eta_qpu"].iloc[0]) if "YX45" in names else np.nan,
               eta_inj_aer_mean=float(inj.eta_aer.mean()), eta_inj_qpu_mean=float(inj.eta_qpu.mean()),
               d_aer=float(np.sqrt(np.mean((a.pred_noisy - q.pred_ideal) ** 2))),
               d_qpu=float(np.sqrt(np.mean((q.pred_qpu - q.pred_ideal) ** 2))))
    return row, ob


def noise_roundtrip():
    """Delay-relaxation passes of an Aer noise model before and after qrc/noise.py save/load."""
    import tempfile
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_aer.noise import NoiseModel
    from qrc.noise import load_noise_model, save_noise_model
    nm = NoiseModel.from_backend(GenericBackendV2(num_qubits=5, seed=1), gate_error=True, thermal_relaxation=True,
                                 readout_error=False)
    path = os.path.join(tempfile.mkdtemp(), "noise.json")
    save_noise_model(nm, path)
    return dict(delay_relaxation_passes_before=len(getattr(nm, "_custom_noise_passes", [])),
                delay_relaxation_passes_after_save_load=len(getattr(load_noise_model(path), "_custom_noise_passes", [])))


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = pd.read_csv(os.path.join(ROOT, "results", "runs.csv"))
    runs = runs[runs.approach == "hardware-aware"]
    rows, obs = [], []
    for _, r in runs.iterrows():
        row, ob = analyse(r.source)
        rows.append(dict(candidate=int(r.candidate), round=int(r["round"]), **row))
        obs.append(ob.assign(candidate=int(r.candidate), round=int(r["round"])))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "per_run.csv"), index=False)
    pd.concat(obs).to_csv(os.path.join(OUT, "per_observable.csv"), index=False)
    cols = ["duration_us", "T2_mem_median", "mem_busy_fraction", "exposure_real_T2", "exposure_aer_T2", "eta_YX45_aer",
            "eta_YX45_qpu", "eta_inj_aer_mean", "eta_inj_qpu_mean", "d_aer", "d_qpu"]
    summ = R.groupby("candidate")[cols].median()
    summ.insert(0, "runs", R.groupby("candidate").size())
    summ.to_csv(os.path.join(OUT, "summary_by_candidate.csv"))
    rt = noise_roundtrip()
    with open(os.path.join(OUT, "noise_model_roundtrip.json"), "w") as f:
        json.dump(rt, f, indent=1)
    pd.set_option("display.width", 220)
    print("HA runs:", len(R), "| runs with delays in the Aer circuits:", int((R.delays_in_aer_circuit > 0).sum()))
    print("qrc/noise.py round trip:", rt)
    print(summ.round(3).to_string())
    print(f"results: {OUT}")


if __name__ == "__main__":
    main()
