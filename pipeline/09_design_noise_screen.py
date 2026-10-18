"""Design stage 5: calibrated-noise screening of the 33 rule representatives and the Pareto front.

  python pipeline/09_design_noise_screen.py
  python pipeline/09_design_noise_screen.py --simulate RWP_H3_R4 --calibration <run folder>|fake [--days 20]

Writes results/design/noise_*.csv.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import GRAPHS, load_data, training_scales  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "design")
OUT = os.path.join(HERE, "..", "results", "design")


def pareto(df, cols):
    """True for points not dominated in all `cols` (minimized)."""
    v = df[cols].to_numpy(float)
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        dom = np.all(v <= v[i], axis=1) & np.any(v < v[i], axis=1)
        keep[i] = not dom.any()
    return keep


def analyse():
    s = pd.read_csv(os.path.join(DATA, "noise_sampled_summary.csv"))
    s["pareto_recomputed"] = pareto(s, ["rmse_mean", "feature_vector_n_cz", "shots_per_setting"])
    assert (s.pareto_recomputed == s.pareto).all()
    front = s[s.pareto_recomputed].sort_values("rmse_mean")
    front.to_csv(os.path.join(OUT, "noise_pareto.csv"), index=False)
    print(f"{len(s)} operating points of {s.candidate_key.nunique()} reservoirs; Pareto front:")
    print(front[["candidate_key", "r", "shots_per_setting", "rmse_mean", "feature_vector_n_cz"]].to_string(index=False))

    d = pd.read_csv(os.path.join(DATA, "noise_direct_best.csv"))
    d["effect"] = np.where(d.direct_rmse < d.ideal_rmse, "improved", "degraded")
    d.to_csv(os.path.join(OUT, "noise_candidates.csv"), index=False)
    print(d.effect.value_counts().to_string())
    h4 = s[s.topology == "H4"].sort_values("rmse_mean").iloc[0]
    dom = s[(s.rmse_mean <= h4.rmse_mean) & (s.feature_vector_n_cz <= h4.feature_vector_n_cz)
            & (s.shots_per_setting <= h4.shots_per_setting) & (s.candidate_key != h4.candidate_key)]
    print(f"best H4 point {h4.candidate_key} r={h4.r} N={h4.shots_per_setting}: RMSE {h4.rmse_mean:.3f}, "
          f"{h4.feature_vector_n_cz:.0f} CZ; dominated by {dom.candidate_key.tolist()}")


def noisy_backend(calibration):
    if calibration == "fake":
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap.from_heavy_hex(3)
        return GenericBackendV2(cm.size(), coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], seed=7)
    from qiskit.providers import BackendV2, Options
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    run = Path(calibration)
    target = convert_to_target(BackendConfiguration.from_dict(json.loads((run / "backend_configuration.json").read_text())),
                               BackendProperties.from_dict(json.loads((run / "backend_properties.json").read_text())),
                               include_fractional_gates=False)

    class Snapshot(BackendV2):
        def __init__(self):
            super().__init__(name="calibration_snapshot")

        @property
        def target(self):
            return target

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options()

        def run(self, *a, **k):
            raise NotImplementedError
    return Snapshot()


def simulate(key, calibration, days):
    """Direct noisy expectations of one RWP representative on 2025 (first `days` days), best native layout."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qrc.circuits import reservoir_circuit
    from qrc.layouts import coupling_graph, native_embeddings, workload_proxy
    from qrc.design import Reservoir
    from qrc.reservoir import OBS, encode
    rep = pd.read_csv(os.path.join(DATA, "rule_representatives.csv")).set_index("candidate_key").loc[key]
    if rep.protocol != "RWP":
        raise SystemExit("--simulate supports replay representatives; continuous ones need chronological noisy evolution")
    J = json.loads(rep.J_json)
    edges = GRAPHS[rep.topology]
    c = dict(edges=edges, J=[J[f"J{min(e)}{max(e)}"] for e in edges], hx=rep.hx, hy=rep.hy, dt=rep["dt"],
             r=int(rep.original_r), window=int(rep.window))
    df = load_data().iloc[:1460].reset_index(drop=True)
    theta = encode(df, rep.alpha, **training_scales(df))
    backend = noisy_backend(calibration)
    noise = NoiseModel.from_backend(backend, gate_error=True, thermal_relaxation=True, readout_error=False)
    adj = coupling_graph(backend)
    layouts = native_embeddings(edges, adj, set(adj))[:200]
    probe_qc = reservoir_circuit(c, theta, 1100)
    layout = min(layouts, key=lambda l: workload_proxy(transpile(probe_qc, backend, initial_layout=list(l),
                                                                 routing_method="none", optimization_level=1), backend))
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    from qrc.readout import fit_ridge
    names = [n.replace("YX_45", "YX45") for n in (["X0", "X1", "X2", "Z0", "Z1", "Z2", "Z3"] +
                                                   (["YX45"] if "YX45" in rep.test_readout else []))]
    if "dropX3" not in rep.test_readout:
        names.insert(3, "X3")
    res = Reservoir(2, edges, c["J"], c["hx"], c["hy"], c["dt"], c["r"])
    S, F = res.maps(theta, names)
    ends = np.arange(c["window"] - 1, 1460)
    X = res.rwp(S, F, ends, c["window"])
    y = df.target_property_damage_claim_count.values.astype(float)
    tr = ends < 1095
    w, b = fit_ridge(X[tr], y[ends[tr]], rep.source_selected_lambda)
    rows = []
    for e in range(1095, 1095 + days):
        qc = reservoir_circuit(c, theta, e)
        isa = transpile(qc, backend, initial_layout=list(layout), routing_method="none", optimization_level=1)
        phys = [isa.layout.initial_layout[qc.qubits[q]] for q in range(6)]
        isa.save_density_matrix(qubits=[isa.qubits[p] for p in phys])   # idle qubits are truncated by Aer
        rho6 = np.asarray(sim.run(isa).result().data()["density_matrix"])
        # Qiskit orders qubit 0 last; OBS uses qubit 0 first
        perm = np.arange(64).reshape([2] * 6).transpose(list(range(5, -1, -1))).reshape(-1)
        rho6 = rho6[np.ix_(perm, perm)]
        z = np.array([np.trace(rho6 @ OBS[n]).real for n in names])
        rows.append(dict(endpoint=e, pred_noisy=z @ w + b, pred_ideal=X[ends == e][0] @ w + b, actual=y[e]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f"noise_simulation_{key}.csv"), index=False)
    print(out.describe().round(3).to_string(), "\nlayout", layout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate")
    ap.add_argument("--calibration", default="fake")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.simulate:
        simulate(args.simulate, args.calibration, args.days)
    else:
        analyse()


if __name__ == "__main__":
    main()
