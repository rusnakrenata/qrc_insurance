"""One hardware round: one candidate, one layout policy (ha, random or qiskit).

  python pipeline/13_hardware_round.py --candidate 2 --policy ha --round runs/c2_r1 --token <IBM token>
  python pipeline/13_hardware_round.py --candidate 2 --policy ha --round runs/test --dry-run --limit 6

Run the three policies on the same --round folder at the same time.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.circuits import features_from_counts, reservoir_circuit  # noqa: E402
from qrc.layouts import (coupling_graph, distortion, faulty_qubits, native_embeddings,  # noqa: E402
                         qiskit_layout, random_layout, strict_compile, workload_proxy)
from qrc.noise import load_noise_model, save_noise_model  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

TEST = np.arange(1460, 1692)
HA_POOL, HA_SIM_DATES, HA_SIM_SHOTS = 6, 20, 64          # Aer screening of the six best-proxy layouts
PROBE_DATES, PROBE_SHOTS = 100, 256                      # QPU probe of the two Aer finalists


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def get_backend(args):
    if args.dry_run:
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap
        cmap = CouplingMap.from_heavy_hex(3)
        return GenericBackendV2(num_qubits=cmap.size(), coupling_map=cmap, basis_gates=["cz", "rz", "sx", "x"],
                                seed=7), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=args.token)
    return service.backend(args.backend, use_fractional_gates=False), service


def run_counts(backend, circuits, shots, dry_run, noise_model=None, seed=1):
    """Return one counts dict per circuit, from the QPU (SamplerV2) or from Aer."""
    if dry_run or noise_model is not None:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed) if noise_model is not None \
            else AerSimulator.from_backend(backend, seed_simulator=seed)
        result = sim.run(circuits, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(circuits, shots=shots)
    print("  submitted job", job.job_id(), flush=True)
    return [r.data.c.get_counts() for r in job.result()]


def frozen_model(k):
    """Ideal features, readout and ideal predictions of candidate k (as in pipeline/12)."""
    c = CANDIDATES[k]
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    theta = encode(df, c["alpha"], **training_scales(df))
    A = step_operators(trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"]), theta)
    if c["protocol"] == "CONT":
        w, b = fit_ridge(cont_features(A[:1095], c["features"]), y[:1095], c["lam"])
    else:
        fit_idx = np.arange(c["window"] - 1, 1095)
        w, b = fit_ridge(rwp_features(A, fit_idx, c["window"], c["features"]), y[fit_idx], c["lam"])
    train = np.arange(c["window"] - 1, 1095)
    probe = train[np.linspace(0, len(train) - 1, PROBE_DATES, dtype=int)]
    sim = probe[np.rint(np.linspace(0, PROBE_DATES - 1, HA_SIM_DATES)).astype(int)]
    return dict(c=c, df=df, y=y, theta=theta, A=A, w=w, b=b, probe=probe, sim=sim)


def compiled(backend, m, endpoints, layout, opt_level):
    return [strict_compile(backend, reservoir_circuit(m["c"], m["theta"], int(e), s), layout, opt_level)
            for e in endpoints for s in m["c"]["settings"]]


def predict(m, counts, endpoints):
    """Features and forecasts from counts ordered as (date, setting)."""
    s = m["c"]["settings"]
    X = np.array([features_from_counts(m["c"], {s[0]: counts[2 * i], s[1]: counts[2 * i + 1]})
                  for i in range(len(endpoints))])
    return X, X @ m["w"] + m["b"]


def select_layout(policy, backend, m, args, noise_model, folder):
    c = m["c"]
    rep = int(m["sim"][10])                                       # representative training date
    probe_circuit = [reservoir_circuit(c, m["theta"], rep, s) for s in c["settings"]]

    def check(layout):
        return [strict_compile(backend, qc, layout, args.opt_level) for qc in probe_circuit]

    if policy == "qiskit":
        layout = qiskit_layout(backend, probe_circuit[0], args.opt_level)
        check(layout)
        return layout, dict(policy="qiskit", optimization_level=args.opt_level)

    if policy == "random":
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 32)
        layout, pool = random_layout(backend, c["edges"], check, seed)
        return layout, dict(policy="random", seed=seed, native_pool=pool)

    # hardware-aware: native embeddings -> workload proxy -> Aer screening -> QPU probe
    eligible = set(range(backend.num_qubits)) - faulty_qubits(backend)
    rows = []
    for layout in native_embeddings(c["edges"], coupling_graph(backend), eligible):
        try:
            rows.append(dict(layout=list(layout), proxy=sum(workload_proxy(q, backend) for q in check(layout))))
        except Exception:
            continue
    proxy = pd.DataFrame(rows).sort_values("proxy")
    proxy.to_csv(os.path.join(folder, "proxy.csv"), index=False)
    pool = proxy.layout.head(HA_POOL).tolist()
    ideal_sim = rwp_features(m["A"], m["sim"], c["window"], c["features"]) @ m["w"] + m["b"]
    screen = []
    for layout in pool:
        counts = run_counts(backend, compiled(backend, m, m["sim"], layout, args.opt_level), HA_SIM_SHOTS,
                            args.dry_run, noise_model)
        d, _ = distortion(predict(m, counts, m["sim"])[1], ideal_sim)
        screen.append(dict(layout=layout, aer_d=d))
    screen = pd.DataFrame(screen).sort_values("aer_d")
    screen.to_csv(os.path.join(folder, "aer_screening.csv"), index=False)
    finalists = screen.layout.head(2).tolist()
    ideal_probe = rwp_features(m["A"], m["probe"], c["window"], c["features"]) @ m["w"] + m["b"]
    probe = []
    for layout in finalists:
        counts = run_counts(backend, compiled(backend, m, m["probe"], layout, args.opt_level), PROBE_SHOTS,
                            args.dry_run)
        d, shift = distortion(predict(m, counts, m["probe"])[1], ideal_probe)
        probe.append(dict(layout=layout, probe_d=d, probe_shift=shift))
    probe = pd.DataFrame(probe).sort_values("probe_d")
    probe.to_csv(os.path.join(folder, "probe.csv"), index=False)
    return probe.layout.iloc[0], dict(policy="hardware-aware", pool=pool, finalists=finalists)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=int, choices=range(1, 6), required=True)
    ap.add_argument("--policy", choices=["ha", "random", "qiskit"], required=True)
    ap.add_argument("--round", required=True, help="folder shared by the three policies of one round")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--token", default=os.environ.get("IBM_TOKEN"))
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed", type=int, help="random-policy seed (default: fresh)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N test dates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from qiskit_aer.noise import NoiseModel
    os.makedirs(args.round, exist_ok=True)
    backend, _ = get_backend(args)
    m = frozen_model(args.candidate)
    test = TEST[: args.limit] if args.limit else TEST
    ideal_X = rwp_features(m["A"], test, m["c"]["window"], m["c"]["features"])
    ideal = ideal_X @ m["w"] + m["b"]

    # shared snapshot: the first policy to arrive stores it, the others reuse it
    snap = os.path.join(args.round, "noise_model.json")
    if not os.path.exists(snap):
        model = NoiseModel.from_backend(backend, gate_error=True, thermal_relaxation=True, readout_error=False)
        save_noise_model(model, snap)
        dump(os.path.join(args.round, "snapshot.json"),
             dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), backend=backend.name))
    noise_model = load_noise_model(snap)

    folder = os.path.join(args.round, {"ha": "hardware-aware", "random": "random", "qiskit": "qiskit"}[args.policy])
    os.makedirs(folder, exist_ok=True)
    layout, info = select_layout(args.policy if args.policy != "ha" else "ha", backend, m, args, noise_model, folder)
    dump(os.path.join(folder, "layout.json"), dict(layout=list(map(int, layout)), **info))
    print(f"C{args.candidate} {args.policy}: layout {list(layout)}", flush=True)

    circuits = compiled(backend, m, test, layout, args.opt_level)
    noisy = run_counts(backend, circuits, args.shots, True, noise_model)          # calibrated Aer, same layout
    hw = run_counts(backend, circuits, args.shots, args.dry_run)                  # processor
    X_aer, p_aer = predict(m, noisy, test)
    X_hw, p_hw = predict(m, hw, test)
    y = m["y"][test]
    daily = pd.DataFrame(dict(endpoint=test, target_date=m["df"].target_date.values[test], actual=y,
                              pred_ideal=ideal, pred_aer=p_aer, pred_qpu=p_hw, delta_pred=p_hw - ideal))
    for j, name in enumerate(m["c"]["features"]):
        daily[name + "_ideal"], daily[name + "_qpu"] = ideal_X[:, j], X_hw[:, j]
    daily.to_csv(os.path.join(folder, "2026_features_predictions.csv"), index=False)
    d, shift = distortion(p_hw, ideal)
    metrics = dict(layout=list(map(int, layout)), shots=args.shots, n=len(test),
                   rmse_qpu=float(np.sqrt(np.mean((p_hw - y) ** 2))), rmse_ideal=float(np.sqrt(np.mean((ideal - y) ** 2))),
                   rmse_aer=float(np.sqrt(np.mean((p_aer - y) ** 2))), transfer_d=d, level_shift=shift)
    dump(os.path.join(folder, "2026_metrics.json"), metrics)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
