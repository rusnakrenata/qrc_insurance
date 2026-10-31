"""Classical pre- and post-processing time of one hardware evaluation per candidate and layout policy.

  python analysis/timing_pre_post.py
  python analysis/timing_pre_post.py --candidates 3 --policies qiskit
  python analysis/timing_pre_post.py --quick --candidates 3

Writes results/timing/pre_post_by_policy.csv.
"""
import argparse
import gc
import glob
import importlib.util
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.circuits import features_from_counts, reservoir_circuit  # noqa: E402
from qrc.layouts import coupling_graph, distortion, native_embeddings, qiskit_layout, strict_compile, workload_proxy  # noqa: E402
from qrc.reservoir import encode, rwp_features  # noqa: E402

spec = importlib.util.spec_from_file_location("hw", os.path.join(ROOT, "pipeline", "12_hardware_round.py"))
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)
OUT = os.path.join(ROOT, "results", "timing", "pre_post_by_policy.csv")
EXCLUDED = ("20261007T110222", "20261008T163051")          # rounds without complete QPU results


def snapshot_backend(run):
    """Backend built from the stored calibration of a round (target and properties)."""
    from qiskit.providers import BackendV2, Options
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    conf = BackendConfiguration.from_dict(json.load(open(os.path.join(run, "backend_configuration.json"))))
    props = BackendProperties.from_dict(json.load(open(os.path.join(run, "backend_properties.json"))))
    target = convert_to_target(conf, props, include_fractional_gates=False)

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

        def properties(self):
            return props

        def run(self, *a, **k):
            raise NotImplementedError
    return Snapshot()


# worker processes
W = {}


def init_worker(run, k):
    W.update(backend=snapshot_backend(run), c=CANDIDATES[k])
    df = load_data()
    W["theta"] = encode(df, W["c"]["alpha"], **training_scales(df))


def compile_dates(args):
    """Build and transpile the circuits of some dates on one layout; returns the number compiled."""
    dates, layout = args
    return len([strict_compile(W["backend"], reservoir_circuit(W["c"], W["theta"], int(e), s), layout, 1)
                for e in dates for s in W["c"]["settings"]])


def proxy_of(args):
    """Workload proxy of several layouts (None if a layout does not compile)."""
    layouts, rep = args
    qcs = [reservoir_circuit(W["c"], W["theta"], rep, s) for s in W["c"]["settings"]]
    out = []
    for layout in layouts:
        try:
            out.append((sum(workload_proxy(strict_compile(W["backend"], q, layout, 1), W["backend"]) for q in qcs),
                        list(layout)))
        except Exception:
            out.append(None)
    return out


def compile_circuits(args):
    """Build and transpile the circuits of some dates on one layout and return them."""
    dates, layout = args
    return [strict_compile(W["backend"], reservoir_circuit(W["c"], W["theta"], int(e), s), layout, 1)
            for e in dates for s in W["c"]["settings"]]


def aer_stage(args):
    """Aer screening in a separate process."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    run, k, compiled, n_dates, w, b, ideal, threads = args
    backend, c = snapshot_backend(run), CANDIDATES[k]
    t0 = time.perf_counter()
    noise = NoiseModel.from_backend(backend, gate_error=True, thermal_relaxation=True, readout_error=False)
    sim = AerSimulator(noise_model=noise, seed_simulator=1, max_parallel_threads=threads)
    screen, batches, s_ = [], [], c["settings"]
    for lay, qcs in compiled:
        counts, batch = aer_counts(sim, qcs, hw.HA_SIM_SHOTS)
        batches.append(batch)
        X = np.array([features_from_counts(c, {s_[0]: counts[2 * i], s_[1]: counts[2 * i + 1]}) for i in range(n_dates)])
        screen.append((distortion(X @ w + b, ideal)[0], lay))
    return screen, min(batches), time.perf_counter() - t0


# main process
def first_round(k):
    runs = [r for r in sorted(glob.glob(os.path.join(ROOT, "data", "ibm_kingston_runs", f"candidate{k}", "*")))
            if os.path.isdir(os.path.join(r, "hardware-aware")) and not any(x in r for x in EXCLUDED)]
    return runs[0]


def split(seq, n):
    return [list(x) for x in np.array_split(np.asarray(seq, dtype=object), n) if len(x)]


def start_pool(jobs, run, k, attempts=3):
    """Worker pool with every worker initialised; returns the pool and its start-up time."""
    from concurrent.futures.process import BrokenProcessPool
    gc.collect()
    t0 = time.perf_counter()
    for attempt in range(attempts):
        pool = ProcessPoolExecutor(jobs, initializer=init_worker, initargs=(run, k))
        try:
            list(pool.map(int, range(jobs)))
            return pool, time.perf_counter() - t0
        except BrokenProcessPool:
            pool.shutdown(cancel_futures=True)
            if attempt == attempts - 1:
                raise
            print(f"worker start failed, retrying ({attempt + 1})", flush=True)
            gc.collect()
            time.sleep(10)


def aer_counts(sim, qcs, shots):
    """One Aer call per layout as in pipeline/12; split the call only if memory runs out."""
    batch = len(qcs)
    while True:
        try:
            counts = []
            for i in range(0, len(qcs), batch):
                res = sim.run(qcs[i:i + batch], shots=shots).result()
                counts += [res.get_counts(j) for j in range(len(qcs[i:i + batch]))]
            return counts, batch
        except MemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)


def synthetic_counts(rng, n, settings, shots):
    return [{s: {format(j, "06b"): int(v) for j, v in enumerate(rng.multinomial(shots, rng.dirichlet(np.ones(64)))) if v}
             for s in settings} for _ in range(n)]


def select(policy, pools, backend, m, run, jobs, quick):
    """Classical layout selection; `pools` holds the worker pool and is renewed around the Aer step."""
    c, t = m["c"], {}
    rep = int(m["sim"][10])
    probe_circuit = [reservoir_circuit(c, m["theta"], rep, s) for s in c["settings"]]

    def check(layout):
        return [strict_compile(backend, qc, layout, 1) for qc in probe_circuit]

    t0 = time.perf_counter()
    if policy == "qiskit":
        layout = qiskit_layout(backend, probe_circuit[0], 1)
        check(layout)
        t["selection_s"] = time.perf_counter() - t0
        return layout, t
    eligible = set(range(backend.num_qubits)) - set(backend.properties().faulty_qubits())
    emb = sorted(native_embeddings(c["edges"], coupling_graph(backend), eligible))
    if policy == "random":
        seed = json.load(open(os.path.join(run, "random", "random_layout_selection.json")))["seed"]
        random.Random(seed).shuffle(emb)
        for layout in emb:
            try:
                check(layout)
                break
            except Exception:
                continue
        t["selection_s"] = time.perf_counter() - t0
        return list(layout), t
    # HA: proxy, Aer screening of six layouts, probe compilation
    t1 = time.perf_counter()
    emb = emb[:12] if quick else emb
    rows = [r for part in pools["pool"].map(proxy_of, [(x, rep) for x in split(emb, jobs * 4)]) for r in part if r]
    rows.sort(key=lambda r: r[0])
    t["proxy_s"], t["proxy_layouts"] = time.perf_counter() - t1, len(rows)
    t1 = time.perf_counter()
    sim_dates = m["sim"][:4] if quick else m["sim"]
    ideal_sim = rwp_features(m["A"], sim_dates, c["window"], c["features"]) @ m["w"] + m["b"]
    pool_layouts = [r[1] for r in rows[:hw.HA_POOL]]
    compiled = [[q for part in pools["pool"].map(compile_circuits, [(d, lay) for d in split(sim_dates, jobs)]) for q in part]
                for lay in pool_layouts]
    t["aer_compile_s"] = time.perf_counter() - t1
    pools["pool"].shutdown()                                   # free the workers' memory for the simulation
    with ProcessPoolExecutor(1) as aer:                        # its memory is released when it ends
        screen, t["aer_batch"], t["aer_simulation_s"] = aer.submit(
            aer_stage, (run, m["k"], list(zip(pool_layouts, compiled)), len(sim_dates), m["w"], m["b"], ideal_sim,
                        jobs)).result()
    del compiled
    screen.sort(key=lambda r: r[0])
    t["aer_screening_s"] = t["aer_compile_s"] + t["aer_simulation_s"]
    pools["pool"], t["pool_restart_s"] = start_pool(jobs, run, m["k"])
    t1 = time.perf_counter()
    dates = m["probe"][:10] if quick else m["probe"]
    list(pools["pool"].map(compile_dates, [(d, layout) for _, layout in screen[:2] for d in split(dates, jobs)]))
    rng = np.random.default_rng(1)
    for _ in screen[:2]:
        X = np.array([features_from_counts(c, cs) for cs in synthetic_counts(rng, len(dates), c["settings"], hw.PROBE_SHOTS)])
        _ = X @ m["w"] + m["b"]
    t["probe_classical_s"] = time.perf_counter() - t1
    t["selection_s"] = t["proxy_s"] + t["aer_screening_s"] + t["probe_classical_s"]    # without process start-up
    return json.load(open(os.path.join(run, "hardware-aware", "layouts.json")))["hardware-aware"], t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="1,2,3,4,5")
    ap.add_argument("--policies", default="qiskit,random,ha")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    out = OUT.replace(".csv", "_quick.csv") if args.quick else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = pd.read_csv(out) if os.path.exists(out) and not args.redo else pd.DataFrame(columns=["candidate", "policy"])
    machine = (f"{platform.processor() or platform.machine()} | {os.cpu_count()} logical CPUs | {args.jobs} workers | "
               f"{platform.system()} {platform.release()} | Python {platform.python_version()}")
    for k in map(int, args.candidates.split(",")):
        todo = [p for p in args.policies.split(",") if not ((done.candidate == k) & (done.policy == p)).any()]
        if not todo:
            continue
        run = first_round(k)
        m = dict(hw.frozen_model(k), k=k)
        pools = {}
        pools["pool"], t_pool = start_pool(args.jobs, run, k)
        try:
            for policy in todo:
                t0 = time.perf_counter()
                backend = snapshot_backend(run)
                t_backend = time.perf_counter() - t0
                layout, t = select(policy, pools, backend, m, run, args.jobs, args.quick)
                test = hw.TEST[:10] if args.quick else hw.TEST
                t0 = time.perf_counter()
                list(pools["pool"].map(compile_dates, [(d, list(map(int, layout))) for d in split(test, args.jobs * 2)]))
                t_eval = time.perf_counter() - t0
                counts = synthetic_counts(np.random.default_rng(0), len(test), m["c"]["settings"], 2048)
                t0 = time.perf_counter()
                X = np.array([features_from_counts(m["c"], cs) for cs in counts])
                _ = X @ m["w"] + m["b"]
                t_post = time.perf_counter() - t0
                row = dict(candidate=k, policy=policy, round=os.path.basename(run),
                           layout=json.dumps(list(map(int, layout))), backend_s=t_backend, pool_start_s=t_pool,
                           evaluation_transpile_s=t_eval, **t, pre_s=t_backend + t["selection_s"] + t_eval,
                           post_s=t_post, machine=machine)
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(out, index=False)
                print({a: (round(v, 2) if isinstance(v, float) else v) for a, v in row.items() if a != "machine"},
                      flush=True)
        finally:
            pools["pool"].shutdown()


if __name__ == "__main__":
    main()
