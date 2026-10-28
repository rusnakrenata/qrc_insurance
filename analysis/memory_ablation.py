"""C4 memory ablation on IBM Kingston (Sec. VI-C).

Three arms on one HA layout, run in the order w1, scrambled, full:
  w1         a single step: encode day t, evolve, measure (about 5 us)
  scrambled  117 steps; steps 1-116 take the inputs of random training days, step 117 is day t
  full       117 steps with the true input history (the deployed circuit)
The frozen C4 readout is applied to every arm, and the 2026 RMSE is compared with each stored
full-window C4 run in data/ibm_kingston_runs/candidate4.

  python analysis/memory_ablation.py --compare-only data/ibm_kingston_ablation/candidate4/20261023T123117Z
  python analysis/memory_ablation.py --ideal-only
  python analysis/memory_ablation.py --dry-run --limit 4
  python analysis/memory_ablation.py --token <IBM token>
  python analysis/memory_ablation.py --resume <session folder> --token <IBM token>

One token for all arms (--token or IBM_TOKEN) or one per arm (--token-w1, --token-scrambled,
--token-full); tokens are never stored. On --resume, finished batches are kept and failed
jobs are submitted again.
"""
import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.readout import fit_ridge  # noqa: E402
from qrc.reservoir import cont_features, encode, rwp_features, step_operators, trotter_unitary  # noqa: E402

CAND, FULL_W, N_TRAIN = 4, 117, 1095
TEST = np.arange(1460, 1692)
ARMS = ("w1", "scrambled", "full")
REF = os.path.join(ROOT, "data", "ibm_kingston_runs", "candidate4")
OUT = os.path.join(ROOT, "data", "ibm_kingston_ablation", "candidate4")
ALPHA = 0.05


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# frozen C4 model and the inputs of each arm

def frozen_model():
    c = CANDIDATES[CAND]
    df = load_data()
    y = df.target_property_damage_claim_count.values.astype(float)
    theta = encode(df, c["alpha"], **training_scales(df))
    U = trotter_unitary(c["edges"], c["J"], c["hx"], c["hy"], c["dt"], c["r"])
    w, b = fit_ridge(cont_features(step_operators(U, theta[:N_TRAIN]), c["features"]), y[:N_TRAIN], c["lam"])
    return dict(c=c, df=df, y=y, theta=theta, U=U, w=w, b=b)


def angles(m, arm, ends, seed, keep_true=1):
    """Input angles of every step, shape (days, W, 4); the last step is always day t."""
    theta = m["theta"]
    if arm == "w1":
        return theta[ends][:, None, :].copy()
    seq = np.stack([theta[e - FULL_W + 1: e + 1] for e in ends])
    if arm == "scrambled":
        rng = np.random.default_rng(seed)
        seq[:, :FULL_W - keep_true] = theta[rng.integers(0, N_TRAIN, size=(len(ends), FULL_W - keep_true))]
    return seq


def ideal_features(m, seqs):
    names = m["c"]["features"]
    return np.array([rwp_features(step_operators(m["U"], s), [len(s) - 1], len(s), names)[0] for s in seqs])


# circuits and execution

def build_circuits(m, seqs):
    from qrc.circuits import reservoir_circuit
    return [reservoir_circuit(dict(m["c"], window=len(s)), s, len(s) - 1, setting)
            for s in seqs for setting in m["c"]["settings"]]


def compile_all(backend, circuits, layout, opt_level):
    from qiskit import transpile
    kw = dict(backend=backend, initial_layout=list(layout), routing_method="none",
              optimization_level=opt_level, seed_transpiler=42)
    try:
        isa = transpile(circuits, scheduling_method="alap", **kw)
    except Exception as exc:
        if "Duration" not in str(exc):
            raise
        isa = transpile(circuits, **kw)
    if any(qc.count_ops().get("swap", 0) for qc in isa):
        raise RuntimeError(f"SWAP needed for layout {layout}")
    return isa


def duration_us(qc, backend):
    try:
        return float(qc.estimate_duration(backend.target, unit="s")) * 1e6
    except Exception:
        return float("nan")


def resources(isa, backend):
    dur = [duration_us(qc, backend) for qc in isa]
    return dict(n_circuits=len(isa), max_duration_us=float(np.nanmax(dur)),
                cz_per_circuit=int(max(qc.count_ops().get("cz", 0) for qc in isa)),
                resets_per_circuit=int(max(qc.count_ops().get("reset", 0) for qc in isa)))


def same_as_full(scrambled, full, backend):
    """The scrambled arm must execute the complete 117-step circuit of the deployed model."""
    for a, f in zip(scrambled, full):
        oa, of = a.count_ops(), f.count_ops()
        if oa.get("cz", 0) != of.get("cz", 0) or oa.get("reset", 0) != of.get("reset", 0):
            raise RuntimeError("scrambled circuits differ from the full circuits; nothing submitted")
        da, df_ = duration_us(a, backend), duration_us(f, backend)
        if not (math.isnan(da) or abs(da - df_) <= 0.01 * df_):
            raise RuntimeError("scrambled circuits differ in duration from the full circuits; nothing submitted")


def predict(m, counts, n):
    from qrc.circuits import features_from_counts
    s = m["c"]["settings"]
    X = np.array([features_from_counts(m["c"], {s[0]: counts[2 * i], s[1]: counts[2 * i + 1]}) for i in range(n)])
    return X, X @ m["w"] + m["b"]


def batches(n, size):
    return [(b, i, min(i + size, n)) for b, i in enumerate(range(0, n, size))]


def counts_file(arm_dir, b):
    return os.path.join(arm_dir, f"qpu_counts_batch{b:03d}.json")


class Accounts:
    """IBM services built from the tokens; nothing about the accounts is written to disk."""

    def __init__(self, args):
        self.args, self.cache = args, {}
        self.tokens = {arm: getattr(args, "token_" + arm) or args.token or os.environ.get("IBM_TOKEN") for arm in ARMS}

    def service(self, token):
        if token not in self.cache:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.cache[token] = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token) if token
                                 else QiskitRuntimeService(channel="ibm_quantum_platform"))
        return self.cache[token]

    def backend(self, arm):
        return self.service(self.tokens[arm]).backend(self.args.backend, use_fractional_gates=False)

    def job(self, job_id):
        """Fetch a job with whichever of the given accounts submitted it."""
        for token in dict.fromkeys(self.tokens.values()):
            try:
                return self.service(token).job(job_id)
            except Exception:
                continue
        raise RuntimeError(f"job {job_id} not found with the given tokens; pass the token that submitted it")


def run_arm(args, accounts, arm, isa, arm_dir, backend):
    """Submit every batch without a live job, then collect; failed jobs are submitted again (3 attempts)."""
    plan = batches(len(isa) // 2, args.batch_dates)
    for _ in range(3):
        jobs, record = {}, load(os.path.join(arm_dir, "jobs.json"), {})
        for b, i, j in plan:
            if os.path.exists(counts_file(arm_dir, b)) or record.get(str(b), {}).get("job_id"):
                continue
            circuits = isa[2 * i: 2 * j]
            if args.dry_run:
                from qiskit_aer import AerSimulator
                res = AerSimulator.from_backend(backend, seed_simulator=b).run(circuits, shots=args.shots).result()
                dump(counts_file(arm_dir, b), [res.get_counts(k) for k in range(len(circuits))])
                continue
            from qiskit_ibm_runtime import SamplerV2
            job = SamplerV2(mode=backend).run(circuits, shots=args.shots)
            record[str(b)] = dict(job_id=job.job_id(), dates=[i, j], submitted_utc=now(),
                                  failed_attempts=record.get(str(b), {}).get("failed_attempts", []))
            dump(os.path.join(arm_dir, "jobs.json"), record)
            jobs[b] = job
            print(f"  {arm} batch {b}: submitted {job.job_id()}", flush=True)
        failed = []
        for b, i, j in plan:
            if os.path.exists(counts_file(arm_dir, b)):
                continue
            rec = record[str(b)]
            job = jobs.get(b) or accounts.job(rec["job_id"])
            try:
                result = job.result()
            except Exception as exc:
                rec["failed_attempts"] = rec.get("failed_attempts", []) + [dict(job_id=rec.pop("job_id"), error=str(exc)[:300])]
                failed.append(b)
                print(f"  {arm} batch {b}: failed ({str(exc)[:120]})", flush=True)
                continue
            dump(counts_file(arm_dir, b), [r.data.c.get_counts() for r in result])
            rec["finished_utc"] = now()
        dump(os.path.join(arm_dir, "jobs.json"), record)
        if not failed:
            return
    raise RuntimeError(f"{arm}: batches keep failing; see {arm}/jobs.json")


# outputs and comparison

def write_arm(m, arm, ends, X_ideal, folder, layout, X_qpu=None, extra=None):
    y = m["y"][ends]
    p_ideal = X_ideal @ m["w"] + m["b"]
    daily = pd.DataFrame(dict(endpoint=ends, target_date=m["df"].target_date.values[ends], actual=y, pred_ideal=p_ideal))
    met = dict(candidate=CAND, arm=arm, window=1 if arm == "w1" else FULL_W, n=len(ends), layout=layout,
               ideal=dict(rmse=rmse(p_ideal - y), bias=float(np.mean(p_ideal - y))))
    if X_qpu is not None:
        p_qpu = X_qpu @ m["w"] + m["b"]
        daily["pred_qpu"], daily["delta_pred"] = p_qpu, p_qpu - p_ideal
        met["qpu"] = dict(rmse=rmse(p_qpu - y), mae=float(np.mean(np.abs(p_qpu - y))), bias=float(np.mean(p_qpu - y)))
        met.update(transfer_d=rmse(p_qpu - p_ideal), level_shift=float(np.mean(p_qpu - p_ideal)))
    for j, name in enumerate(m["c"]["features"]):
        daily[name + "_ideal"] = X_ideal[:, j]
        if X_qpu is not None:
            daily[name + "_qpu"] = X_qpu[:, j]
    met.update(extra or {})
    daily.to_csv(os.path.join(folder, arm, "2026_features_predictions.csv"), index=False)
    dump(os.path.join(folder, arm, "2026_metrics.json"), met)
    return met


def t_sf(x, df):
    from scipy.stats import t
    return float(t.sf(x, df))


def paired(e_a, e_b, margin):
    """Diebold-Mariano test (Newey-West variance, HLN correction) and TOST for RMSE_a within RMSE_b +- margin."""
    d = e_a ** 2 - e_b ** 2
    if len(d) < 10:
        return dict(n=len(d), rmse_a=rmse(e_a), rmse_b=rmse(e_b), delta_rmse=rmse(e_a) - rmse(e_b),
                    dm_p=float("nan"), tost_p=float("nan"), margin=margin)
    n, lag = len(d), int(np.floor(4 * (len(d) / 100) ** (2 / 9)))
    dc = d - d.mean()
    var = dc @ dc / n + sum(2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    se = math.sqrt(max(var, 1e-300) / n) / math.sqrt((n - 1) / n)
    rb = rmse(e_b)
    lo, hi = max(rb - margin, 0) ** 2 - rb ** 2, (rb + margin) ** 2 - rb ** 2
    return dict(n=n, rmse_a=rmse(e_a), rmse_b=rb, delta_rmse=rmse(e_a) - rb, dm_p=2 * t_sf(abs(d.mean() / se), n - 1),
                tost_p=max(t_sf((d.mean() - lo) / se, n - 1), t_sf((hi - d.mean()) / se, n - 1)), margin=margin)


def verdict(t):
    if math.isnan(t["dm_p"]):
        return "too few days"
    side = "worse" if t["delta_rmse"] > 0 else "better"
    if t["tost_p"] < ALPHA:
        return f"equivalent within +-{t['margin']:.2f}"
    return f"different (ablation {side})" if t["dm_p"] < ALPHA else "no detectable difference"


def md(df):
    if df.empty:
        return "none\n"
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"


def reference_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(REF, "*", "*", "2026_features_predictions.csv"))):
        policy, stamp = f.split(os.sep)[-2], f.split(os.sep)[-3]
        d = pd.read_csv(f)
        if "pred_qpu" in d and len(d) == len(TEST):
            runs.append(dict(name=f"{stamp}/{policy}", policy=policy, daily=d))
    return runs


def compare(folder):
    arms = {a: pd.read_csv(os.path.join(folder, a, "2026_features_predictions.csv")) for a in ARMS
            if os.path.exists(os.path.join(folder, a, "2026_features_predictions.csv"))}
    refs = reference_runs()
    ha = [rmse(r["daily"].pred_qpu - r["daily"].actual) for r in refs if r["policy"] == "hardware-aware"]
    margin = max(ha) - min(ha)                       # spread of the repeated HA full-window runs
    refs_q = [(r["name"], r["daily"]) for r in refs]
    if "full" in arms and "pred_qpu" in arms["full"]:
        refs_q.insert(0, ("this session/full", arms["full"]))
    levels, rows = [], []
    for arm, d in arms.items():
        for col in ("pred_ideal", "pred_qpu"):
            if col in d:
                levels.append(dict(arm=arm, level=col[5:], n=len(d), rmse=rmse(d[col] - d.actual),
                                   bias=float(np.mean(d[col] - d.actual))))
        if arm == "full" or "pred_qpu" not in d:
            continue
        for name, ref in refs_q:
            j = d[["endpoint", "actual", "pred_qpu"]].merge(ref[["endpoint", "pred_qpu"]], on="endpoint", suffixes=("", "_ref"))
            t = paired((j.pred_qpu - j.actual).values, (j.pred_qpu_ref - j.actual).values, margin)
            rows.append(dict(arm=arm, reference=name, **t, verdict=verdict(t)))
    # noiseless model: each ablation against the ideal W = 117 forecast
    ideal = []
    if "full" in arms:
        for arm in ("w1", "scrambled"):
            if arm in arms:
                d, f = arms[arm], arms["full"]
                t = paired((d.pred_ideal - d.actual).values, (f.pred_ideal - f.actual).values, margin)
                ideal.append(dict(arm=arm, **t, verdict=verdict(t),
                                  max_abs_pred_diff=float(np.max(np.abs(d.pred_ideal - f.pred_ideal)))))
    # where each arm lies among the stored full-window runs
    ref_rmse = pd.DataFrame([dict(policy=r["policy"], rmse=rmse(r["daily"].pred_qpu - r["daily"].actual)) for r in refs])
    spread = [dict(arm=a, group=g, n_runs=len(s), arm_rmse=rmse(d.pred_qpu - d.actual), ref_min=s.rmse.min(),
                   ref_mean=s.rmse.mean(), ref_max=s.rmse.max())
              for a, d in arms.items() if "pred_qpu" in d
              for g, s in (("all", ref_rmse), ("hardware-aware", ref_rmse[ref_rmse.policy == "hardware-aware"]))]
    out = os.path.join(folder, "comparison")
    os.makedirs(out, exist_ok=True)
    L, Q, I, S = pd.DataFrame(levels), pd.DataFrame(rows), pd.DataFrame(ideal), pd.DataFrame(spread)
    L.to_csv(os.path.join(out, "rmse_by_arm.csv"), index=False)
    Q.to_csv(os.path.join(out, "qpu_vs_full_runs.csv"), index=False)
    I.to_csv(os.path.join(out, "ideal_vs_full.csv"), index=False)
    S.to_csv(os.path.join(out, "rmse_vs_full_run_spread.csv"), index=False)
    daily = None
    for arm, d in arms.items():
        keep = d[["endpoint", "target_date", "actual"] + [c for c in ("pred_ideal", "pred_qpu") if c in d]]
        keep = keep.rename(columns={c: f"{arm}_{c[5:]}" for c in keep.columns if c.startswith("pred_")})
        daily = keep if daily is None else daily.merge(keep, on=["endpoint", "target_date", "actual"])
    daily.to_csv(os.path.join(out, "daily_predictions.csv"), index=False)
    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write("# C4 memory ablation\n\n## 2026 RMSE (claims/day)\n\n" + md(L)
                + "\n## Noiseless model against the ideal W = 117 forecast\n\n" + md(I)
                + f"\n## Paired tests against the full-window runs (TOST margin {margin:.3f})\n\n" + md(Q)
                + "\n## Arm RMSE among the stored full-window runs\n\n" + md(S))
    print(L.round(3).to_string(index=False))
    return L, Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--layout", help="six physical qubits, default: the latest stored C4 HA layout")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--batch-dates", type=int, default=78)
    ap.add_argument("--keep-true", type=int, default=1, help="scrambled arm: final steps with true inputs")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--limit", type=int, help="first N test days only")
    ap.add_argument("--token")
    for arm in ARMS:
        ap.add_argument(f"--token-{arm}")
    ap.add_argument("--ideal-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="fake backend and Aer instead of the QPU")
    ap.add_argument("--out-root", default=OUT, help="where new sessions are written")
    ap.add_argument("--resume")
    ap.add_argument("--compare-only")
    args = ap.parse_args()
    if args.compare_only:
        compare(args.compare_only)
        return

    m = frozen_model()
    folder = args.resume
    if folder:
        exp = load(os.path.join(folder, "experiment.json"))
        for k in ("seed", "keep_true", "batch_dates", "shots", "limit", "backend"):
            if k in exp:
                setattr(args, k, exp[k])
        args.opt_level, args.layout = exp["optimization_level"], ",".join(map(str, exp["layout"]))
    ends = TEST[: args.limit] if args.limit else TEST
    seqs = {a: angles(m, a, ends, args.seed, args.keep_true) for a in ARMS}
    X_ideal = {a: ideal_features(m, seqs[a]) for a in ARMS}
    stored = pd.read_csv(sorted(glob.glob(os.path.join(REF, "*", "ideal_predictions.csv")))[-1]).set_index("endpoint")
    gap = float(np.max(np.abs(stored.loc[ends, "prediction"].values - (X_ideal["full"] @ m["w"] + m["b"]))))
    print(f"ideal W=117 against the stored ideal forecast: max |diff| = {gap:.1e}")
    if gap > 1e-6:
        raise RuntimeError("frozen C4 model does not reproduce the stored ideal forecast")

    if not folder:
        tag = "ideal" if args.ideal_only else "dryrun" if args.dry_run else "qpu"
        folder = os.path.join(args.out_root, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + tag)
    for a in ARMS:
        os.makedirs(os.path.join(folder, a), exist_ok=True)
        np.savez_compressed(os.path.join(folder, a, "input_angles.npz"), endpoints=ends, angles=seqs[a])
    if args.ideal_only:
        for a in ARMS:
            write_arm(m, a, ends, X_ideal[a], folder, None)
        compare(folder)
        return

    accounts = Accounts(args)
    if args.dry_run:
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap
        cmap = CouplingMap.from_heavy_hex(3)
        fake = GenericBackendV2(num_qubits=cmap.size(), coupling_map=cmap, basis_gates=["cz", "rz", "sx", "x"], seed=7)
        backends = {a: fake for a in ARMS}
    else:
        backends = {a: accounts.backend(a) for a in ARMS}
    if args.layout:
        layout = [int(q) for q in args.layout.split(",")]
    elif args.dry_run:
        from qrc.layouts import coupling_graph, native_embeddings
        layout = list(native_embeddings(m["c"]["edges"], coupling_graph(fake), set(range(fake.num_qubits)))[0])
    else:
        layout = load(sorted(glob.glob(os.path.join(REF, "*", "hardware-aware", "2026_metrics.json")))[-1])["layout"]
    exp = load(os.path.join(folder, "experiment.json")) or dict(
        created_utc=now(), candidate=CAND, arms=list(ARMS), backend=args.backend, layout=layout, shots=args.shots,
        optimization_level=args.opt_level, batch_dates=args.batch_dates, keep_true=args.keep_true, seed=args.seed,
        limit=args.limit, settings=m["c"]["settings"], features=m["c"]["features"])
    dump(os.path.join(folder, "experiment.json"), exp)

    todo = [a for a in ARMS if any(not os.path.exists(counts_file(os.path.join(folder, a), b))
                                   for b, _, _ in batches(len(ends), args.batch_dates))]
    isa, res = {}, load(os.path.join(folder, "resources.json"), {})
    for a in todo + (["full"] if "scrambled" in todo and "full" not in todo else []):
        isa[a] = compile_all(backends[a], build_circuits(m, seqs[a]), layout, args.opt_level)
        res[a] = resources(isa[a], backends[a])
        print(f"{a}: {res[a]}", flush=True)
    if "scrambled" in todo:
        same_as_full(isa["scrambled"], isa["full"], backends["scrambled"])
    dump(os.path.join(folder, "resources.json"), res)

    for a in ARMS:
        if a in todo:
            run_arm(args, accounts, a, isa[a], os.path.join(folder, a), backends[a])
            print(f"{a}: done {now()}", flush=True)
    for a in ARMS:
        counts = []
        for b, _, _ in batches(len(ends), args.batch_dates):
            counts += load(counts_file(os.path.join(folder, a), b))
        X_qpu, _ = predict(m, counts, len(ends))
        write_arm(m, a, ends, X_ideal[a], folder, layout, X_qpu, dict(resources=res.get(a)))
    exp["finished_utc"] = now()
    dump(os.path.join(folder, "experiment.json"), exp)
    compare(folder)
    print(f"results: {folder}")


if __name__ == "__main__":
    main()
