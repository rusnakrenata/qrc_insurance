"""Probe of one reservoir on IQM Emerald (two layouts, 100 training dates, 512 shots).

  python pipeline/11_iqm_probe.py
  python pipeline/11_iqm_probe.py --submit --batch 1 --url <IQM url> --token <token>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from qrc.candidates import load_data, training_scales  # noqa: E402
from qrc.circuits import reservoir_circuit  # noqa: E402
from qrc.reservoir import encode  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
LAYOUTS = {"L1": ["QB42", "QB34", "QB32", "QB40", "QB41", "QB33"],
           "L2": ["QB47", "QB48", "QB34", "QB33", "QB41", "QB42"]}
SETTINGS, SHOTS, PER_JOB = ["XXXXYX", "ZZZZZZ"], 512, 25


def candidate():
    ident = json.load(open(os.path.join(DATA, "iqm", "iqm_candidate.json")))
    keys = ["01", "15", "52", "23", "34", "40", "45"]
    return dict(edges=[(int(k[0]), int(k[1])) for k in keys], J=[ident["J"][k] for k in keys],
                hx=ident["hx"], hy=ident["hy"], dt=ident["dt"], r=ident["r"], window=ident["window"],
                alpha=ident["alpha"])


def probe_dates(window):
    train = np.arange(window - 1, 1095)
    return train[np.linspace(0, len(train) - 1, 100, dtype=int)]


def circuits(c, theta, dates):
    """(date, layout, setting, circuit), layouts alternating so drift affects both equally."""
    out = []
    for i, e in enumerate(dates):
        order = list(LAYOUTS) if i % 2 == 0 else list(LAYOUTS)[::-1]
        for lay in order:
            for s in SETTINGS:
                out.append((int(e), lay, s, reservoir_circuit(c, theta, int(e), s)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--batch", type=int, choices=[1, 2, 3, 4], default=1)
    ap.add_argument("--url")
    ap.add_argument("--token", default=os.environ.get("IQM_TOKEN"))
    ap.add_argument("--computer", default="emerald")
    args = ap.parse_args()
    c = candidate()
    df = load_data()
    theta = encode(df, c["alpha"], **training_scales(df))
    dates = probe_dates(c["window"])
    batch = dates[(args.batch - 1) * PER_JOB: args.batch * PER_JOB]
    circs = circuits(c, theta, batch)
    archived = pd.read_csv(os.path.join(DATA, "iqm", "iqm_probe_counts.csv"))
    if args.batch == 1:
        assert sorted(set(archived.endpoint)) == sorted(map(int, batch)), "probe dates differ from the archive"
    print(f"batch {args.batch}: {len(circs)} circuits, dates {batch[0]}..{batch[-1]}")
    if not args.submit:
        return
    from qiskit import transpile
    from iqm.qiskit_iqm import IQMProvider
    backend = IQMProvider(args.url, quantum_computer=args.computer, token=args.token).get_backend()
    isa = []
    for e, lay, s, qc in circs:
        idx = [backend.qubit_name_to_index(q) for q in LAYOUTS[lay]]
        t = transpile(qc, backend, initial_layout=idx, routing_method="none", optimization_level=1, seed_transpiler=42)
        assert "swap" not in t.count_ops()
        isa.append(t)
    job = backend.run(isa, shots=SHOTS)
    counts = job.result().get_counts()
    rows = [dict(endpoint=e, layout_id=lay, layout=json.dumps(LAYOUTS[lay]), setting=s, shots_returned=SHOTS,
                 counts_json=json.dumps(cnt), job_id=job.job_id()) for (e, lay, s, _), cnt in zip(circs, counts)]
    out = os.path.join(HERE, "..", "results", f"iqm_probe_batch_{args.batch}_raw_counts.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
