"""Qiskit optimization levels 1-3 on a saved Kingston calibration (no QPU jobs).

  python analysis/optimization_levels.py
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
from qrc.candidates import CANDIDATES, load_data, training_scales  # noqa: E402
from qrc.circuits import reservoir_circuit  # noqa: E402
from qrc.reservoir import encode  # noqa: E402


def saved_target(run):
    from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
    conf = BackendConfiguration.from_dict(json.loads((run / "backend_configuration.json").read_text()))
    props = BackendProperties.from_dict(json.loads((run / "backend_properties.json").read_text()))
    return convert_to_target(conf, props, include_fractional_gates=False)


def main():
    from qiskit import transpile
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm-final", default=os.path.join(HERE, "..", "data", "ibm_kingston_runs"))
    args = ap.parse_args()
    df = load_data()
    rows = []
    for k, c in CANDIDATES.items():
        runs = sorted(Path(args.ibm_final, f"candidate{k}").glob("*"))
        run = next(r for r in runs if all((r / p / "2026_metrics.json").exists()
                                          for p in ("hardware-aware", "random", "qiskit")))
        target = saved_target(run)
        theta = encode(df, c["alpha"], **training_scales(df))
        train = np.arange(c["window"] - 1, 1095)            # representative training date of pipeline/13
        probe = train[np.linspace(0, len(train) - 1, 100, dtype=int)]
        endpoint = int(probe[np.rint(np.linspace(0, 99, 20)).astype(int)][10])
        for level in (1, 2, 3):
            for s in c["settings"]:
                passes = []

                def trace(**kw):                           # remember which layout pass found a solution
                    p = kw["property_set"]
                    passes.append((kw["pass_"].name(), str(p.get("VF2Layout_stop_reason"))))
                isa = transpile(reservoir_circuit(c, theta, endpoint, s), target=target, optimization_level=level,
                                seed_transpiler=42, scheduling_method="alap", callback=trace)
                names = [n for n, _ in passes]
                selector = ("TrivialLayout" if "TrivialLayout" in names and not any(
                    n == "VF2Layout" and "SOLUTION_FOUND" in r for n, r in passes) else
                    "VF2Layout" if any(n == "VF2Layout" and "SOLUTION_FOUND" in r for n, r in passes) else
                    "SabreLayout" if "SabreLayout" in names else "other")
                rows.append(dict(candidate=k, level=level, setting=s, selector=selector,
                                 qubits=sorted(isa.layout.initial_index_layout(filter_ancillas=True)[:6]),
                                 cz=isa.count_ops().get("cz", 0),
                                 duration_us=round(isa.estimate_duration(target, unit="s") * 1e6, 3),
                                 snapshot=run.name))
                print(rows[-1], flush=True)
    out = os.path.join(HERE, "..", "results", "tables", "optimization_levels.csv")
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
