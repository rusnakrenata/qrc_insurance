"""Collect all IBM Kingston runs into one table.

  python analysis/build_run_table.py

Writes results/runs.csv and results/daily.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

APPROACHES = ["hardware-aware", "random", "qiskit"]

# logical graphs of C1-C5
EDGES = {
    "H0": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "H2": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "H3": [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5)],
}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def calibration_of_layout(props, layout, edges):
    """Worst-case and mean calibration figures of the six physical qubits."""
    q = {}
    for i, entries in enumerate(props["qubits"]):
        q[i] = {e["name"]: e["value"] for e in entries}
    cz = {}
    for g in props["gates"]:
        if g["gate"] == "cz":
            err = [p["value"] for p in g["parameters"] if p["name"] == "gate_error"]
            if err:
                cz[tuple(sorted(g["qubits"]))] = err[0]
    t1 = [q[p].get("T1", np.nan) for p in layout]
    t2 = [q[p].get("T2", np.nan) for p in layout]
    ro = [q[p].get("readout_error", np.nan) for p in layout]
    cze = [cz.get(tuple(sorted((layout[a], layout[b]))), np.nan) for a, b in edges]
    return dict(cal_min_t1_us=float(np.nanmin(t1)), cal_min_t2_us=float(np.nanmin(t2)),
                cal_mem_min_t2_us=float(np.nanmin(t2[4:])), cal_missing_entries=int(np.isnan(t1 + t2 + ro).sum()),
                cal_max_ro=float(np.nanmax(ro)), cal_mean_ro=float(np.nanmean(ro)),
                cal_max_cz=float(np.nanmax(cze)), cal_mean_cz=float(np.nanmean(cze)))


def qpu_usage(folder):
    """Charged QPU seconds and whether calibration changed between selection and execution."""
    files = sorted(glob.glob(os.path.join(folder, "chunk_*", "job_metrics.json")))
    charged, started, drift = 0, [], False
    for f in files:
        m = read_json(f)
        charged += m["usage"]["qpu_charge_time_seconds"]
        started.append(m["timestamps"]["running"])
        cal = os.path.join(os.path.dirname(f), "calibration_execution.json")
        if os.path.exists(cal):
            drift = drift or bool(read_json(cal).get("differs_from_selection"))
    return charged, min(started) if started else None, drift, len(files)


def run_rows(run_dir, rnd, props, proxy):
    meta = read_json(os.path.join(run_dir, "experiment.json"))
    ideal = pd.read_csv(os.path.join(run_dir, "ideal_predictions.csv"))
    rows, daily = [], []
    for approach in APPROACHES:
        folder = os.path.join(run_dir, approach)
        metrics_file = os.path.join(folder, "2026_metrics.json")
        if not os.path.exists(metrics_file):
            continue
        plan = read_json(os.path.join(folder, "plan.json"))
        cfg = plan["candidate_config"]
        m = read_json(metrics_file)
        d = pd.read_csv(os.path.join(folder, "2026_features_predictions.csv"))
        assert len(d) == 232 and (d.endpoint.values == ideal.endpoint.values).all()
        assert np.allclose(d.pred_ideal, ideal.prediction, atol=1e-8)
        y, pi, pq = d.actual.values, d.pred_ideal.values, d.pred_qpu.values
        e_q, e_i, delta = pq - y, pi - y, pq - pi
        assert abs(rmse(e_q) - m["qpu"]["rmse"]) < 1e-8

        # Aer with the calibration snapshot of the round
        noisy = {}
        for label in ["noisy_direct"] + [f"noisy_shots{plan['test_shots']}_rep{k}" for k in (1, 2, 3)]:
            p = os.path.join(folder, label + "_predictions.csv")
            if os.path.exists(p):
                n = pd.read_csv(p)
                noisy[label] = n.pred_noisy.values
        sampled = [v for k, v in noisy.items() if "shots" in k]

        charged, started, drift, n_jobs = qpu_usage(folder)
        layout = m["layout"]
        res = read_json(os.path.join(folder, "layout_pairing.json"))["resource"]
        row = dict(
            candidate=meta["candidate"], round=rnd, run_id=os.path.basename(run_dir),
            approach=approach, topology=cfg["topology"], window=cfg["window"], trotter_r=cfg["r"],
            layout=json.dumps(layout), shots=m["shots"], optimization_level=m["optimization_level"],
            n_dates=len(d), qpu_jobs=n_jobs, qpu_charged_s=charged, qpu_start_utc=started,
            calibration_changed_before_execution=drift,
            cz_per_feature_vector=res["feature_vector_n_cz"], max_depth=res["max_setting_depth"],
            max_duration_us=res["max_setting_duration_us"],
            # measured quantities
            qpu_rmse=rmse(e_q), qpu_mae=float(np.mean(np.abs(e_q))), qpu_bias=float(np.mean(e_q)),
            ideal_rmse=rmse(e_i), ideal_mae=float(np.mean(np.abs(e_i))), ideal_bias=float(np.mean(e_i)),
            # delta = hardware - ideal
            transfer_d=rmse(delta), level_shift=float(np.mean(delta)),
            level_share=float(np.mean(delta) ** 2 / np.mean(delta ** 2)),
            gap_median=float(np.median(np.abs(delta))), gap_p95=float(np.percentile(np.abs(delta), 95)),
            gap_max=float(np.max(np.abs(delta))), gap_frac_over3=float(np.mean(np.abs(delta) > 3)),
            corr_qpu_ideal=float(np.corrcoef(pq, pi)[0, 1]),
            qpu_crmse=float(np.std(e_q)), ideal_crmse=float(np.std(e_i)),
            cross_term=float(2 * np.mean(e_i * delta)),
            noisy_direct_rmse=rmse(noisy["noisy_direct"] - y) if "noisy_direct" in noisy else np.nan,
            noisy_direct_d=rmse(noisy["noisy_direct"] - pi) if "noisy_direct" in noisy else np.nan,
            noisy_direct_shift=float(np.mean(noisy["noisy_direct"] - pi)) if "noisy_direct" in noisy else np.nan,
            noisy_sampled_rmse=float(np.mean([rmse(v - y) for v in sampled])) if sampled else np.nan,
            noisy_sampled_rmse_sd=float(np.std([rmse(v - y) for v in sampled], ddof=1)) if len(sampled) > 1 else np.nan,
            noisy_sampled_d=float(np.mean([rmse(v - pi) for v in sampled])) if sampled else np.nan,
            source=os.path.relpath(folder, os.path.dirname(os.path.dirname(run_dir))),
        )
        row.update(calibration_of_layout(props, layout, EDGES[cfg["topology"]]))
        hit = proxy[proxy.layout == json.dumps(layout).replace(" ", "")] if proxy is not None else []
        row["error_proxy_nll"] = float(hit.actual_workload_proxy_nll.iloc[0]) if len(hit) else np.nan

        if approach == "hardware-aware":
            winner = read_json(os.path.join(folder, "probe_winner.json"))["layout"]
            probe = pd.read_csv(os.path.join(folder, "probe_distortion_summary.csv"))
            probe["key"] = probe.layout.apply(lambda s: json.dumps(json.loads(s)))
            pw = probe[probe.key == json.dumps(winner)].iloc[0]
            pl = probe[probe.key != json.dumps(winner)].iloc[0]
            sim = pd.read_csv(os.path.join(folder, "simulation_summary.csv"))
            sim["key"] = sim.layout.apply(lambda s: json.dumps(json.loads(s)))
            sw = sim[sim.key == json.dumps(winner)]
            row.update(probe_d=pw.qpu_pred_distortion_rmse_claims, probe_shift=pw.qpu_pred_distortion_bias_claims,
                       probe_d_other=pl.qpu_pred_distortion_rmse_claims,
                       probe_rel_gap=(pl.qpu_pred_distortion_rmse_claims - pw.qpu_pred_distortion_rmse_claims)
                       / pl.qpu_pred_distortion_rmse_claims,
                       aer_screen_d=float(sw.sim_pred_distortion_rmse_claims.iloc[0]) if len(sw) else np.nan,
                       aer_rank_of_winner=int(sw.sim_rank.iloc[0]) if len(sw) else np.nan,
                       ha_pool=len(sim), ha_prep_s=plan.get("preparation_seconds"))
            charged_probe = read_json(os.path.join(folder, "probe_job_metrics.json"))["usage"]["qpu_charge_time_seconds"]
            row["probe_charged_s"] = charged_probe
        elif approach == "random":
            sel = read_json(os.path.join(folder, "random_layout_selection.json"))
            row.update(random_seed=sel["seed"], native_pool=sel["native_embedding_count"])
        else:
            tr = read_json(os.path.join(folder, "transpiler_layout_trace.json"))
            row.update(qiskit_selector=tr["initial_selector"])
        rows.append(row)

        dd = pd.DataFrame(dict(candidate=meta["candidate"], round=rnd, approach=approach,
                               endpoint=d.endpoint, target_date=d.target_date, actual=y,
                               pred_ideal=pi, pred_qpu=pq, delta=delta))
        daily.append(dd)
    return rows, daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm-final", default=os.path.join(HERE, "..", "data", "ibm_kingston_runs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    rows, daily, skipped = [], [], []
    for cand_dir in sorted(glob.glob(os.path.join(a.ibm_final, "candidate*"))):
        rnd = 0
        for run_dir in sorted(glob.glob(os.path.join(cand_dir, "*/"))):
            run_dir = run_dir.rstrip("/")
            complete = all(os.path.exists(os.path.join(run_dir, x, "2026_metrics.json")) for x in APPROACHES)
            if not complete:
                skipped.append(dict(run=os.path.relpath(run_dir, a.ibm_final),
                                    reason="not all three approaches have final QPU metrics"))
                continue
            rnd += 1
            props = read_json(os.path.join(run_dir, "backend_properties.json"))
            px = os.path.join(run_dir, "hardware-aware", "proxy.csv")
            proxy = pd.read_csv(px) if os.path.exists(px) else None
            if proxy is not None:
                proxy["layout"] = proxy.layout.apply(lambda s: json.dumps(json.loads(s)).replace(" ", ""))
            r, dl = run_rows(run_dir, rnd, props, proxy)
            rows += r
            daily += dl
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "runs.csv"), index=False)
    pd.concat(daily).to_csv(os.path.join(a.out, "daily.csv"), index=False)
    pd.DataFrame(skipped).to_csv(os.path.join(a.out, "runs_excluded.csv"), index=False)
    print(f"{len(rows)} runs, {len(skipped)} excluded folders")


if __name__ == "__main__":
    main()
