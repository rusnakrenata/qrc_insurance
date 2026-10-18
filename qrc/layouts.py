"""Native embeddings, strict compilation and the three layout policies."""
import math
import random

import numpy as np
from qiskit import transpile


def coupling_graph(backend):
    """Undirected CZ coupling graph of the backend as an adjacency dict."""
    adj = {q: set() for q in range(backend.num_qubits)}
    for i, j in backend.target.build_coupling_map(two_q_gate="cz").get_edges():
        adj[i].add(j)
        adj[j].add(i)
    return adj


def faulty_qubits(backend):
    try:
        return set(backend.properties().faulty_qubits())
    except Exception:
        return set()


def native_embeddings(edges, adj, eligible):
    """All injective maps logical->physical that send every logical edge to a physical coupler."""
    nbr = {q: set() for q in range(6)}
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    order = sorted(range(6), key=lambda q: -len(nbr[q]))      # place high-degree vertices first
    eligible = set(eligible)
    found, assign = [], {}

    def place(k):
        if k == 6:
            found.append(tuple(assign[q] for q in range(6)))
            return
        lq = order[k]
        placed = [n for n in nbr[lq] if n in assign]
        if placed:
            options = set.intersection(*(adj[assign[n]] for n in placed)) & eligible
        else:
            options = {p for p in eligible if len(adj[p] & eligible) >= len(nbr[lq])}
        for p in sorted(options - set(assign.values())):
            assign[lq] = p
            place(k + 1)
            del assign[lq]

    place(0)
    return found


def strict_compile(backend, circuit, layout, opt_level=1, seed=42):
    """Transpile on a fixed layout with routing disabled; fail if a SWAP would be needed."""
    try:
        isa = transpile(circuit, backend=backend, initial_layout=list(layout), routing_method="none",
                        optimization_level=opt_level, seed_transpiler=seed, scheduling_method="alap")
    except Exception as exc:
        if "Duration" not in str(exc):         # simulated test backends lack reset durations
            raise
        isa = transpile(circuit, backend=backend, initial_layout=list(layout), routing_method="none",
                        optimization_level=opt_level, seed_transpiler=seed)
    if isa.count_ops().get("swap", 0):
        raise RuntimeError(f"SWAP needed for layout {layout}")
    return isa


def workload_proxy(isa, backend):
    """L = -sum log(1-e_gate) - sum log(1-e_readout) over the compiled circuit, eq. (7)."""
    target = backend.target
    nll = 0.0
    for inst in isa.data:
        name = inst.operation.name
        qubits = tuple(isa.find_bit(q).index for q in inst.qubits)
        if name in ("barrier", "delay"):
            continue
        props_ = target[name].get(qubits) if name in target else None   # measure error = readout error
        e = props_.error if props_ is not None else None
        if e is not None:
            nll += -math.log1p(-min(max(e, 0.0), 1 - 1e-15))
    return nll


def random_layout(backend, edges, compile_check, seed):
    """Shuffle native embeddings that avoid faulty qubits; return the first that compiles."""
    eligible = set(range(backend.num_qubits)) - faulty_qubits(backend)
    pool = sorted(native_embeddings(edges, coupling_graph(backend), eligible))
    random.Random(seed).shuffle(pool)
    for layout in pool:
        try:
            compile_check(layout)
            return list(layout), len(pool)
        except Exception:
            continue
    raise RuntimeError("no native embedding compiles")


def qiskit_layout(backend, circuit, opt_level=1, seed=42):
    """Initial layout chosen by the Qiskit preset pass manager (VF2Layout or trivial)."""
    isa = transpile(circuit, backend=backend, optimization_level=opt_level, seed_transpiler=seed)
    return list(isa.layout.initial_index_layout(filter_ancillas=True))[:6]


def distortion(pred_hw, pred_ideal):
    """Transfer distortion d and level shift of a set of predictions."""
    delta = np.asarray(pred_hw) - np.asarray(pred_ideal)
    return float(np.sqrt(np.mean(delta ** 2))), float(np.mean(delta))
