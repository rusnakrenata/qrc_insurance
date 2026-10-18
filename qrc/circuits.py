"""Qiskit circuits for the reservoir and conversion of measured counts into features.

One circuit per forecast date and measurement setting: W reset-encode-evolve
steps followed by a basis change and a full measurement. Logical qubit q is
circuit qubit q; the physical placement is chosen later at transpilation.
"""
import numpy as np
from qiskit import QuantumCircuit


def evolve(qc, c):
    """r first-order Trotter slices of H = sum J ZZ + hx sum X + hy (Y4+Y5)."""
    tau = c["dt"] / c["r"]
    for _ in range(c["r"]):
        for (i, j), J in zip(c["edges"], c["J"]):
            qc.rzz(2 * J * tau, i, j)          # RZZ(phi) = exp(-i phi/2 ZZ)
        for q in range(6):
            qc.rx(2 * c["hx"] * tau, q)
        for q in (4, 5):
            qc.ry(2 * c["hy"] * tau, q)


def reservoir_circuit(c, theta, endpoint, setting=None):
    """Reset-window replay ending at `endpoint`; measured in `setting` if given."""
    W = c["window"]
    qc = QuantumCircuit(6, 6 if setting else 0)
    for k, t in enumerate(range(endpoint - W + 1, endpoint + 1)):
        if k > 0:
            for q in range(4):
                qc.reset(q)                    # partial trace over the injection register
        for q in range(4):
            qc.ry(theta[t, q], q)
        evolve(qc, c)
    if setting:
        setting = setting.ljust(6, "Z")
        for q, b in enumerate(setting):
            if b == "X":
                qc.h(q)
            elif b == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure(range(6), range(6))
    return qc


def expectation_from_counts(counts, qubits):
    """<P_q1 P_q2 ...> from bitstring counts (Qiskit order: qubit 0 is the last character)."""
    total = sum(counts.values())
    value = 0
    for bits, n in counts.items():
        sign = 1
        for q in qubits:
            if bits[::-1][q] == "1":
                sign = -sign
        value += sign * n
    return value / total


def features_from_counts(c, counts_by_setting):
    """Feature vector in the order of c['features'] from the two measurement settings."""
    x_setting, z_setting = c["settings"]
    out = []
    for name in c["features"]:
        if name == "YX45":
            out.append(expectation_from_counts(counts_by_setting[x_setting], [4, 5]))
        elif name[0] == "Z":
            out.append(expectation_from_counts(counts_by_setting[z_setting], [int(name[1])]))
        else:
            out.append(expectation_from_counts(counts_by_setting[x_setting], [int(name[1])]))
    return np.array(out)
