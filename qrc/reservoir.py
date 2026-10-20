"""Six-qubit quantum reservoir: encoding, Trotterised dynamics, RWP features.

Qubits 0-3 are injection qubits (claims C_t, target weekday D, exposure P_t,
holiday H), qubits 4-5 form the memory register. Qubit 0 is the leftmost
tensor factor. Exact density-matrix simulation (no sampling, no noise).
"""
from functools import reduce

import numpy as np

I2 = np.eye(2, dtype=complex)
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
N, N_INJ, N_MEM = 6, 4, 2
DIM_I, DIM_M = 2 ** N_INJ, 2 ** N_MEM


def op_on(paulis):
    """Tensor product with the given {qubit: 'X'|'Y'|'Z'} and identity elsewhere."""
    return reduce(np.kron, [PAULI[paulis[q]] if q in paulis else I2 for q in range(N)])


def encode(df, alpha, mean_c, std_c, mean_p, std_p):
    """Rotation angles theta_t (one row per input day, four injection qubits)."""
    zc = (df["property_damage_claim_count_t"].values - mean_c) / std_c
    zp = (df["active_policy_count_t"].values - mean_p) / std_p
    theta = np.zeros((len(df), 4))
    theta[:, 0] = alpha * np.clip(zc, -3, 3) / 3          # claims, clipped at +-3 SD
    theta[:, 1] = 2 * np.pi * df["target_date"].dt.dayofweek.values / 7  # weekday of t+1
    theta[:, 2] = alpha * zp / (3 + np.abs(zp))           # exposure, soft-saturated
    theta[:, 3] = np.pi * df["is_public_holiday_t_plus_1"].values  # holiday of t+1
    return theta


def trotter_unitary(edges, J, hx, hy, dt, r):
    """First-order product formula of H = sum J ZZ + hx sum X + hy (Y4 + Y5)."""
    U = np.eye(2 ** N, dtype=complex)
    step = dt / r
    terms = [(op_on({i: "Z", j: "Z"}), J[k]) for k, (i, j) in enumerate(edges)]
    terms += [(op_on({q: "X"}), hx) for q in range(N)]
    terms += [(op_on({q: "Y"}), hy) for q in (4, 5)]
    for _ in range(r):
        for P, c in terms:
            U = (np.cos(c * step) * np.eye(2 ** N) - 1j * np.sin(c * step) * P) @ U
    return U


def step_operators(U, theta):
    """Kraus-like maps A_t[a] acting on the memory register for every input row.
    Change in memory register: rho_m -> sum_a A_t[a] rho_m A_t[a]^dagger, where a = 0..15 is the injection register basis."""
    U4 = U.reshape(DIM_I, DIM_M, DIM_I, DIM_M)
    out = np.empty((len(theta), DIM_I, DIM_M, DIM_M), dtype=complex)
    for t, row in enumerate(theta):
        # State after R_Y rotations on the injection qubits, then apply U and trace out the injection register
        phi = reduce(np.kron, [np.array([np.cos(a / 2), np.sin(a / 2)]) for a in row])
        out[t] = np.einsum("ambn,b->amn", U4, phi)
    return out


OBS = {f"{p}{q}": op_on({q: p}) for p in "XYZ" for q in range(N)}
OBS["YX45"] = op_on({4: "Y", 5: "X"})


def rwp_features(A, endpoints, window, names):
    """Reset-window replay: memory starts in |00>, the last `window` inputs are replayed."""
    rows = []
    for e in endpoints:
        rho_m = np.zeros((DIM_M, DIM_M), dtype=complex)
        rho_m[0, 0] = 1 # initial memory state |00><00|
        for t in range(e - window + 1, e + 1):
            # full state after evolution: |psi_a> branches of the injection register
            # partial trace for rho_m = sum_a <psi_a|rho|psi_a>
            rho = np.einsum("amn,nk,bpk->ambp", A[t], rho_m, A[t].conj()).reshape(64, 64)
            rho_m = np.einsum("amn,nk,apk->mp", A[t], rho_m, A[t].conj())
        rows.append([np.trace(rho @ OBS[n]).real for n in names])
    return np.array(rows)


def cont_features(A, names):
    """Continuous operation: memory is carried through the whole chronology."""
    rho_m = np.zeros((DIM_M, DIM_M), dtype=complex)
    rho_m[0, 0] = 1 # initial memory state |00><00|
    rows = []
    for t in range(len(A)):
        rho = np.einsum("amn,nk,bpk->ambp", A[t], rho_m, A[t].conj()).reshape(64, 64)
        rho_m = np.einsum("amn,nk,apk->mp", A[t], rho_m, A[t].conj())
        rows.append([np.trace(rho @ OBS[n]).real for n in names])
    return np.array(rows)
