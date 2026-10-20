"""Memory capacity, washout, CV and a generic reservoir with any number of memory qubits."""
from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .reservoir import DIM_M, cont_features

PROBE_SEED, K_MAX, MC_ALPHA = 79001, 20, 1e-6
ONE_BODY = [f"{p}{q}" for p in "XYZ" for q in range(6)]


def probe(theta):
    """Permute every input sequence = channel independently: same marginals, no temporal structure."""
    rng = np.random.default_rng(PROBE_SEED)
    out = np.empty_like(theta)
    for ch in range(theta.shape[1]):
        out[:, ch] = theta[rng.permutation(len(theta)), ch]
    return out


def _ket(v):
    v = np.asarray(v, dtype=complex)
    return np.outer(v, v.conj())


INITIAL_STATES = [_ket([1, 0, 0, 0]), _ket([0, 0, 0, 1]), _ket([.5, .5, .5, .5]),
                  _ket(np.array([1, 0, 0, 1]) / np.sqrt(2)), np.eye(4, dtype=complex) / 4]


def washout(A, threshold=0.01):
    """Steps until all initial memory states agree to `threshold` in trace distance (None if never)."""
    states = list(INITIAL_STATES)
    curve = []
    for At in A:
        states = [np.einsum("amn,nk,apk->mp", At, r, At.conj()) for r in states]
        curve.append(max(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum() for a, b in combinations(states, 2)))
    tail = np.maximum.accumulate(np.asarray(curve)[::-1])[::-1]
    idx = np.flatnonzero(tail <= threshold)
    return int(idx[0]) if len(idx) else None


def _zridge(Xtr, ytr, Xte, alpha):
    """Ridge on standardized features (constant columns dropped)."""
    m, s = Xtr.mean(0), Xtr.std(0)
    keep = s > 1e-12
    if not keep.any():
        return np.full(len(Xte), ytr.mean())
    return Ridge(alpha=alpha).fit((Xtr[:, keep] - m[keep]) / s[keep], ytr).predict((Xte[:, keep] - m[keep]) / s[keep])


def memory_capacity(A_probe, probe_theta, n_train=1095, burnin=100, X=None):
    """Corrected MC per input channel (see module docstring). X: precomputed one-body features."""
    X = cont_features(A_probe, ONE_BODY) if X is None else X
    n_val = len(probe_theta) - n_train
    val = np.arange(n_train, n_train + n_val)
    total = 0.0
    for ch in range(probe_theta.shape[1]):
        for k in range(1, K_MAX + 1):
            tr = np.arange(max(burnin, k), n_train)
            p = _zridge(X[tr], probe_theta[tr - k, ch], X[val], MC_ALPHA)
            y = probe_theta[val - k, ch]
            r2 = np.corrcoef(y, p)[0, 1] ** 2 if np.std(p) > 0 and np.std(y) > 0 else 0.0
            total += max(r2 - 1.0 / (n_val - 1), 0.0)
    return total / probe_theta.shape[1]


def mc_and_washout(A_probe, probe_theta, n_train=1095):
    """MC with a burn-in of max(100, T_w + 20) steps, as in the design search."""
    tw = washout(A_probe)
    burnin = max(100, tw + 20) if tw is not None else n_train - 100
    return memory_capacity(A_probe, probe_theta, n_train, min(burnin, n_train - 100)), tw


def cv_rmse(X, y, alpha, folds=5):
    """Mean RMSE of a standardized ridge readout over expanding chronological folds."""
    errs = [np.sqrt(np.mean((_zridge(X[tr], y[tr], X[va], alpha) - y[va]) ** 2))
            for tr, va in TimeSeriesSplit(folds).split(X)]
    return float(np.mean(errs)), float(np.std(errs, ddof=1))


from functools import reduce as _reduce  # noqa: E402

_P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


class Reservoir:
    """4 injection qubits + n_mem memory qubits; H = sum J ZZ + hx sum X + hy sum_mem Y, r Trotter slices.

    Precomputes, per input day, the memory channel S_t (vec(rho) -> vec(rho')) and the linear
    functionals F_t that give observable expectations from the memory state before the step."""

    def __init__(self, n_mem, edges, J, hx, hy=0.0, dt=1.6, r=2):
        self.n_mem, self.n = n_mem, 4 + n_mem
        self.DI, self.DM = 16, 2 ** n_mem
        U = np.eye(2 ** self.n, dtype=complex)
        terms = [(self.op({i: "Z", j: "Z"}), c) for (i, j), c in zip(edges, J)]
        terms += [(self.op({q: "X"}), hx) for q in range(self.n)]
        terms += [(self.op({q: "Y"}), hy) for q in range(4, self.n)] if hy else []
        for _ in range(r):
            for P, c in terms:
                U = (np.cos(c * dt / r) * np.eye(len(U)) - 1j * np.sin(c * dt / r) * P) @ U
        self.U4 = U.reshape(self.DI, self.DM, self.DI, self.DM)

    def op(self, paulis):
        return _reduce(np.kron, [_P[paulis.get(q, "I")] for q in range(self.n)])

    def maps(self, theta, names):
        """S (T, DM^2, DM^2) and F (T, n_features, DM^2) for every input row.
        S_t: state of the memory register and F_t: expectation value of observables.
        both are just fucntion and are used to simplify the calculations (we do not need to perfrom matrix operations)"""
        obs = []
        for name in names:                                  # 'X3', 'Z5' or two-body like 'YX45'
            ps, qs = (name[:1], name[1:]) if len(name) == 2 else (name[:2], name[2:])
            obs.append(self.op({int(q): p for p, q in zip(ps, qs)}).reshape(self.DI, self.DM, self.DI, self.DM))
        S = np.empty((len(theta), self.DM ** 2, self.DM ** 2), dtype=complex)
        F = np.empty((len(theta), len(names), self.DM ** 2), dtype=complex)
        for t, row in enumerate(theta):
            phi = _reduce(np.kron, [np.array([np.cos(a / 2), np.sin(a / 2)]) for a in row])
            A = np.einsum("ambn,b->amn", self.U4, phi)
            S[t] = np.einsum("amn,apk->mpnk", A, A.conj()).reshape(self.DM ** 2, -1)
            for j, O in enumerate(obs):
                F[t, j] = np.einsum("bpam,amn,bpk->nk", O, A, A.conj()).reshape(-1)
        return S, F

    def rho0(self):
        v = np.zeros(self.DM ** 2, dtype=complex)
        v[0] = 1
        return v

    def rwp(self, S, F, endpoints, window):
        out = []
        for e in endpoints:
            v = self.rho0()
            for t in range(e - window + 1, e):
                v = S[t] @ v
            out.append((F[e] @ v).real)
        return np.array(out)

    def cont(self, S, F):
        v, out = self.rho0(), []
        for t in range(len(S)):
            out.append((F[t] @ v).real)
            v = S[t] @ v
        return np.array(out)


LAMBDAS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5]


def select_lambda(X, y, lambdas=LAMBDAS):
    """Readout penalty by mean CV RMSE (ties: smaller lambda). Returns (lambda, cv mean, cv sd)."""
    best = None
    for lam in lambdas:
        m, s = cv_rmse(X, y, lam)
        if best is None or m < best[1] - 1e-15:
            best = (lam, m, s)
    return best


def washout_S(S, threshold=0.01):
    """Same as washout(), from the memory channels S_t of a Reservoir."""
    vs = [r.reshape(-1) for r in INITIAL_STATES]
    curve = []
    dm = int(np.sqrt(S.shape[1]))
    for St in S:
        vs = [St @ v for v in vs]
        rs = [v.reshape(dm, dm) for v in vs]
        curve.append(max(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum() for a, b in combinations(rs, 2)))
    tail = np.maximum.accumulate(np.asarray(curve)[::-1])[::-1]
    idx = np.flatnonzero(tail <= threshold)
    return int(idx[0]) if len(idx) else None


def mc_fast(res, probe_theta, n_train=1095):
    """Memory capacity and washout of a Reservoir (6 qubits), using its precomputed maps."""
    S, F = res.maps(probe_theta, ONE_BODY)
    tw = washout_S(S)
    burnin = min(max(100, tw + 20) if tw is not None else n_train - 100, n_train - 100)
    return memory_capacity(None, probe_theta, n_train, burnin, X=res.cont(S, F)), tw


def process_fidelity(edges, J, hx, hy, dt, r):
    """|Tr(U_exact^dagger U_trotter)/d|^2 for the Hamiltonian of the reservoir."""
    from .reservoir import op_on, trotter_unitary
    H = sum(c * op_on({i: "Z", j: "Z"}) for (i, j), c in zip(edges, J))
    H = H + hx * sum(op_on({q: "X"}) for q in range(6)) + hy * (op_on({4: "Y"}) + op_on({5: "Y"}))
    w, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * w * dt)) @ V.conj().T
    return float(abs(np.trace(U.conj().T @ trotter_unitary(edges, J, hx, hy, dt, r)) / 64) ** 2)
