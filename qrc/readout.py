"""Ridge readout on standardised reservoir features (closed form, unpenalised intercept)."""
import sys

import numpy as np


def fit_ridge(X, y, lam, names=None):
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    if not keep.all():                 # constant features cannot be standardised: weight 0
        dropped = [names[j] if names is not None else f"column {j}" for j in np.flatnonzero(~keep)]
        print(f"WARNING fit_ridge: constant training feature(s) {', '.join(map(str, dropped))} "
              "get weight 0", file=sys.stderr)
        w = np.zeros(X.shape[1])
        w[keep], _ = fit_ridge(X[:, keep], y, lam)
        return w, y.mean() - mu @ w
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    w = beta / sd                      # weights in raw expectation-value units
    b = y.mean() - mu @ w
    return w, b
