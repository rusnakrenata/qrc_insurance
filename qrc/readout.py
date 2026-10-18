"""Ridge readout on standardised reservoir features (closed form, unpenalised intercept)."""
import numpy as np


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    w = beta / sd                      # weights in raw expectation-value units
    b = y.mean() - mu @ w
    return w, b
