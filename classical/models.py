"""Classical forecasters: Ridge, Poisson/NB2 GLMs, XGBoost, ESN, vanilla RNN and GRU.

All models predict C_{t+1} from the F2/F3/F4 inputs of day t (claims, policies)
and day t+1 (weekday, holiday). Scalers are fitted on training rows only.
Chronological CV uses five expanding folds (scikit-learn TimeSeriesSplit).
"""
import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import xgboost as xgb
import torch
import torch.nn as nn

C, P, Y = "property_damage_claim_count_t", "active_policy_count_t", "target_property_damage_claim_count"
H = "is_public_holiday_t_plus_1"
FEATURES = {"F2": [C, "D_sin", "D_cos"],
            "F3": [C, "D_sin", "D_cos", P],
            "F4": [C, "D_sin", "D_cos", P, H]}
WINDOWS = [1, 2, 5, 7, 14, 21, 28]
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]      # ESN reservoir seeds
NEURAL_SEEDS = list(range(42, 52))                              # RNN/GRU initialization seeds
N_TRAIN = 1095


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def folds(n):
    return list(TimeSeriesSplit(5).split(np.arange(n)))


def ridge_design(train, frame, feature):
    """Claims and policies standardized with training statistics; calendar inputs unchanged."""
    cols = FEATURES[feature]
    X = frame[cols].to_numpy(float).copy()
    for j, col in enumerate(cols):
        if col in (C, P):
            m, s = train[col].mean(), train[col].std(ddof=0)
            X[:, j] = (X[:, j] - m) / s
    return X


def ridge_fit_predict(train, frame, feature, alpha):
    model = Ridge(alpha=alpha).fit(ridge_design(train, train, feature), train[Y])
    return model.predict(ridge_design(train, frame, feature)), model


GLM_SPECS = [("F2", "no_policy")] + [(f, v) for f in ("F3", "F4")
                                     for v in ("standardized_policy", "fixed_log_policy_offset", "learned_log_policy")]


def glm_design(train, frame, feature, variant):
    """Log-link design: exposure as standardized covariate, fixed log offset, or learned log slope."""
    X = pd.DataFrame({"const": 1.0}, index=frame.index)
    X["C_t_z"] = (frame[C] - train[C].mean()) / train[C].std(ddof=0)
    X["D_sin"], X["D_cos"] = frame.D_sin, frame.D_cos
    if variant == "standardized_policy":
        X["P_t_z"] = (frame[P] - train[P].mean()) / train[P].std(ddof=0)
    if feature == "F4":
        X[H] = frame[H]
    if variant == "learned_log_policy":
        X["log_P_centered"] = np.log(frame[P]) - np.log(train[P]).mean()
    offset = np.log(frame[P].to_numpy(float)) if variant == "fixed_log_policy_offset" else np.zeros(len(frame))
    return X.astype(float), offset


def glm_fit_predict(train, frame, feature, variant, family):
    """family: 'poisson' (IRLS) or 'nb' (NB2, dispersion by maximum likelihood)."""
    X, off = glm_design(train, train, feature, variant)
    Xe, off_e = glm_design(train, frame, feature, variant)
    y = train[Y].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "poisson":
            fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(maxiter=100, tol=1e-8)
        else:
            fit = NegativeBinomial(y, X, loglike_method="nb2", offset=off).fit(method="bfgs", maxiter=1000, disp=False)
    return np.asarray(fit.predict(Xe, offset=off_e), dtype=float), fit


def xgb_model(cfg):
    return xgb.XGBRegressor(objective=cfg["objective"], n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                            learning_rate=cfg["learning_rate"], min_child_weight=cfg["min_child_weight"],
                            reg_lambda=cfg["reg_lambda"], reg_alpha=0.0, gamma=0.0, subsample=1.0,
                            colsample_bytree=1.0, tree_method="hist", max_bin=256, random_state=42, n_jobs=1,
                            verbosity=0)


def windows(raw, endpoints, w):
    """Input windows ending at each endpoint: shape (n, w, d)."""
    return raw[np.asarray(endpoints)[:, None] - np.arange(w - 1, -1, -1)]


def scaled(raw, last_train, continuous):
    """Standardize claims/policies on the unique input rows up to the last training endpoint."""
    hist = raw[: last_train + 1]
    mean, scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    mean[continuous] = hist[:, continuous].mean(0)
    scale[continuous] = hist[:, continuous].std(0)
    scale[scale == 0] = 1.0
    return (raw - mean) / scale, mean, scale


def continuous_idx(feature):
    return [FEATURES[feature].index(c) for c in (C, P) if c in FEATURES[feature]]


ESN_ALPHAS = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0]


def esn_reservoir(size, dim, seed, connectivity):
    """Sparse uniform recurrent matrix normalized to unit spectral radius, and input matrix."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1, 1, (size, size))
    W *= rng.random((size, size)) < connectivity
    W /= np.max(np.abs(np.linalg.eigvals(W)))
    W_in = np.random.default_rng(seed + 10000 + dim).uniform(-1, 1, (size, dim)) / np.sqrt(dim)
    return W, W_in


def esn_states(X, W, W_in, rho, scaling, leak):
    """Leaky-tanh state after reading each window from a zero state."""
    s = np.zeros((len(X), len(W)))
    for k in range(X.shape[1]):
    # s_k = (1 - leak) * s_{k-1} + leak * tanh(scaling * W_in @ x_k + rho * W @ s_{k-1})
    # Each window starts from s_0 = 0; return its final reservoir state.
        s = (1 - leak) * s + leak * np.tanh(X[:, k] @ (scaling * W_in).T + s @ (rho * W).T)
    return s


def esn_prepare(data, feature, w):
    """Fold arrays (train windows, targets, assessment windows, targets) and the final 2022-2024 fit arrays."""
    raw, y = data[FEATURES[feature]].to_numpy(float), data[Y].to_numpy(float)
    ends = np.arange(w - 1, N_TRAIN)
    out = []
    for tr, va in folds(len(ends)):
        Z, _, _ = scaled(raw, ends[tr][-1], continuous_idx(feature))
        out.append((windows(Z, ends[tr], w), y[ends[tr]], windows(Z, ends[va], w), y[ends[va]]))
    Z, mean, scale = scaled(raw, N_TRAIN - 1, continuous_idx(feature))
    return out, (Z, ends, mean, scale)


def esn_cv(fold_arrays, W, W_in, cfg):
    """Fold RMSE for every readout alpha; alpha chosen by mean CV RMSE, then SD."""
    errs = []
    for X, y, V, v in fold_arrays:
        R = esn_states(X, W, W_in, cfg["spectral_radius"], cfg["input_scaling"], cfg["leak_rate"])
        S = esn_states(V, W, W_in, cfg["spectral_radius"], cfg["input_scaling"], cfg["leak_rate"])
        errs.append([rmse(Ridge(alpha=a).fit(R, y).predict(S), v) for a in ESN_ALPHAS])
    errs = np.asarray(errs).T
    i = min(range(len(ESN_ALPHAS)), key=lambda k: (errs[k].mean(), errs[k].std(), ESN_ALPHAS[k]))
    return dict(ridge_alpha=ESN_ALPHAS[i], cv_rmse=float(errs[i].mean()), cv_std=float(errs[i].std()))


def esn_fit_predict(data, cfg, seed, endpoints):
    """Fit the ESN readout on 2022-2024 and forecast the given endpoints."""
    feat, w = cfg["feature_set"], int(cfg["window_length"])
    _, (Z, ends, _, _) = esn_prepare(data, feat, w)
    W, W_in = esn_reservoir(int(cfg["reservoir_size"]), len(FEATURES[feat]), seed, cfg["requested_connectivity"])
    args = (W, W_in, cfg["spectral_radius"], cfg["input_scaling"], cfg["leak_rate"])
    model = Ridge(alpha=cfg["ridge_alpha"]).fit(esn_states(windows(Z, ends, w), *args), data[Y].to_numpy(float)[ends])
    return model.predict(esn_states(windows(Z, endpoints, w), *args))


MAX_EPOCHS = {"rnn": 200, "gru": 2000}
PATIENCE = {"rnn": 20, "gru": 30}
MIN_DELTA, INNER_FRACTION, BATCH = 1e-4, 0.2, 64


def neural_class(kind):

    class RNN(nn.Module):
        def __init__(self, d, hidden, layers):
            super().__init__()
            self.rnn = nn.RNN(d, hidden, layers, nonlinearity="tanh", batch_first=True)
            self.readout = nn.Linear(hidden, 1)
            for k in range(layers):
                nn.init.xavier_uniform_(getattr(self.rnn, f"weight_ih_l{k}"))
                nn.init.orthogonal_(getattr(self.rnn, f"weight_hh_l{k}"))
                nn.init.zeros_(getattr(self.rnn, f"bias_ih_l{k}"))
                nn.init.zeros_(getattr(self.rnn, f"bias_hh_l{k}"))
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

        def forward(self, x):
            return self.readout(self.rnn(x)[0][:, -1]).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self, d, hidden, layers):
            super().__init__()
            self.gru = nn.GRU(d, hidden, layers, batch_first=True)
            self.output = nn.Linear(hidden, 1)
            for k in range(layers):
                for g in getattr(self.gru, f"weight_ih_l{k}").chunk(3, 0):
                    nn.init.xavier_uniform_(g)
                for g in getattr(self.gru, f"weight_hh_l{k}").chunk(3, 0):
                    nn.init.orthogonal_(g)
                nn.init.zeros_(getattr(self.gru, f"bias_ih_l{k}"))
                nn.init.zeros_(getattr(self.gru, f"bias_hh_l{k}"))
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, x):
            return self.output(self.gru(x)[1][-1]).squeeze(-1)

    return RNN if kind == "rnn" else GRU


def neural_arrays(data, cfg, train_ends, eval_ends):
    feat = cfg["feature_set"]
    Z, mean, scale = scaled(data[FEATURES[feat]].to_numpy(float), int(train_ends[-1]), continuous_idx(feat))
    y = data[Y].to_numpy(float)
    return windows(Z, train_ends, cfg["window"]), y[train_ends], windows(Z, eval_ends, cfg["window"]), y[eval_ends]


def neural_predict(model, X, mean, scale):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().astype(float) * scale + mean


def neural_fit(kind, X, y, cfg, seed, epochs, stopping=None):
    """Adam, minibatches of 64, gradient-norm clipping at 1, standardized target.
    With `stopping`, return the epoch with the best hold-out RMSE (patience rule)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = neural_class(kind)(X.shape[2], cfg["hidden_size"], cfg["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    mean, scale = float(y.mean()), float(y.std()) or 1.0
    xs, ys = torch.tensor(X, dtype=torch.float32), torch.tensor((y - mean) / scale, dtype=torch.float32)
    best, best_epoch, stale = np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for i in range(0, len(y), BATCH):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xs[i:i + BATCH]), ys[i:i + BATCH])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if stopping is not None:
            score = rmse(neural_predict(model, stopping[0], mean, scale), stopping[1])
            best, best_epoch, stale = (score, epoch, 0) if score < best - MIN_DELTA else (best, best_epoch, stale + 1)
            if stale >= PATIENCE[kind]:
                break
    return model, (best_epoch if stopping is not None else epochs), mean, scale


def neural_cv(kind, data, cfg):
    """Outer chronological CV. In each fold the last 20 % of the training windows choose the epoch
    count, then the model is refitted on the whole fold training set. Final epochs = median over folds."""
    ends = np.arange(cfg["window"] - 1, N_TRAIN)
    scores, chosen = [], []
    for k, (tr, va) in enumerate(folds(len(ends)), 1):
        train, test = ends[tr], ends[va]
        n_stop = max(1, int(np.ceil(len(train) * INNER_FRACTION)))
        X, y, V, v = neural_arrays(data, cfg, train[:-n_stop], train[-n_stop:])
        _, epochs, _, _ = neural_fit(kind, X, y, cfg, 42 + k, MAX_EPOCHS[kind], stopping=(V, v))
        X, y, V, v = neural_arrays(data, cfg, train, test)
        model, _, m, s = neural_fit(kind, X, y, cfg, 42 + k, epochs)
        scores.append(rmse(neural_predict(model, V, m, s), v))
        chosen.append(epochs)
    return dict(cfg, cv_rmse=float(np.mean(scores)), cv_std=float(np.std(scores)), epochs=int(np.median(chosen)))


def neural_fit_predict(kind, data, cfg, seed, endpoints):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ends = np.arange(cfg["window"] - 1, N_TRAIN)
    X, y, V, _ = neural_arrays(data, cfg, ends, np.asarray(endpoints))
    model, _, m, s = neural_fit(kind, X, y, cfg, seed, cfg["epochs"])
    return neural_predict(model, V, m, s)


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
