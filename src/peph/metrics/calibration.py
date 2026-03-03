from __future__ import annotations

from typing import Dict, List

import numpy as np
import statsmodels.api as sm

from peph.metrics.kaplan_meier import fit_censoring_km


def brier_ipcw(time: np.ndarray, event: np.ndarray, pred_risk: np.ndarray, horizons: List[float]) -> Dict[str, float]:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    pr = np.asarray(pred_risk, dtype=float)

    kmG = fit_censoring_km(t, e)
    out: Dict[str, float] = {}

    for tau in horizons:
        tau = float(tau)
        y = ((t <= tau) & (e == 1)).astype(float)

        w = np.zeros_like(t, dtype=float)

        # KEY FIX: left-limit at tau
        Gtau = kmG.G(tau, left_limit=True)

        for i in range(t.size):
            if t[i] <= tau and e[i] == 1:
                Gi = kmG.G(t[i], left_limit=True)  # G(T_i-)
                w[i] = 0.0 if Gi <= 0 else 1.0 / Gi
            elif t[i] >= tau:
                # include T == tau as known event-free at tau (admin censoring)
                w[i] = 0.0 if Gtau <= 0 else 1.0 / Gtau
            else:
                w[i] = 0.0  # censored before tau

        mask = w > 0
        if mask.sum() == 0:
            out[f"brier_t{int(tau)}"] = float("nan")
            continue
        bs = np.average((y[mask] - pr[mask]) ** 2, weights=w[mask])
        out[f"brier_t{int(tau)}"] = float(bs)

    return out


def calibration_logistic_ipcw(
    time: np.ndarray,
    event: np.ndarray,
    pred_risk: np.ndarray,
    horizons: List[float],
) -> Dict[str, float]:
    """
    Calibration-in-the-large and slope at horizon tau using weighted logistic regression:
      outcome: I(T<=tau, event=1)
      predictor: logit(pred_risk)
      weights: IPCW at tau

    IMPORTANT: use G(tau-) to avoid G(tau)=0 with admin censoring at exactly tau.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    pr = np.asarray(pred_risk, dtype=float)

    kmG = fit_censoring_km(t, e)
    out: Dict[str, float] = {}

    pr_clip = np.clip(pr, 1e-6, 1.0 - 1e-6)
    lp = np.log(pr_clip / (1.0 - pr_clip))  # logit

    for tau in horizons:
        tau = float(tau)
        y = ((t <= tau) & (e == 1)).astype(float)

        w = np.zeros_like(t, dtype=float)

        # KEY FIX: left-limit at tau
        Gtau = kmG.G(tau, left_limit=True)

        for i in range(t.size):
            if t[i] <= tau and e[i] == 1:
                Gi = kmG.G(t[i], left_limit=True)
                w[i] = 0.0 if Gi <= 0 else 1.0 / Gi
            elif t[i] >= tau:
                w[i] = 0.0 if Gtau <= 0 else 1.0 / Gtau
            else:
                w[i] = 0.0

        mask = w > 0
        if mask.sum() < 10:
            out[f"cal_int_t{int(tau)}"] = float("nan")
            out[f"cal_slope_t{int(tau)}"] = float("nan")
            continue

        X = sm.add_constant(lp[mask], has_constant="add")
        model = sm.GLM(y[mask], X, family=sm.families.Binomial(), freq_weights=w[mask])
        res = model.fit()

        out[f"cal_int_t{int(tau)}"] = float(res.params[0])
        out[f"cal_slope_t{int(tau)}"] = float(res.params[1])

    return out


def observed_risk_ipcw(time: np.ndarray, event: np.ndarray, tau: float) -> float:
    """
    IPCW estimate of P(T<=tau, event=1), with event-free at tau defined as T >= tau.
    Uses G(tau-) to handle admin censoring at exactly tau.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    kmG = fit_censoring_km(t, e)

    y = ((t <= tau) & (e == 1)).astype(float)
    w = np.zeros_like(t, dtype=float)

    # KEY FIX: left-limit at tau
    Gtau = kmG.G(float(tau), left_limit=True)

    for i in range(t.size):
        if t[i] <= tau and e[i] == 1:
            Gi = kmG.G(t[i], left_limit=True)
            w[i] = 0.0 if Gi <= 0 else 1.0 / Gi
        elif t[i] >= tau:
            w[i] = 0.0 if Gtau <= 0 else 1.0 / Gtau
        else:
            w[i] = 0.0

    mask = w > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(y[mask], weights=w[mask]))