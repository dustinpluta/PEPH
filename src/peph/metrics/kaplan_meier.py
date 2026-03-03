from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class KM:
    times: np.ndarray      # unique event times (sorted)
    surv: np.ndarray       # survival just after time (right-continuous)
    n_risk: np.ndarray
    n_event: np.ndarray

    def G(self, t: float, *, left_limit: bool = False) -> float:
        """
        Evaluate survival at time t.
        - right-continuous by default: G(t)
        - left_limit=True gives G(t-) (product over times < t)
        """
        if self.times.size == 0:
            return 1.0
        if left_limit:
            idx = np.searchsorted(self.times, t, side="left") - 1
        else:
            idx = np.searchsorted(self.times, t, side="right") - 1
        if idx < 0:
            return 1.0
        return float(self.surv[idx])


def fit_km(time: np.ndarray, event: np.ndarray) -> KM:
    """
    Kaplan–Meier for survival with event indicator (1=event occurred).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)

    if t.size == 0:
        return KM(times=np.array([]), surv=np.array([]), n_risk=np.array([]), n_event=np.array([]))

    order = np.argsort(t)
    t = t[order]
    e = e[order]

    uniq = np.unique(t[e == 1])
    if uniq.size == 0:
        return KM(times=np.array([]), surv=np.array([]), n_risk=np.array([]), n_event=np.array([]))

    surv_vals = []
    n_risk_vals = []
    n_event_vals = []

    S = 1.0
    n = t.size
    for tj in uniq:
        at_risk = np.sum(t >= tj)
        d = np.sum((t == tj) & (e == 1))
        if at_risk <= 0:
            continue
        S *= (1.0 - d / at_risk)
        surv_vals.append(S)
        n_risk_vals.append(at_risk)
        n_event_vals.append(d)

    return KM(times=uniq, surv=np.asarray(surv_vals), n_risk=np.asarray(n_risk_vals), n_event=np.asarray(n_event_vals))


def fit_censoring_km(time: np.ndarray, event: np.ndarray) -> KM:
    """
    KM for censoring survival G(t) where "event" is censoring.
    For right-censored survival data, censoring indicator is (event==0).
    """
    censor_event = (np.asarray(event, dtype=int) == 0).astype(int)
    return fit_km(np.asarray(time, dtype=float), censor_event)