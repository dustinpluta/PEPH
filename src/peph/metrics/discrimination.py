from __future__ import annotations

from typing import List

import numpy as np

from peph.metrics.kaplan_meier import fit_censoring_km


def c_index_harrell(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    """
    Harrell's C-index for right-censored data.
    Comparable pairs: i has event and T_i < T_j.
    Concordant if score_i > score_j (higher score = higher risk).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    s = np.asarray(score, dtype=float)

    n = t.size
    num = 0.0
    den = 0.0

    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if t[i] < t[j]:
                den += 1.0
                if s[i] > s[j]:
                    num += 1.0
                elif s[i] == s[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float("nan")


def _weighted_auc(scores: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted AUC for binary labels (1=case,0=control).
    Rank-based: P(score_case > score_control), with weights.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(w, dtype=float)

    case = y == 1
    ctrl = y == 0
    if case.sum() == 0 or ctrl.sum() == 0:
        return float("nan")

    s_case = s[case]
    s_ctrl = s[ctrl]
    w_case = w[case]
    w_ctrl = w[ctrl]

    num = 0.0
    den = float(np.sum(w_case) * np.sum(w_ctrl))
    for i in range(s_case.size):
        gt = (s_case[i] > s_ctrl).astype(float)
        eq = (s_case[i] == s_ctrl).astype(float)
        num += float(w_case[i] * np.sum(w_ctrl * (gt + 0.5 * eq)))
    return float(num / den) if den > 0 else float("nan")


def time_dependent_auc_ipcw(
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
    horizons: List[float],
) -> dict:
    """
    Cumulative/dynamic AUC at horizon tau:
      cases: event by tau
      controls: event-free at tau  (T >= tau)
    IPCW weights using censoring KM on evaluation set.

    IMPORTANT: use G(tau-) (left limit) to avoid G(tau)=0 when there is
    administrative censoring at exactly tau.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    s = np.asarray(score, dtype=float)

    kmG = fit_censoring_km(t, e)
    out = {}

    for tau in horizons:
        tau = float(tau)

        case = (t <= tau) & (e == 1)
        ctrl = t >= tau

        labels = np.zeros_like(e)
        labels[case] = 1
        labels[ctrl] = 0

        w = np.zeros_like(t, dtype=float)

        # KEY FIX: left-limit at tau
        Gtau = kmG.G(tau, left_limit=True)

        for i in range(t.size):
            if case[i]:
                Gi = kmG.G(t[i], left_limit=True)  # G(T_i-)
                w[i] = 0.0 if Gi <= 0 else 1.0 / Gi
            elif ctrl[i]:
                w[i] = 0.0 if Gtau <= 0 else 1.0 / Gtau
            else:
                w[i] = 0.0

        keep = case | ctrl
        auc = _weighted_auc(s[keep], labels[keep], w[keep])
        out[f"auc_t{int(tau)}"] = float(auc)

    return out