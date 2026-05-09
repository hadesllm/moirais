# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Calibration via Platt scaling and isotonic regression.

Post-hoc probability calibration transforms raw classifier scores
into well-calibrated probability estimates.

References
----------
Platt, J. (1999). Probabilistic outputs for support vector machines
and comparisons to regularized likelihood methods.
*Advances in Large Margin Classifiers*, 10(3), 61-74.

Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
accurate multiclass probability estimates.
*Proceedings of ACM SIGKDD*, 694-699.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities
with supervised learning. *Proceedings of ICML*, 625-632.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logit
from ._richresult import RichResult

__all__ = ["calbt"]


def calbt(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    method: str = "platt",
    n_bins: int = 10,
) -> dict[str, Any]:
    r"""Calibrate predicted probabilities.

    **Platt scaling** fits a logistic regression on the logit of
    predicted probabilities to learn a linear calibration:

    .. math::

        P(y=1 \mid f) = \sigma(A \cdot f + B)

    where :math:`A` and :math:`B` are fitted via maximum likelihood.

    **Isotonic regression** fits a non-decreasing step function
    via pool adjacent violators (PAV) algorithm.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels ``{0, 1}``, shape ``(n,)``.
    y_prob : np.ndarray
        Raw predicted probabilities, shape ``(n,)``.
    method : str
        ``"platt"`` or ``"isotonic"``.
    n_bins : int
        Number of bins for reliability diagram (evaluation).

    Returns
    -------
    dict
        ``calibrated_probs`` (shape ``(n,)``),
        ``ece`` (expected calibration error of calibrated probs),
        ``ece_before`` (ECE before calibration),
        ``platt_A``, ``platt_B`` (if method="platt"),
        ``reliability_before``, ``reliability_after``
        (bin midpoints and mean accuracies),
        ``n``, ``method``.

    References
    ----------
    Platt (1999). Advances in Large Margin Classifiers, 61-74.
    Zadrozny & Elkan (2002). KDD, 694-699.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0 and 1.")

    n = len(y_true)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    ece_before = _ece(y_true, y_prob, n_bins)
    rel_before = _reliability(y_true, y_prob, n_bins)

    A, B = 1.0, 0.0

    if method == "platt":
        # Fit A and B via MLE using scipy optimize
        def _nll(params):
            a, b = params
            p = expit(a * logit(y_prob_clipped) + b)
            p = np.clip(p, 1e-8, 1 - 1e-8)
            return -float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

        from scipy.optimize import minimize
        res = minimize(_nll, x0=[1.0, 0.0], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
        A, B = float(res.x[0]), float(res.x[1])
        calibrated = expit(A * logit(y_prob_clipped) + B)

    elif method == "isotonic":
        calibrated = _isotonic_regression(y_prob, y_true)
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'platt' or 'isotonic'.")

    ece_after = _ece(y_true, calibrated, n_bins)
    rel_after = _reliability(y_true, calibrated, n_bins)

    result = {
        "calibrated_probs": calibrated,
        "ece": ece_after,
        "ece_before": ece_before,
        "reliability_before": rel_before,
        "reliability_after": rel_after,
        "n": n,
        "method": f"calibration-{method}",
    }
    if method == "platt":
        result["platt_A"] = A
        result["platt_B"] = B

    return result


def _ece(y_true, y_prob, n_bins):
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def _reliability(y_true, y_prob, n_bins):
    """Return (bin_centers, fraction_positive) for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    frac_pos = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            frac_pos[i] = float(y_true[mask].mean())
    return RichResult(payload={"bin_centers": centers, "fraction_positive": frac_pos})


def _isotonic_regression(x, y):
    """Pool-adjacent violators (PAV) isotonic regression."""
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order].copy().astype(float)
    n = len(y_s)

    # PAV
    blocks = [[float(y_s[i]), 1] for i in range(n)]
    merged = True
    while merged:
        merged = False
        i = 0
        new_blocks = []
        while i < len(blocks):
            if i + 1 < len(blocks) and blocks[i][0] > blocks[i + 1][0]:
                # Merge
                total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
                cnt = blocks[i][1] + blocks[i + 1][1]
                new_blocks.append([total / cnt, cnt])
                i += 2
                merged = True
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks

    # Expand blocks
    result = np.empty(n)
    pos = 0
    for val, cnt in blocks:
        result[pos: pos + cnt] = val
        pos += cnt

    # Unsort
    out = np.empty(n)
    out[order] = result
    return out


def cheatsheet() -> str:
    return "calbt(y_true, y_prob) -> Platt/isotonic calibration (Platt 1999; Zadrozny & Elkan 2002)."
