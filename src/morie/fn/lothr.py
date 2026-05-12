# morie.fn -- function file (hadesllm/morie)
"""Template matching (NCC). 'All our lives we fought this war.' -- Logos crew"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def template_match(
    signal: np.ndarray,
    template: np.ndarray,
    *,
    method: str = "ncc",
) -> DescriptiveResult:
    """Find a template pattern within a signal via cross-correlation.

    Parameters
    ----------
    signal : ndarray
        1-D or 2-D signal/image to search in.
    template : ndarray
        1-D or 2-D pattern to find (must be smaller than signal).
    method : str
        ``"ncc"`` for normalised cross-correlation, ``"ssd"`` for sum of
        squared differences.

    Returns
    -------
    DescriptiveResult
        ``value`` is the best match score; ``extra`` has the match
        position and the score array.
    """
    sig = np.asarray(signal, dtype=np.float64)
    tpl = np.asarray(template, dtype=np.float64)
    if sig.ndim != tpl.ndim:
        raise ValueError("signal and template must have the same dimensionality")

    if sig.ndim == 1:
        n, m = len(sig), len(tpl)
        if m > n:
            raise ValueError("template longer than signal")
        scores = np.zeros(n - m + 1)
        tpl_c = tpl - tpl.mean()
        tpl_norm = np.sqrt(np.sum(tpl_c**2))
        if tpl_norm == 0:
            tpl_norm = 1.0
        for i in range(n - m + 1):
            window = sig[i : i + m]
            wc = window - window.mean()
            w_norm = np.sqrt(np.sum(wc**2))
            if method == "ncc":
                scores[i] = np.sum(wc * tpl_c) / (w_norm * tpl_norm) if w_norm > 0 else 0.0
            elif method == "ssd":
                scores[i] = -np.sum((window - tpl) ** 2)
            else:
                raise ValueError(f"method must be 'ncc' or 'ssd', got '{method}'")
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        pos = best_idx
    elif sig.ndim == 2:
        sr, sc = sig.shape
        tr, tc = tpl.shape
        if tr > sr or tc > sc:
            raise ValueError("template larger than signal")
        scores = np.zeros((sr - tr + 1, sc - tc + 1))
        tpl_c = tpl - tpl.mean()
        tpl_norm = np.sqrt(np.sum(tpl_c**2))
        if tpl_norm == 0:
            tpl_norm = 1.0
        for i in range(sr - tr + 1):
            for j in range(sc - tc + 1):
                window = sig[i : i + tr, j : j + tc]
                wc = window - window.mean()
                w_norm = np.sqrt(np.sum(wc**2))
                if method == "ncc":
                    scores[i, j] = np.sum(wc * tpl_c) / (w_norm * tpl_norm) if w_norm > 0 else 0.0
                elif method == "ssd":
                    scores[i, j] = -np.sum((window - tpl) ** 2)
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        best_score = float(scores[idx])
        pos = list(idx)
    else:
        raise ValueError("signal must be 1-D or 2-D")

    return DescriptiveResult(
        name="Template Match",
        value=best_score,
        extra={
            "position": pos,
            "method": method,
            "signal_shape": list(sig.shape),
            "template_shape": list(tpl.shape),
        },
    )


lothr = template_match


def cheatsheet() -> str:
    return "template_match({}) -> Template matching (NCC). 'All our lives we fought this war.'"
