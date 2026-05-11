"""Template library matching."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is the way."


def template_library(templates, signal, method="correlation", **kwargs) -> DescriptiveResult:
    """Store/match signal templates against a library.

    Parameters
    ----------
    templates : list of array-like
        Library of reference templates.
    signal : array-like
        Input signal to match against templates.
    method : str
        Matching method: ``"correlation"`` (default) or ``"euclidean"``.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    templates = [np.asarray(t, dtype=float) for t in templates]

    scores = []
    for t in templates:
        L = len(t)
        if len(signal) < L:
            scores.append(-np.inf)
            continue
        best = -np.inf if method == "correlation" else np.inf
        for start in range(len(signal) - L + 1):
            seg = signal[start : start + L]
            if method == "correlation":
                s_std = np.std(seg)
                t_std = np.std(t)
                if s_std > 0 and t_std > 0:
                    val = float(np.corrcoef(seg, t)[0, 1])
                else:
                    val = 0.0
                best = max(best, val)
            else:
                val = float(np.sqrt(np.sum((seg - t) ** 2)))
                best = min(best, val)
        scores.append(best)

    scores = np.array(scores)
    if method == "correlation":
        best_idx = int(np.argmax(scores))
    else:
        best_idx = int(np.argmin(scores))

    return DescriptiveResult(
        name="template_library",
        value=float(scores[best_idx]),
        extra={
            "scores": scores,
            "best_index": best_idx,
            "method": method,
            "n_templates": len(templates),
        },
    )


tmplb = template_library


def cheatsheet() -> str:
    return "template_library({}) -> Template library matching."
