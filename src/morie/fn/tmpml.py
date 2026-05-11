"""Template library matching via normalized cross-correlation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def template_match_lib(signal, templates, **kwargs) -> DescriptiveResult:
    """Match signal against a library of templates using normalized xcorr.

    Parameters
    ----------
    signal : array-like
        Input signal.
    templates : list of array-like
        Library of template waveforms.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    best_corr = -np.inf
    best_idx = 0
    best_lag = 0
    all_corrs = []
    for i, tmpl in enumerate(templates):
        tmpl = np.asarray(tmpl, dtype=float)
        ns = np.linalg.norm(signal)
        nt = np.linalg.norm(tmpl)
        if ns < 1e-12 or nt < 1e-12:
            all_corrs.append(0.0)
            continue
        corr = np.correlate(signal, tmpl, mode="full") / (ns * nt)
        peak = float(np.max(corr))
        lag = int(np.argmax(corr)) - (len(tmpl) - 1)
        all_corrs.append(peak)
        if peak > best_corr:
            best_corr = peak
            best_idx = i
            best_lag = lag
    return DescriptiveResult(
        name="template_match_lib",
        value=float(best_corr) if best_corr > -np.inf else 0.0,
        extra={
            "best_template_index": best_idx,
            "best_lag": best_lag,
            "all_correlations": np.array(all_corrs),
            "n_templates": len(templates),
        },
    )


tmpml = template_match_lib


def cheatsheet() -> str:
    return "template_match_lib({}) -> Template library matching via normalized cross-correlation."
