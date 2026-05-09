# moirais.fn — function file (hadesllm/moirais)
"""Rectangular matrix formatter. 'Spirit Gun!' -- Yusuke, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def rectangular_matrix(respondents, stimuli):
    """Format a rectangular preference matrix (respondents x stimuli).

    Parameters
    ----------
    respondents : array-like
        Matrix of respondent ratings (n_resp x n_stim).
    stimuli : list of str or None
        Stimulus labels.

    Returns
    -------
    DescriptiveResult
        value = formatted matrix (ndarray), extra has shape info.
    """
    import numpy as np

    R = np.asarray(respondents, dtype=float)
    n_resp, n_stim = R.shape
    if stimuli is None:
        stimuli = [f"S{i + 1}" for i in range(n_stim)]
    return DescriptiveResult(
        name="rectangular_matrix",
        value=R,
        extra={"n_resp": n_resp, "n_stim": n_stim, "stimuli": list(stimuli)},
    )


rcmat = rectangular_matrix


def cheatsheet() -> str:
    return "rectangular_matrix({}) -> Rectangular matrix formatter. 'Spirit Gun!' -- Yusuke, Yu Yu"
