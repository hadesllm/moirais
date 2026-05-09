# moirais.fn — function file (hadesllm/moirais)
"""Plot ideal points and stimuli. 'Spirit Bomb!' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_ideal_and_stimuli(X_resp, X_stim):
    """Combine respondent ideal points and stimulus coords for plotting.

    Parameters
    ----------
    X_resp : array-like
        Respondent ideal points (n_resp x p).
    X_stim : array-like
        Stimulus coordinates (n_stim x p).

    Returns
    -------
    DescriptiveResult
        value = dict with 'respondents' and 'stimuli' arrays.
    """
    import numpy as np

    Xr = np.asarray(X_resp, dtype=float)
    Xs = np.asarray(X_stim, dtype=float)
    return DescriptiveResult(
        name="plot_ideal_and_stimuli",
        value={"respondents": Xr, "stimuli": Xs},
        extra={"n_resp": Xr.shape[0], "n_stim": Xs.shape[0], "n_dims": Xr.shape[1]},
    )


plidl = plot_ideal_and_stimuli


def cheatsheet() -> str:
    return "plot_ideal_and_stimuli({}) -> Plot ideal points and stimuli. 'Spirit Bomb!' -- Goku, Drago"
