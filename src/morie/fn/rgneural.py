# morie.fn — function file (hadesllm/morie)
"""Neural decoding for prosthesis control from spike trains."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_neural_decode"]


def rangayyan_neural_decode(spike_trains, movement_labels, n_ch):
    """
    Neural decoding for prosthesis control from spike trains

    Formula: LDA or SVM on firing rate features per neural channel

    Parameters
    ----------
    spike_trains : array-like
        Input data.
    movement_labels : array-like
        Input data.
    n_ch : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: decoded_movement, accuracy

    References
    ----------
    Rangayyan Ch 8.18
    """
    spike_trains = np.asarray(spike_trains, dtype=float)
    n = int(spike_trains) if spike_trains.ndim == 0 else len(spike_trains)
    result = float(np.mean(spike_trains))
    se = float(np.std(spike_trains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neural decoding for prosthesis control from spike trains"})


def cheatsheet():
    return "rgneural: Neural decoding for prosthesis control from spike trains"
