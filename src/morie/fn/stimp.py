"""Format stimulus positions for plotting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def plot_stimuli_positions(stim_pos, labels=None) -> DescriptiveResult:
    """Prepare formatted stimulus position data for visualization.

    :param stim_pos: Stimulus positions (1-D or 2-D).
    :param labels: Optional stimulus labels.
    :return: DescriptiveResult with formatted coordinates.

    .. epigraph:: Logic is the foundation of all certain knowledge. -- Leonhard Euler
    """
    import numpy as np

    pos = np.asarray(stim_pos, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    n_stim = pos.shape[0]
    if labels is None:
        labels = [f"S{i + 1}" for i in range(n_stim)]
    result = {"positions": pos.tolist(), "labels": list(labels), "n_stimuli": n_stim, "n_dims": pos.shape[1]}
    return DescriptiveResult(name="plot_stimuli_positions", value=n_stim, extra=result)


stimp = plot_stimuli_positions


def cheatsheet() -> str:
    return "plot_stimuli_positions({}) -> Format stimulus positions for plotting."
