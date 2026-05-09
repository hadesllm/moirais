# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Annotated signal plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_annotated_signal_fn(
    x: np.ndarray,
    fs: float = 1.0,
    annotations: np.ndarray | None = None,
    title: str = "Annotated Signal",
    annotation_label: str = "Events",
) -> DescriptiveResult:
    """Plot a signal with event annotations marked.

    Parameters
    ----------
    x : array-like
        1-D signal.
    fs : float
        Sampling frequency in Hz.
    annotations : np.ndarray, optional
        Sample indices of annotated events.
    title : str
        Plot title.
    annotation_label : str
        Legend label for annotation markers.

    Returns
    -------
    DescriptiveResult
        *value* is number of annotations; *extra* contains ``figure``.
    """
    from moirais._bioplot import plot_annotated_signal

    x = np.asarray(x, dtype=float)
    if annotations is not None:
        annotations = np.asarray(annotations, dtype=float)
    n_ann = len(annotations) if annotations is not None else 0
    fig = plot_annotated_signal(x, fs=fs, annotations=annotations, title=title, annotation_label=annotation_label)
    return DescriptiveResult(
        name="annotated_signal",
        value=n_ann,
        extra={"figure": fig, "n_annotations": n_ann},
    )


annplt = plot_annotated_signal_fn


def cheatsheet() -> str:
    return "plot_annotated_signal_fn({}) -> Annotated signal plot visualization."
