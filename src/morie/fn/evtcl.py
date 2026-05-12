# morie.fn -- function file (hadesllm/morie)
"""Cluster detected events by morphological similarity."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Our greatest glory is not in never falling, but in rising every time we fall. -- Confucius"


def event_cluster(events, n_clusters=3, **kwargs) -> DescriptiveResult:
    """Cluster detected events by morphological similarity using k-means.

    Parameters
    ----------
    events : list of array-like
        List of event waveforms (may differ in length; zero-padded).
    n_clusters : int
        Number of clusters. Default 3.

    Returns
    -------
    DescriptiveResult
    """
    if len(events) == 0:
        raise ValueError("No events to cluster.")

    max_len = max(len(np.asarray(e)) for e in events)
    padded = np.zeros((len(events), max_len))
    for i, e in enumerate(events):
        arr = np.asarray(e, dtype=float)
        padded[i, : len(arr)] = arr

    n_clusters = min(n_clusters, len(events))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(events), size=n_clusters, replace=False)
    centroids = padded[idx].copy()

    labels = np.zeros(len(events), dtype=int)
    for _ in range(100):
        dists = np.array([np.sqrt(np.sum((padded - c) ** 2, axis=1)) for c in centroids]).T
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            members = padded[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

    return DescriptiveResult(
        name="event_cluster",
        value=float(n_clusters),
        extra={
            "labels": labels,
            "centroids": centroids,
            "n_clusters": n_clusters,
            "n_events": len(events),
        },
    )


evtcl = event_cluster


def cheatsheet() -> str:
    return "event_cluster({}) -> Cluster detected events by morphological similarity."
