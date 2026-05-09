# moirais.fn — function file (hadesllm/moirais)
"""K-means clustering with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from sklearn.cluster import KMeans


def kmeans2(X: Union[Sequence, np.ndarray],
            n_clusters: int = 3,
            n_init: int = 10,
            random_state: int = 42):
    """K-means clustering."""
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float)
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    sizes = np.bincount(labels, minlength=n_clusters)
    rows = [[f"Cluster {i}", int(sizes[i]),
             ", ".join(f"{c:.3g}" for c in km.cluster_centers_[i])]
            for i in range(n_clusters)]
    return RichResult(
        title="K-means clustering",
        summary_lines=[
            ("k (clusters)", n_clusters), ("n observations", len(X)),
            ("Inertia (within-cluster SSE)", float(km.inertia_)),
            ("n_init restarts", n_init),
        ],
        tables=[{
            "title": "Cluster summary:",
            "headers": ["Cluster", "Size", "Centroid"],
            "rows": rows,
        }],
        warnings=[] if min(sizes) > 1 else
                 ["one or more clusters has only 1 point - check k or initialization."],
        payload={"labels": labels.tolist(),
                 "centroids": km.cluster_centers_.tolist(),
                 "inertia": float(km.inertia_)},
    )
