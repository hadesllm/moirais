# morie.fn -- function file (hadesllm/morie)
"""Statistics is the grammar of science. -- Karl Pearson"""

from __future__ import annotations

from ._containers import DescriptiveResult


def label_mds_points(X, labels):
    """Attach labels to MDS coordinate matrix.

    Parameters
    ----------
    X : array-like
        Coordinate matrix (n x p).
    labels : list of str
        Labels for each point.

    Returns
    -------
    DescriptiveResult
        value = dict mapping label -> coordinate array.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    labeled = {str(lab): X[i].tolist() for i, lab in enumerate(labels)}
    return DescriptiveResult(name="label_mds_points", value=labeled, extra={"n": len(labels), "n_dims": X.shape[1]})


mdslb = label_mds_points


def cheatsheet() -> str:
    return "Statistics is the grammar of science. -- Karl Pearson"
