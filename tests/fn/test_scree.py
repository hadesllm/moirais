"""Tests for moirais.fn.scree -- scree plot data."""

import numpy as np
from moirais.fn.scree import scree_plot_data, scree


def test_scree_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((6, 3))
    D = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    r = scree(D, max_dims=4)
    assert r.name == "scree_plot_data"
    assert len(r.value["dims"]) == 4
    assert len(r.value["stress_values"]) == 4


def test_scree_alias():
    assert scree is scree_plot_data
