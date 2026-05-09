"""Tests for moirais.fn.plpty -- plot by party."""
import numpy as np
from moirais.fn.plpty import plot_by_party, plpty


def test_alias():
    assert plpty is plot_by_party


def test_smoke():
    X = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1], [-0.2, -0.3]])
    labels = np.array(["D", "R", "D", "R"])
    r = plot_by_party(X, labels)
    assert r.name == "plot_by_party"
    assert r.extra["n_parties"] == 2
    assert "D" in r.extra["groups"]
    assert "R" in r.extra["groups"]
