"""Tests for morie.fn.plcut -- cutting lines plot data."""
import numpy as np
from morie.fn.plcut import plot_cutting_lines_data, plcut


def test_alias():
    assert plcut is plot_cutting_lines_data


def test_smoke():
    normals = np.array([[1.0, 1.0], [0.0, 1.0]])
    cutpoints = np.array([0.5, 0.3])
    X = np.random.default_rng(42).standard_normal((10, 2))
    r = plot_cutting_lines_data(normals, cutpoints, X)
    assert r.name == "plot_cutting_lines_data"
    assert "lines" in r.extra
    assert len(r.extra["lines"]) == 2
