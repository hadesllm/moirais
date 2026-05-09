"""Tests for moirais.fn.noidl -- extract ideal points."""
import numpy as np
from moirais.fn.noidl import nominate_ideal_extract, noidl


def test_alias():
    assert noidl is nominate_ideal_extract


def test_from_array():
    X = np.array([[-0.5, 0.2], [0.3, -0.1], [0.8, 0.4]])
    r = nominate_ideal_extract(X)
    assert r.name == "nominate_ideal_extract"
    assert r.extra["n_legislators"] == 3
    assert r.extra["n_dimensions"] == 2


def test_from_dict():
    r = nominate_ideal_extract({"ideal_points": [0.1, 0.2, 0.3]})
    assert r.extra["n_legislators"] == 3
