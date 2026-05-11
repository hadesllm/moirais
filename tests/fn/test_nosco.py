"""Tests for morie.fn.nosco -- compare W vs DW scores."""
import numpy as np
from morie.fn.nosco import nominate_score_types, nosco


def test_alias():
    assert nosco is nominate_score_types


def test_smoke():
    rng = np.random.default_rng(42)
    w = rng.standard_normal(20)
    dw = w + 0.1 * rng.standard_normal(20)
    r = nominate_score_types(w, dw)
    assert r.name == "nominate_score_types"
    assert r.value > 0.8  # high correlation


def test_extra():
    r = nominate_score_types([1, 2, 3], [1, 2, 3])
    assert "correlation" in r.extra
    assert abs(r.extra["correlation"] - 1.0) < 1e-10
