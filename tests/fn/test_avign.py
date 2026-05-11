"""Tests for anchoring vignettes."""
import numpy as np
from morie.fn.avign import avign


def test_avign_smoke():
    rng = np.random.default_rng(42)
    Y = rng.integers(1, 6, size=30).astype(float)
    V = rng.integers(1, 6, size=(30, 3)).astype(float)
    r = avign(Y, V)
    assert r.name == "anchoring_vignettes"
    assert "corrected_scores" in r.extra
    assert len(r.extra["corrected_scores"]) == 30


def test_cheatsheet():
    from morie.fn.avign import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
