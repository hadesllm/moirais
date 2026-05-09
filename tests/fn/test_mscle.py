"""Tests for multiscale entropy."""
import numpy as np
from moirais.fn.mscle import multiscale_entropy, mscle


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r = multiscale_entropy(x, m=2, max_scale=3)
    assert np.isfinite(r.estimate) or r.estimate >= 0


def test_alias():
    assert mscle is multiscale_entropy


def test_has_scales():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r = multiscale_entropy(x, max_scale=3)
    assert len(r.extra["scales"]) > 0
