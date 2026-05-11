"""Tests for morie.fn.polyA -- Polya urn model."""

import numpy as np

from morie.fn.polya import polya_urn


def test_polya_smoke():
    result = polya_urn(n_draws=50)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_polya_seed_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    r1 = polya_urn(n_draws=100, rng=rng1)
    r2 = polya_urn(n_draws=100, rng=rng2)
    assert r1 == r2


def test_cheatsheet():
    from morie.fn.polya import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
