"""Tests for morie.fn.smutl -- simulate utility shocks."""
import numpy as np
from morie.fn.smutl import simulate_utility_shocks, smutl


def test_alias():
    assert smutl is simulate_utility_shocks


def test_smoke():
    r = simulate_utility_shocks(n=50, sigma=2.0)
    assert r.name == "simulate_utility_shocks"
    assert r.extra["n"] == 50
    assert r.extra["sigma"] == 2.0
    assert len(r.extra["shocks"]) == 50
