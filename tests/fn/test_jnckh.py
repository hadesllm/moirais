"""Test jnckh."""
import numpy as np
import pytest
from morie.fn.jnckh import jonckheere_terpstra


def test_jnckh_basic():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(20)
    b = rng.standard_normal(20) + 1
    c = rng.standard_normal(20) + 2
    r = jonckheere_terpstra(a, b, c)
    assert r.test_name == "Jonckheere-Terpstra"
    assert r.p_value < 0.05


def test_jnckh_no_trend():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(20)
    b = rng.standard_normal(20)
    r = jonckheere_terpstra(a, b)
    assert r.statistic >= 0
