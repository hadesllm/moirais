"""Tests for Wiener entropy."""
import numpy as np
import pytest
from moirais.fn.wnent import wiener_entropy, wnent


def test_noise():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    r = wiener_entropy(x)
    assert 0.5 < r.estimate <= 1.0


def test_sine():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 50 * t)
    r = wiener_entropy(x)
    assert r.estimate < 0.5


def test_alias():
    assert wnent is wiener_entropy
