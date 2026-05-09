"""Tests for total information content."""
import numpy as np
from moirais.fn.ticmp import total_information_content, ticmp


def test_positive():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    r = total_information_content(x, bins=8)
    assert r.estimate > 0


def test_scales_with_n():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 1000)
    r = total_information_content(x, bins=8)
    assert r.estimate > 100


def test_alias():
    assert ticmp is total_information_content
