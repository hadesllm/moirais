"""Tests for renyi_entropy."""
import numpy as np
import pytest
from morie.fn.renyi import renyi_entropy, renyi


def test_alpha1_is_shannon():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    r1 = renyi_entropy(x, alpha=1.0, bins=20)
    from morie.fn.shent import shannon_entropy
    r2 = shannon_entropy(x, bins=20)
    assert abs(r1.estimate - r2.estimate) < 0.01


def test_alias():
    assert renyi is renyi_entropy


def test_bad_alpha():
    with pytest.raises(ValueError):
        renyi_entropy([1, 2], alpha=-1)


def test_alpha2():
    r = renyi_entropy([1, 2, 3, 4, 5], alpha=2.0, bins=5)
    assert r.estimate > 0
