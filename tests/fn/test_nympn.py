"""Tests for neyman_pearson."""
import numpy as np
import pytest
from morie.fn.nympn import neyman_pearson, nympn


def test_basic():
    scores = np.arange(100, dtype=float)
    r = neyman_pearson(scores, alpha=0.05)
    assert r.estimate >= 94


def test_alias():
    assert nympn is neyman_pearson


def test_bad_alpha():
    with pytest.raises(ValueError):
        neyman_pearson([1, 2], alpha=0.0)
