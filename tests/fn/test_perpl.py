"""Tests for perplexity."""
import numpy as np
import pytest
from morie.fn.perpl import perplexity, perpl


def test_uniform():
    n = 100
    log_probs = np.full(n, np.log(1.0 / 8))
    r = perplexity(log_probs)
    assert r.estimate == pytest.approx(8.0, rel=0.01)


def test_alias():
    assert perpl is perplexity


def test_empty_raises():
    with pytest.raises(ValueError):
        perplexity([])
