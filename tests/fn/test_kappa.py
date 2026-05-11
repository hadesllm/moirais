"""Tests for morie.fn.kappa — Cohen's kappa."""
import numpy as np
import pytest

from morie.fn.kappa import cohens_kappa, kappa


def test_perfect_agreement():
    """Identical raters should give kappa = 1."""
    r = [0, 1, 2, 0, 1, 2, 0, 1]
    result = cohens_kappa(r, r)
    assert abs(result.estimate - 1.0) < 1e-10


def test_random_agreement():
    """Random raters should give kappa near 0."""
    rng = np.random.default_rng(42)
    r1 = rng.integers(0, 3, size=200)
    r2 = rng.integers(0, 3, size=200)
    result = cohens_kappa(r1, r2)
    assert abs(result.estimate) < 0.3


def test_kappa_alias():
    assert kappa is cohens_kappa


def test_mismatched_length():
    with pytest.raises(ValueError):
        cohens_kappa([1, 2], [1, 2, 3])
