"""Tests for moirais.fn.covnm — covering number estimation."""

import numpy as np
import pytest

from moirais.fn.covnm import covnm


def test_basic_output():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((50, 2))
    result = covnm(pts)
    assert "covering_numbers" in result
    assert "log_covering" in result


def test_covering_decreases_with_epsilon():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((30, 2))
    result = covnm(pts, n_eps=10)
    cn = result["covering_numbers"]
    assert cn[0] >= cn[-1]


def test_single_point():
    pts = np.array([[0.0, 0.0]])
    result = covnm(pts)
    assert np.all(result["covering_numbers"] == 1)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        covnm(np.array([]))
