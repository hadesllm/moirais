"""Tests for moirais.fn.slice -- slice sampling."""

import numpy as np
from moirais.fn.slice import slice_sampler


def test_returns_dict():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=100)
    assert isinstance(result, dict)
    assert "samples" in result


def test_correct_length():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=500)
    assert len(result["samples"]) == 500


def test_standard_normal_mean():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=10000, seed=42)
    assert abs(np.mean(result["samples"][1000:])) < 0.2


def test_standard_normal_var():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=10000, seed=42)
    assert abs(np.var(result["samples"][1000:]) - 1.0) < 0.5


def test_invalid_width():
    try:
        slice_sampler(lambda x: 0.0, width=-1)
        assert False
    except ValueError:
        pass


def test_reproducibility():
    f = lambda x: -0.5 * x ** 2
    r1 = slice_sampler(f, n_iter=100, seed=99)
    r2 = slice_sampler(f, n_iter=100, seed=99)
    np.testing.assert_array_equal(r1["samples"], r2["samples"])
