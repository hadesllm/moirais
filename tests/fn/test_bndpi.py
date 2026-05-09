"""Tests for moirais.fn.bndpi — Sheather-Jones plug-in bandwidth."""

import numpy as np
import pytest
from moirais.fn.bndpi import bndpi


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = bndpi(x)
    assert isinstance(result, dict)
    for key in ("h_opt", "roughness", "n_obs"):
        assert key in result


def test_h_opt_positive():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    assert bndpi(x)["h_opt"] > 0


def test_non_gaussian_raises():
    with pytest.raises(ValueError, match="Gaussian"):
        bndpi(np.ones(100), kernel="epanechnikov")


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 5"):
        bndpi(np.ones(3))
