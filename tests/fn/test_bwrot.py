"""Tests for morie.fn.bwrot — Bandwidth selection via rule-of-thumb."""

import numpy as np
import pytest

from morie.fn.bwrot import bwrot


@pytest.fixture()
def normal_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def test_returns_dict(normal_data):
    result = bwrot(normal_data)
    assert isinstance(result, dict)
    for key in ("bandwidth", "sigma", "iqr", "method", "n_obs"):
        assert key in result


def test_bandwidth_positive(normal_data):
    result = bwrot(normal_data)
    assert result["bandwidth"] > 0


def test_silverman_default(normal_data):
    result = bwrot(normal_data)
    assert result["method"] == "silverman"


def test_scott_method(normal_data):
    result = bwrot(normal_data, method="scott")
    assert result["method"] == "scott"
    assert result["bandwidth"] > 0


def test_epanechnikov_wider(normal_data):
    gauss = bwrot(normal_data, kernel="gaussian")
    epan = bwrot(normal_data, kernel="epanechnikov")
    assert epan["bandwidth"] > gauss["bandwidth"]


def test_sigma_positive(normal_data):
    result = bwrot(normal_data)
    assert result["sigma"] > 0


def test_n_obs(normal_data):
    result = bwrot(normal_data)
    assert result["n_obs"] == 200


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 2"):
        bwrot(np.array([1.0]))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        bwrot(np.array([1.0, 2.0]), method="plug-in")


def test_larger_n_smaller_bw():
    rng = np.random.default_rng(7)
    small = bwrot(rng.standard_normal(50))
    large = bwrot(rng.standard_normal(500))
    assert large["bandwidth"] < small["bandwidth"]
