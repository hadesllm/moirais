"""Tests for morie.fn.empbs — Empirical bootstrap process."""

import numpy as np
import pytest

from morie.fn.empbs import empbs


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def test_returns_dict(data):
    result = empbs(data)
    assert isinstance(result, dict)
    for key in ("eval_points", "ecdf", "cb_lower", "cb_upper", "sup_stats", "ks_critical_value", "n", "n_boot"):
        assert key in result


def test_ecdf_monotone(data):
    result = empbs(data)
    assert np.all(np.diff(result["ecdf"]) >= -1e-12)


def test_cb_brackets_ecdf(data):
    result = empbs(data)
    assert np.all(result["cb_lower"] <= result["ecdf"] + 1e-12)
    assert np.all(result["cb_upper"] >= result["ecdf"] - 1e-12)


def test_cb_in_01(data):
    result = empbs(data)
    assert np.all(result["cb_lower"] >= 0.0)
    assert np.all(result["cb_upper"] <= 1.0)


def test_ks_critical_positive(data):
    result = empbs(data)
    assert result["ks_critical_value"] > 0


def test_sup_stats_shape(data):
    result = empbs(data, n_boot=100)
    assert len(result["sup_stats"]) == 100


def test_custom_eval_points(data):
    pts = np.linspace(-3, 3, 50)
    result = empbs(data, eval_points=pts)
    np.testing.assert_array_equal(result["eval_points"], pts)
    assert len(result["ecdf"]) == 50


def test_deterministic_with_seed(data):
    r1 = empbs(data, seed=7)
    r2 = empbs(data, seed=7)
    np.testing.assert_array_equal(r1["sup_stats"], r2["sup_stats"])
