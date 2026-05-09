"""Tests for moirais.fn.dicon -- DIC."""

import numpy as np
from moirais.fn.dicon import compute_dic


def test_returns_dict():
    result = compute_dic([-10, -11, -12, -9], -10)
    assert isinstance(result, dict)
    assert "dic" in result
    assert "p_d" in result


def test_dic_equals_2dbar_minus_dmean():
    ll_samples = [-10.0, -11.0, -12.0, -9.0, -10.5]
    ll_mean = -10.2
    result = compute_dic(ll_samples, ll_mean)
    expected = 2 * result["d_bar"] - result["d_at_mean"]
    np.testing.assert_allclose(result["dic"], expected, atol=1e-10)


def test_d_bar_is_mean_deviance():
    ll_samples = np.array([-10.0, -11.0])
    result = compute_dic(ll_samples, -10.5)
    expected_d_bar = -2 * np.mean(ll_samples)
    np.testing.assert_allclose(result["d_bar"], expected_d_bar, atol=1e-10)


def test_d_at_mean_value():
    result = compute_dic([-10, -11], -10.5)
    np.testing.assert_allclose(result["d_at_mean"], 21.0, atol=1e-10)


def test_too_few_samples():
    try:
        compute_dic([-10], -10)
        assert False
    except ValueError:
        pass
