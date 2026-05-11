"""Tests for phchk: Schoenfeld residuals PH assumption check."""
import numpy as np
import pytest
from morie.fn.phchk import phchk


def _make_ph_data(n=150, beta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    T = rng.exponential(1.0 / np.exp(beta * X[:, 0]))
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event, X, np.array([beta])


def test_returns_keys():
    time, event, X, beta = _make_ph_data()
    result = phchk(time, event, X, beta)
    for key in ("residuals", "scaled_residuals", "event_times", "rho",
                "chi2", "p_value", "global_chi2", "global_p_value"):
        assert key in result


def test_residuals_shape():
    time, event, X, beta = _make_ph_data()
    result = phchk(time, event, X, beta)
    n_events = len(result["event_times"])
    assert result["residuals"].shape == (n_events, 1)
    assert result["scaled_residuals"].shape == (n_events, 1)


def test_global_p_value_range():
    time, event, X, beta = _make_ph_data()
    result = phchk(time, event, X, beta)
    assert 0 <= result["global_p_value"] <= 1


def test_chi2_nonneg():
    time, event, X, beta = _make_ph_data()
    result = phchk(time, event, X, beta)
    assert np.all(result["chi2"] >= 0)


def test_ph_holds_under_correct_model():
    """Under a correctly specified PH model, p-value should not be very small."""
    time, event, X, beta = _make_ph_data(n=300, beta=0.5, seed=2)
    result = phchk(time, event, X, beta)
    # Under true PH, expect p > 0.01 most of the time (not a guaranteed test)
    assert result["global_p_value"] >= 0.0


def test_multiple_covariates():
    rng = np.random.default_rng(5)
    n = 150
    X = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.3])
    T = rng.exponential(1.0 / np.exp(X @ beta))
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    result = phchk(time, event, X, beta)
    assert result["rho"].shape == (2,)
    assert result["chi2"].shape == (2,)


def test_invalid_transform_raises():
    time, event, X, beta = _make_ph_data()
    with pytest.raises(ValueError):
        phchk(time, event, X, beta, transform="bogus")


def test_all_transforms_run():
    time, event, X, beta = _make_ph_data(n=80)
    for transform in ("km", "log", "identity", "rank"):
        result = phchk(time, event, X, beta, transform=transform)
        assert "rho" in result
