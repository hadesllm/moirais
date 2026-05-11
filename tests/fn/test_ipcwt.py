"""Tests for ipcwt: IPCW weights."""
import numpy as np
import pytest
from morie.fn.ipcwt import ipcwt


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.exponential(2.0, size=n)
    C = rng.exponential(3.0, size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return time, event


def test_returns_keys():
    time, event = _make_data()
    result = ipcwt(time, event)
    for key in ("weights", "km_c_times", "km_c_surv", "tau"):
        assert key in result


def test_weights_shape():
    time, event = _make_data()
    result = ipcwt(time, event)
    assert result["weights"].shape == (len(time),)


def test_censored_weights_zero():
    """Censored subjects must have weight 0."""
    time, event = _make_data()
    result = ipcwt(time, event)
    censored = event == 0
    assert np.all(result["weights"][censored] == 0.0)


def test_event_weights_positive():
    """Event subjects (before tau) should have positive weights."""
    time, event = _make_data()
    result = ipcwt(time, event)
    events_before_tau = (event == 1) & (time <= result["tau"])
    assert np.all(result["weights"][events_before_tau] > 0)


def test_weights_geq_1():
    """IPCW weights >= 1 since KM_C <= 1."""
    time, event = _make_data()
    result = ipcwt(time, event)
    positive_w = result["weights"][result["weights"] > 0]
    assert np.all(positive_w >= 1.0 - 1e-10)


def test_tau_default_is_last_event_time():
    time, event = _make_data()
    result = ipcwt(time, event)
    expected_tau = float(np.max(time[event == 1]))
    assert abs(result["tau"] - expected_tau) < 1e-10


def test_custom_tau():
    time, event = _make_data()
    tau_custom = 1.0
    result = ipcwt(time, event, tau=tau_custom)
    assert result["tau"] == tau_custom
    beyond_tau = time > tau_custom
    assert np.all(result["weights"][beyond_tau] == 0.0)


def test_km_c_surv_decreasing():
    """Censoring survival function is non-increasing."""
    time, event = _make_data(n=300, seed=1)
    result = ipcwt(time, event)
    assert np.all(np.diff(result["km_c_surv"]) <= 1e-12)


def test_invalid_time_raises():
    with pytest.raises(ValueError):
        ipcwt(np.array([-1.0, 2.0, 3.0]), np.array([1, 0, 1]))
