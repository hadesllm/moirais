"""Tests for cminc: cumulative incidence function (Aalen-Johansen)."""
import numpy as np
import pytest
from moirais.fn.cminc import cminc


def _sim_competing_risks(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # Cause 1: exponential rate 0.3; Cause 2: exponential rate 0.2
    T1 = rng.exponential(1 / 0.3, size=n)
    T2 = rng.exponential(1 / 0.2, size=n)
    C = rng.exponential(5.0, size=n)
    time = np.minimum(np.minimum(T1, T2), C)
    event = np.where(T1 < T2, 1, 2)
    event = np.where(C < np.minimum(T1, T2), 0, event)
    return time, event


def test_returns_keys():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=1)
    for key in ("time_points", "cif", "overall_survival", "se", "ci_lower", "ci_upper"):
        assert key in result


def test_cif_monotone_increasing():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=1)
    cif = result["cif"]
    assert np.all(np.diff(cif) >= -1e-12)


def test_cif_between_0_and_1():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=1)
    assert np.all(result["cif"] >= 0)
    assert np.all(result["cif"] <= 1)


def test_ci_bounds_valid():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=1)
    assert np.all(result["ci_lower"] <= result["cif"] + 1e-12)
    assert np.all(result["ci_upper"] >= result["cif"] - 1e-12)


def test_cause2_cif():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=2)
    assert "cif" in result
    assert np.all(result["cif"] >= 0)


def test_cause_not_in_events_raises():
    time, event = _sim_competing_risks()
    with pytest.raises(ValueError, match="Cause"):
        cminc(time, event, cause=99)


def test_cif_cause1_plus_cause2_leq_1():
    """CIF for cause 1 + CIF for cause 2 <= 1 at last time point."""
    time, event = _sim_competing_risks(n=300, seed=1)
    r1 = cminc(time, event, cause=1)
    r2 = cminc(time, event, cause=2)
    # Both CIFs evaluated at same time points (may differ; check common last)
    assert r1["cif"][-1] + r2["cif"][-1] <= 1.0 + 1e-10


def test_time_points_sorted():
    time, event = _sim_competing_risks()
    result = cminc(time, event, cause=1)
    assert np.all(np.diff(result["time_points"]) > 0)
