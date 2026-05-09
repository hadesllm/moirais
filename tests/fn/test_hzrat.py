"""Tests for hzrat: hazard ratio with CI."""
import numpy as np
import pytest
from moirais.fn.hzrat import hzrat


def _make_two_group(n_per_group=150, hr=2.0, seed=0):
    rng = np.random.default_rng(seed)
    rate_0 = 0.5
    rate_1 = rate_0 * hr
    T0 = rng.exponential(1 / rate_0, size=n_per_group)
    T1 = rng.exponential(1 / rate_1, size=n_per_group)
    C = rng.exponential(4.0, size=2 * n_per_group)
    T = np.concatenate([T0, T1])
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    group = np.array([0] * n_per_group + [1] * n_per_group)
    return time, event, group


def test_returns_keys_two_group():
    time, event, group = _make_two_group()
    result = hzrat(time, event, group)
    for key in ("hazard_ratio", "ci_lower", "ci_upper", "p_value", "beta", "se"):
        assert key in result


def test_hr_close_to_truth():
    """HR estimate should be near the true HR=2.0."""
    time, event, group = _make_two_group(n_per_group=400, hr=2.0, seed=1)
    result = hzrat(time, event, group)
    assert abs(result["hazard_ratio"] - 2.0) < 0.5


def test_hr_positive():
    time, event, group = _make_two_group()
    result = hzrat(time, event, group)
    assert result["hazard_ratio"] > 0


def test_ci_brackets_hr():
    time, event, group = _make_two_group()
    result = hzrat(time, event, group)
    assert result["ci_lower"] <= result["hazard_ratio"] <= result["ci_upper"]


def test_p_value_significant():
    """HR=3 with n=400 should be significant."""
    time, event, group = _make_two_group(n_per_group=400, hr=3.0, seed=2)
    result = hzrat(time, event, group)
    assert result["p_value"] < 0.05


def test_hr_null_model():
    """HR=1 (no effect) should give HR estimate near 1."""
    time, event, group = _make_two_group(n_per_group=300, hr=1.0, seed=42)
    result = hzrat(time, event, group)
    # HR near 1; exact value depends on random draw
    assert 0.5 < result["hazard_ratio"] < 2.0


def test_reference_group_selection():
    time, event, group = _make_two_group(hr=2.0, seed=3)
    r0 = hzrat(time, event, group, reference=0)
    r1 = hzrat(time, event, group, reference=1)
    # Switching reference should invert HR
    assert abs(r0["hazard_ratio"] * r1["hazard_ratio"] - 1.0) < 0.1


def test_invalid_reference_raises():
    time, event, group = _make_two_group()
    with pytest.raises(ValueError):
        hzrat(time, event, group, reference=99)


def test_fewer_than_two_groups_raises():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    group = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        hzrat(time, event, group)
