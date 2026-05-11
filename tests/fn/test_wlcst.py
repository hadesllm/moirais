"""Tests for wlcst: Wilcoxon (Breslow) survival test."""
import numpy as np
import pytest
from morie.fn.wlcst import wlcst


def _make_two_group(n_per_group=100, hr=2.0, seed=0):
    rng = np.random.default_rng(seed)
    T0 = rng.exponential(2.0, size=n_per_group)
    T1 = rng.exponential(2.0 / hr, size=n_per_group)
    C = rng.exponential(5.0, size=2 * n_per_group)
    time = np.concatenate([T0, T1])
    event = (time <= C).astype(float)
    time = np.minimum(time, C)
    group = np.array([0] * n_per_group + [1] * n_per_group)
    return time, event, group


def test_returns_keys():
    time, event, group = _make_two_group()
    result = wlcst(time, event, group)
    for key in ("statistic", "p_value", "z_score", "observed", "expected"):
        assert key in result


def test_statistic_nonneg():
    time, event, group = _make_two_group()
    result = wlcst(time, event, group)
    assert result["statistic"] >= 0


def test_significant_with_large_hr():
    time, event, group = _make_two_group(n_per_group=200, hr=3.0, seed=1)
    result = wlcst(time, event, group)
    assert result["p_value"] < 0.05


def test_p_value_range():
    time, event, group = _make_two_group()
    result = wlcst(time, event, group)
    assert 0.0 <= result["p_value"] <= 1.0


def test_observed_length():
    time, event, group = _make_two_group()
    result = wlcst(time, event, group)
    assert len(result["observed"]) == 2
    assert len(result["expected"]) == 2


def test_peto_weight():
    time, event, group = _make_two_group(hr=2.0)
    result = wlcst(time, event, group, weight="peto")
    assert result["statistic"] >= 0


def test_more_than_two_groups_raises():
    time, event, group = _make_two_group()
    group[0] = 2  # introduce third group
    with pytest.raises(ValueError):
        wlcst(time, event, group)


def test_invalid_weight_raises():
    time, event, group = _make_two_group()
    with pytest.raises(ValueError):
        wlcst(time, event, group, weight="bogus")
