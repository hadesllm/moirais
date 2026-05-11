"""Tests for environmental-justice exposure-disparity index."""

import pandas as pd
import pytest

from morie.fn.envjust import envjust, environmental_justice_index


def test_envjust_equal_exposure_gives_ratio_one():
    df = pd.DataFrame({
        "pm25": [12, 12, 12, 12],
        "group": ["a", "a", "b", "b"],
    })
    r = envjust(df, exposure="pm25", group="group")
    assert r.extra["disparity"]["a"] == pytest.approx(1.0)
    assert r.extra["disparity"]["b"] == pytest.approx(1.0)
    assert r.value == pytest.approx(1.0)


def test_envjust_detects_disparity():
    # Group 'minority' exposed to more pollution
    df = pd.DataFrame({
        "pm25": [10, 10, 10, 20, 20, 20],
        "group": ["maj"] * 3 + ["min"] * 3,
    })
    r = envjust(df, exposure="pm25", group="group")
    # Overall mean = 15; min mean = 20 → ratio 20/15 ≈ 1.333
    assert r.extra["disparity"]["min"] == pytest.approx(20 / 15)
    assert r.extra["disparity"]["maj"] == pytest.approx(10 / 15)
    assert r.value == pytest.approx(20 / 15)


def test_envjust_with_reference_group():
    df = pd.DataFrame({
        "pm25": [10, 10, 20, 20, 30, 30],
        "group": ["white"] * 2 + ["hispanic"] * 2 + ["black"] * 2,
    })
    r = envjust(df, exposure="pm25", group="group", reference="white")
    assert r.extra["disparity"]["white"] == pytest.approx(1.0)
    assert r.extra["disparity"]["hispanic"] == pytest.approx(2.0)
    assert r.extra["disparity"]["black"] == pytest.approx(3.0)
    assert r.extra["reference"] == "white"


def test_envjust_weighted():
    # Population-weighted: 'a' has 100 people, 'b' has 10
    # Unweighted mean = 15; weighted mean = (10*100 + 20*10)/110 ≈ 10.9
    df = pd.DataFrame({
        "pm25": [10, 20],
        "group": ["a", "b"],
        "pop": [100, 10],
    })
    r = envjust(df, exposure="pm25", group="group", weights="pop")
    expected_ref = (10 * 100 + 20 * 10) / 110
    assert r.extra["reference_mean"] == pytest.approx(expected_ref)


def test_envjust_missing_column_raises():
    df = pd.DataFrame({"pm25": [1, 2], "group": ["a", "b"]})
    with pytest.raises(ValueError, match="Missing columns"):
        envjust(df, exposure="no_such_col", group="group")


def test_envjust_missing_reference_raises():
    df = pd.DataFrame({"pm25": [1, 2], "group": ["a", "b"]})
    with pytest.raises(ValueError, match="reference group"):
        envjust(df, exposure="pm25", group="group", reference="nonexistent")


def test_envjust_n_by_group_counts_rows():
    df = pd.DataFrame({
        "pm25": [1, 2, 3, 4, 5, 6, 7],
        "group": ["a", "a", "a", "b", "b", "c", "c"],
    })
    r = envjust(df, exposure="pm25", group="group")
    assert r.extra["n_by_group"] == {"a": 3, "b": 2, "c": 2}


def test_envjust_alias_matches():
    assert envjust is environmental_justice_index
