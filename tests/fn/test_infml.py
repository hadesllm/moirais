"""Tests for morie.fn.infml — infer measurement level (NOIR)."""

import pandas as pd
import pytest

from morie.fn.infml import infml, infer_measurement_level
from morie.dataset import MeasurementLevel


def test_alias_is_same_function():
    """infml and infer_measurement_level are the same object."""
    assert infml is infer_measurement_level


def test_nominal_string_column():
    """String columns should be classified as NOMINAL."""
    s = pd.Series(["cat", "dog", "bird", "cat"], name="species")
    assert infml(s) == MeasurementLevel.NOMINAL


def test_ordinal_with_name_hint():
    """Integer column with ordinal-hinting name should be ORDINAL."""
    s = pd.Series([1, 2, 3, 4, 5], name="likert_q1")
    assert infml(s) == MeasurementLevel.ORDINAL


def test_nominal_binary():
    """Binary 0/1 column should be NOMINAL."""
    s = pd.Series([0, 1, 0, 1, 1], name="treatment")
    assert infml(s) == MeasurementLevel.NOMINAL


def test_ratio_positive_integers():
    """Positive integer column should be RATIO."""
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], name="count")
    assert infml(s) == MeasurementLevel.RATIO


def test_interval_negative_integers():
    """Integer column with negatives should be INTERVAL."""
    s = pd.Series([-5, -3, 0, 2, 4, 6, 8, 10, 12, 14], name="temperature_delta")
    assert infml(s) == MeasurementLevel.INTERVAL


def test_ratio_float():
    """Float column without interval-hinting name should be RATIO."""
    s = pd.Series([1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.0, 8.8, 9.1, 10.4], name="weight_kg")
    assert infml(s) == MeasurementLevel.RATIO


def test_interval_float_with_hint():
    """Float column with interval-hinting name (year) should be INTERVAL."""
    s = pd.Series([2020.0, 2021.0, 2022.0, 2023.0, 2024.0,
                    2020.5, 2021.5, 2022.5, 2023.5, 2024.5], name="year")
    assert infml(s) == MeasurementLevel.INTERVAL


def test_boolean_is_nominal():
    """Boolean column should be NOMINAL."""
    s = pd.Series([True, False, True, False], name="flag")
    assert infml(s) == MeasurementLevel.NOMINAL
