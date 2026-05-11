"""Test sync_avg (sncav)."""
import numpy as np
import pytest

from morie.fn.sncav import sync_avg, sncav
from morie.fn._containers import SignalResult


class TestSyncAvg:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = sync_avg(x, trigger_indices=np.array([50, 100, 150]))
        assert isinstance(result, SignalResult)
        assert result.name == "sync_avg"

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = sync_avg(x, trigger_indices=np.array([50, 100, 150]))
        assert result.filtered is not None

    def test_window_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = sync_avg(x, trigger_indices=np.array([50, 100]), window=50)
        assert isinstance(result, SignalResult)

    def test_output_shape_matches_window(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = sync_avg(x, trigger_indices=np.array([50, 100, 150]), window=40)
        assert result.n_samples == 40

    def test_alias(self):
        assert sncav is sync_avg
