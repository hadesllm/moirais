"""Tests for morie.fn.evtcl — event clustering."""
import numpy as np
import pytest

from morie.fn.evtcl import event_cluster, evtcl


def test_basic_clustering():
    rng = np.random.default_rng(42)
    events = [rng.standard_normal(10) for _ in range(9)]
    result = event_cluster(events, n_clusters=3)
    assert result.extra["n_clusters"] == 3
    assert len(result.extra["labels"]) == 9


def test_single_cluster():
    events = [np.ones(5) for _ in range(4)]
    result = event_cluster(events, n_clusters=1)
    assert np.all(result.extra["labels"] == 0)


def test_uneven_lengths():
    events = [np.ones(3), np.ones(5), np.ones(7)]
    result = event_cluster(events, n_clusters=2)
    assert result.extra["centroids"].shape[1] == 7


def test_empty_raises():
    with pytest.raises(ValueError):
        event_cluster([], n_clusters=2)


def test_alias():
    assert evtcl is event_cluster
