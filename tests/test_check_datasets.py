# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.check_datasets — the dataset-availability auditor."""
import morie
from morie.data import DATASET_CATALOG, check_datasets


def test_check_datasets_returns_report():
    res = check_datasets()  # offline-only, no network
    assert res.payload["n_catalog"] == len(DATASET_CATALOG)
    assert res.payload["n_catalog"] > 0
    assert isinstance(res.payload["tiers"], dict)
    # every catalogued dataset is classified into exactly one tier
    assert sum(res.payload["tiers"].values()) == len(DATASET_CATALOG)
    assert "Dataset Availability" in res.title


def test_check_datasets_reachable_count_is_consistent():
    res = check_datasets()
    reachable = res.payload["value"]
    needs = res.payload["needs_attention"]
    assert reachable + needs == len(DATASET_CATALOG)
    assert reachable >= 0 and needs >= 0


def test_check_datasets_top_level_export():
    # reachable via the lazy top-level loader
    assert morie.check_datasets is check_datasets


def test_check_datasets_never_crashes_on_bad_cache(monkeypatch):
    # a misconfigured cache must not crash the auditor
    import morie.data as data
    monkeypatch.setattr(
        data, "cache_load",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))
    res = check_datasets()
    assert res.payload["n_catalog"] == len(DATASET_CATALOG)
