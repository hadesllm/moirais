# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for missing-dataset recommendations -- morie.dataset_recommendation
and the recommendations surfaced by check_datasets()."""
from morie.data import DATASET_CATALOG, check_datasets, dataset_recommendation


def test_recommendation_ckan_dataset():
    rec = dataset_recommendation("occ22")  # has a CKAN resource id
    assert "occ22" in rec
    assert "open.canada.ca" in rec
    assert DATASET_CATALOG["occ22"]["ckan_resource_id"] in rec


def test_recommendation_fetcher_dataset():
    rec = dataset_recommendation("cchs22")  # fetched on demand
    assert "cchs22" in rec
    assert "fetched on demand" in rec.lower()


def test_recommendation_local_only_dataset():
    rec = dataset_recommendation("hibp")  # no public remote source
    assert "no public remote source" in rec
    assert DATASET_CATALOG["hibp"]["local_path"] in rec


def test_recommendation_unknown_key():
    rec = dataset_recommendation("definitely-not-a-dataset")
    assert "not in the morie catalogue" in rec


def test_check_datasets_payload_exposes_recommendations():
    res = check_datasets()
    p = res.payload
    assert isinstance(p["missing"], list)
    assert isinstance(p["recommendations"], dict)
    # every missing dataset has a recommendation, and vice versa
    assert set(p["missing"]) == set(p["recommendations"])
    # with no remote probe, the missing list is exactly the attention set
    assert len(p["missing"]) == p["needs_attention"]
