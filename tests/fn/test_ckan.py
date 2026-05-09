"""Tests for moirais.fn.ckan — CKAN fetch wrapper."""

import pytest

from moirais.fn.ckan import ckan, fetch_ckan_to_cache


def test_alias_is_same_function():
    """ckan and fetch_ckan_to_cache are the same object."""
    assert ckan is fetch_ckan_to_cache


def test_callable():
    """ckan is a callable."""
    assert callable(ckan)


def test_rejects_unknown_dataset():
    """Raises ValueError for unknown CKAN dataset key."""
    with pytest.raises(ValueError, match="Unknown CKAN dataset"):
        ckan("totally_fake_dataset_xyz")
