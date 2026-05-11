"""Tests for morie.fn.dat — morie.db path resolution."""

from pathlib import Path

from morie.fn.dat import dat, morie_db


def test_alias_is_same_function():
    """dat and morie_db are the same object."""
    assert dat is morie_db


def test_returns_path():
    """dat() returns a Path object."""
    result = dat()
    assert isinstance(result, Path)


def test_path_ends_with_morie_db():
    """The returned path ends with morie.db."""
    result = dat()
    assert result.name == "morie.db"


def test_callable():
    """dat is callable without arguments."""
    # Just ensure no exception
    dat()
