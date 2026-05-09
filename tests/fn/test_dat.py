"""Tests for moirais.fn.dat — moirais.db path resolution."""

from pathlib import Path

from moirais.fn.dat import dat, moirais_db


def test_alias_is_same_function():
    """dat and moirais_db are the same object."""
    assert dat is moirais_db


def test_returns_path():
    """dat() returns a Path object."""
    result = dat()
    assert isinstance(result, Path)


def test_path_ends_with_moirais_db():
    """The returned path ends with moirais.db."""
    result = dat()
    assert result.name == "moirais.db"


def test_callable():
    """dat is callable without arguments."""
    # Just ensure no exception
    dat()
