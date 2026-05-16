# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for portable cache-path resolution.

Regression guard: the cache must resolve to a per-user, always-writable
location and must never depend on the install directory or a phantom
``/Volumes/data`` path (the old `_project_root()` parents[5] bug).
"""
from pathlib import Path

from morie.data import _resolve_cache_path, _user_cache_dir, morie_db


def test_user_cache_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _user_cache_dir() == tmp_path / "morie"


def test_user_cache_dir_defaults_to_dot_cache(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert _user_cache_dir() == Path.home() / ".cache" / "morie"


def test_resolve_cache_path_is_portable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("MORIE_CACHE_DB", raising=False)
    p = _resolve_cache_path()
    assert p == tmp_path / "morie" / "morie.db"
    assert "/Volumes/data" not in str(p)


def test_resolve_cache_path_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom" / "c.db"
    monkeypatch.setenv("MORIE_CACHE_DB", str(target))
    assert _resolve_cache_path() == target


def test_resolve_cache_path_relative_env_lands_in_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("MORIE_CACHE_DB", "sub/c.db")
    assert _resolve_cache_path() == tmp_path / "morie" / "sub" / "c.db"


def test_morie_db_never_under_volumes_data():
    assert "/Volumes/data" not in str(morie_db())
