"""Tests for moirais.fn.cconn — SQLite cache connect."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from moirais.fn.cconn import cconn, cache_connect


def test_alias_is_same_function():
    """cconn and cache_connect are the same object."""
    assert cconn is cache_connect


def test_returns_connection(tmp_path):
    """cache_connect returns a sqlite3.Connection."""
    db = tmp_path / "test_cache.db"
    conn = cconn(db)
    try:
        assert isinstance(conn, sqlite3.Connection)
    finally:
        conn.close()


def test_creates_metadata_table(tmp_path):
    """cache_connect creates the _moirais_metadata table."""
    db = tmp_path / "test_cache.db"
    conn = cconn(db)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_moirais_metadata'"
        ).fetchone()
        assert tables is not None
    finally:
        conn.close()


def test_wal_mode(tmp_path):
    """cache_connect sets WAL journal mode."""
    db = tmp_path / "test_cache.db"
    conn = cconn(db)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
    finally:
        conn.close()
