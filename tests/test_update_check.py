# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie._update_check -- the new-version notification and the
`morie update` command. No live network calls are made."""
import json
import time

import morie._update_check as uc


def test_parse_version_numeric():
    assert uc._parse_version("0.9.0") == (0, 9, 0)
    assert uc._parse_version("1.2.3") == (1, 2, 3)
    assert uc._parse_version("0.10.0") == (0, 10, 0)


def test_parse_version_handles_suffixes():
    # a non-numeric character ends that chunk's parse
    assert uc._parse_version("0.9.0a1") == (0, 9, 0)
    assert uc._parse_version("0.9.0+local") == (0, 9, 0)
    assert uc._parse_version("0.0.0+unknown") == (0, 0, 0)
    assert uc._parse_version("garbage") == (0,)


def test_version_ordering():
    assert uc._parse_version("0.9.0") > uc._parse_version("0.8.0")
    assert uc._parse_version("0.10.0") > uc._parse_version("0.9.0")
    assert not uc._parse_version("0.8.0") > uc._parse_version("0.8.0")


def _fake_cache(tmp_path, monkeypatch, latest):
    cache = tmp_path / "update_check.json"
    cache.write_text(json.dumps({"latest": latest, "last_check": time.time()}))
    monkeypatch.setattr(uc, "_cache_path", lambda: str(cache))
    monkeypatch.delenv("MORIE_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "_NOTIFIED", False)


def test_maybe_notify_warns_when_outdated(tmp_path, monkeypatch, capsys):
    _fake_cache(tmp_path, monkeypatch, "99.9.9")
    uc.maybe_notify("0.9.0")
    err = capsys.readouterr().err
    assert "99.9.9" in err
    assert "morie update" in err


def test_maybe_notify_silent_when_current(tmp_path, monkeypatch, capsys):
    _fake_cache(tmp_path, monkeypatch, "0.9.0")
    uc.maybe_notify("0.9.0")
    assert "newer version" not in capsys.readouterr().err


def test_maybe_notify_silent_for_dev_install(tmp_path, monkeypatch, capsys):
    _fake_cache(tmp_path, monkeypatch, "99.9.9")
    uc.maybe_notify("0.0.0+unknown")
    assert capsys.readouterr().err == ""


def test_maybe_notify_respects_optout(tmp_path, monkeypatch, capsys):
    _fake_cache(tmp_path, monkeypatch, "99.9.9")
    monkeypatch.setenv("MORIE_NO_UPDATE_CHECK", "1")
    uc.maybe_notify("0.9.0")
    assert capsys.readouterr().err == ""


def test_check_pypi_latest_fails_soft(monkeypatch):
    # an unreachable endpoint must yield None, never raise
    monkeypatch.setattr(uc, "PYPI_JSON_URL", "https://127.0.0.1:9/nope")
    assert uc.check_pypi_latest(timeout=0.5) is None


def test_update_entry_points_exist():
    # the `morie update` command resolves to a callable
    assert callable(uc.run_update)
    import morie.runner as runner
    assert callable(runner.main)
