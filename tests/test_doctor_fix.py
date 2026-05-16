# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for `morie doctor` -- the --fix healing mode and version check."""
import inspect

import morie.doctor as doctor


def test_check_morie_version_returns_tuple():
    ok, detail = doctor._check_morie_version()
    assert isinstance(ok, bool)
    assert isinstance(detail, str)


def test_run_checks_includes_morie_version():
    labels = [c["label"] for c in doctor.run_checks()["checks"]]
    assert "morie version" in labels


def test_run_doctor_accepts_fix_kwarg():
    assert "fix" in inspect.signature(doctor.run_doctor).parameters


def test_pip_name_mapping():
    # sklearn imports as `sklearn` but installs as `scikit-learn`
    assert doctor._PIP_NAME["sklearn"] == "scikit-learn"


def test_heal_pip_installs_failed_imports(monkeypatch):
    calls = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(
        doctor.subprocess, "run",
        lambda cmd, *a, **k: (calls.append(cmd), _Result())[1])
    results = {"checks": [
        {"label": "import sklearn", "passed": False, "detail": "x", "required": True},
        {"label": "import numpy", "passed": True, "detail": "1.0", "required": True},
    ]}
    healed = doctor._heal(results)
    assert healed is True
    flat = [" ".join(c) for c in calls]
    # sklearn is reinstalled under its pip name; numpy passed so is not
    assert any("scikit-learn" in f for f in flat)
    assert not any("install numpy" in f for f in flat)
