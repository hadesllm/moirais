"""Test pistat (Pi 5 system status)."""
from unittest.mock import patch, MagicMock
import pytest


def test_pi_status_returns_result():
    from moirais.fn.pistat import pistat
    r = pistat()
    assert r.name == "pi_status"
    assert isinstance(r.value, dict)
    assert "cpu_temp_c" in r.value
    assert "ram" in r.value
    assert "disk" in r.value
    assert "ollama_models" in r.value
    assert "hailo" in r.value


def test_pi_status_has_health():
    from moirais.fn.pistat import pistat
    r = pistat()
    assert "healthy" in r.extra
    assert "warnings" in r.extra
    assert isinstance(r.extra["warnings"], list)


def test_cheatsheet():
    from moirais.fn.pistat import cheatsheet
    cs = cheatsheet()
    assert "pi_status" in cs
    assert isinstance(cs, str)


def test_sensors_format_status():
    from moirais.sensors import pi_status as _ps, format_status
    status = _ps()
    report = format_status(status)
    assert "MOIRAIS Pi 5" in report
    assert "Timestamp" in report
    assert isinstance(report, str)


def test_health_check_structure():
    from moirais.sensors import pi_health_check
    h = pi_health_check()
    assert "healthy" in h
    assert "warnings" in h
    assert isinstance(h["healthy"], bool)
    assert isinstance(h["warnings"], list)


def test_thermal_history_short():
    from moirais.sensors import pi_thermal_history
    samples = pi_thermal_history(seconds=3)
    assert isinstance(samples, list)
    assert len(samples) >= 1
    ts, temp = samples[0]
    assert isinstance(ts, str)


def test_health_check_warns_on_high_temp():
    with patch("moirais.sensors._cpu_temp", return_value=85.0):
        from moirais.sensors import pi_health_check
        h = pi_health_check()
        assert any("temperature" in w.lower() for w in h["warnings"])


def test_health_check_warns_ollama_down():
    with patch("moirais.sensors._ollama_models", return_value=None):
        from moirais.sensors import pi_health_check
        h = pi_health_check()
        assert any("ollama" in w.lower() for w in h["warnings"])
