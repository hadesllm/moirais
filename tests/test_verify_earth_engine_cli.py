"""Tests for ``moirais verify-earth-engine`` CLI smoke-check."""

import json
import os
import subprocess
import sys

import pytest

# The Earth-Engine CLI tests exercise behavior when earthengine-api is
# installed vs. absent. They're dual-purpose: if EE is installed we
# verify the library stage passes; if absent we verify the failure
# message is actionable. Pin to the "installed" path by requiring the
# package. (CI installs earthengine-api as part of [test].)
pytest.importorskip("ee")


def _run_cli(*extra_args: str, env_overrides: dict | None = None):
    """Invoke the installed entrypoint as a subprocess."""
    env = os.environ.copy()
    # Purge any inherited EE env so tests are deterministic
    for k in (
        "MOIRAIS_EE_SERVICE_ACCOUNT",
        "MOIRAIS_EE_KEY_PATH",
        "MOIRAIS_EE_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
    ):
        env.pop(k, None)
    # Block runner's auto .env loader — otherwise a dev .env
    # with real Vertex/EE keys re-populates what we just purged.
    env["MOIRAIS_SKIP_DOTENV"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "moirais.runner", "verify-earth-engine", *extra_args],
        capture_output=True, text=True, env=env,
    )


def test_verify_ee_no_credentials_exits_1():
    r = _run_cli()
    assert r.returncode == 1
    assert "credentials" in r.stdout
    assert "FAIL" in r.stdout
    # Should point at the howto for next steps
    assert "earth_engine_auth.md" in r.stdout


def test_verify_ee_library_stage_passes_when_installed():
    # earthengine-api is in the venv; first stage should pass.
    r = _run_cli()
    assert "[PASS] library" in r.stdout
    assert "earthengine-api" in r.stdout


def test_verify_ee_bad_keypath_exit_1():
    r = _run_cli(env_overrides={
        "MOIRAIS_EE_SERVICE_ACCOUNT": "fake@example.iam.gserviceaccount.com",
        "MOIRAIS_EE_KEY_PATH": "/no/such/file.json",
        "MOIRAIS_EE_PROJECT": "test-project",
    })
    assert r.returncode == 1
    assert "points at missing file" in r.stdout


def test_verify_ee_json_mode_structured():
    r = _run_cli("--json")
    assert r.returncode == 1
    payload = json.loads(r.stdout)
    assert "stages" in payload
    stages = payload["stages"]
    # Library should pass; credentials should fail
    lib_stage = next(s for s in stages if s["stage"] == "library")
    cred_stage = next(s for s in stages if s["stage"] == "credentials")
    assert lib_stage["ok"] is True
    assert cred_stage["ok"] is False


def test_verify_ee_skip_query_passes_when_initialize_would():
    # With no creds, --skip-query still exits 1 (credentials fail first).
    # But the *flag* should be accepted by argparse.
    r = _run_cli("--skip-query")
    # argparse accepted the flag; exit is 1 because credentials still fail
    assert r.returncode == 1
    assert "invalid" not in r.stderr.lower()


def test_verify_ee_appears_in_help():
    r = subprocess.run(
        [sys.executable, "-m", "moirais.runner", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "verify-earth-engine" in r.stdout
