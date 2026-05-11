from pathlib import Path

import pytest

from morie.runner import build_parser, execute_pipeline, main

_RTESTS_DIR = Path(__file__).resolve().parents[1] / "rtests"


def test_execute_pipeline_requires_confirmation_by_default(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "n")

    status = execute_pipeline()

    captured = capsys.readouterr()
    assert status == 1
    assert "Selected modules:" in captured.out
    assert "Pipeline aborted." in captured.out


def test_execute_pipeline_runs_when_confirmed(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr("morie.runner.run_module", lambda *_, **__: {})

    status = execute_pipeline(modules=["power-design"], silent=False)

    captured = capsys.readouterr()
    assert status == 0
    assert "Selected modules: power-design" in captured.out
    assert "Pipeline completed successfully." in captured.out
    assert "Completed modules: power-design" in captured.out


def test_main_prints_help_when_no_action_requested(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["morie"])

    status = main()

    captured = capsys.readouterr()
    assert status == 0
    assert "MORIE" in captured.out


def test_parser_accepts_module_overrides():
    parser = build_parser()
    args = parser.parse_args(["pipeline", "--all", "--modules", "power-design", "propensity-scores", "-y"])

    assert args.all is True
    assert args.modules == ["power-design", "propensity-scores"]
    assert args.yes is True


def test_parser_accepts_run_module_arguments():
    parser = build_parser()
    args = parser.parse_args(
        ["run-module", "power-design", "--cpads-csv", "docs/source/datasets/cpads-2021-2022-pumf2.csv"]
    )

    assert args.module == "power-design"
    assert args.cpads_csv.endswith("cpads-2021-2022-pumf2.csv")


def test_main_runs_parity_review(monkeypatch, capsys, tmp_path):
    csv_path = tmp_path / "parity.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "morie",
            "parity-review",
            "--epiml-root",
            str(_RTESTS_DIR),
            "--output",
            str(csv_path),
        ],
    )

    status = main()

    captured = capsys.readouterr()
    assert status == 0
    assert "Audit summary:" in captured.out or "Parity summary:" in captured.out
    assert csv_path.exists()


def test_main_lists_modules(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["morie", "list-modules"])
    status = main()
    captured = capsys.readouterr()
    assert status == 0
    assert "power-design" in captured.out
