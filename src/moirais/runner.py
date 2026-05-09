"""CLI entrypoints for MOIRAIS package workflows and assistant mode.

Provides the ``moirais`` command with subcommands for running analysis modules,
querying the LLM agent, and checking environment health.
"""

from __future__ import annotations

import argparse
import sys

from .modules import DEFAULT_CPADS_CSV, list_modules, run_module
from .perseus import ask_percy

# ---------------------------------------------------------------------------
# Emissions tracking — use vendored moirais.emissions, fall back to codecarbon
# ---------------------------------------------------------------------------

try:
    from moirais.emissions import EmissionsTracker as _EmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        from codecarbon import EmissionsTracker as _EmissionsTracker

        _CODECARBON_AVAILABLE = True
    except ImportError:
        _EmissionsTracker = None
        _CODECARBON_AVAILABLE = False


def execute_pipeline(
    modules: list[str] | None = None,
    *,
    cpads_csv: str = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | None = None,
    silent: bool = False,
    track_carbon: bool = True,
) -> int:
    """
    Run the specified epidemiologic analysis modules.

    :param modules: A list of module names to execute, defaults to None.
    :type modules: list[str], optional
    :param silent: If True, skips the safety confirmation prompt, defaults to False.
    :type silent: bool, optional
    :param track_carbon: If True and codecarbon is installed, track CO₂ emissions.
    :type track_carbon: bool, optional
    """
    selected = modules or [item["name"] for item in list_modules()]

    if not silent:
        print("Selected modules:", ", ".join(selected))
        confirm = input(f"This will run the MOIRAIS module pipeline for: {', '.join(selected)}. Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Pipeline aborted.")
            return 1
    else:
        print("Selected modules:", ", ".join(selected))

    # Start pipeline-level emissions tracker
    tracker = None
    if track_carbon and _CODECARBON_AVAILABLE:
        import os

        emissions_dir = os.path.join(output_dir, "emissions") if output_dir else "data/manifest/outputs/emissions"
        os.makedirs(emissions_dir, exist_ok=True)
        try:
            tracker = _EmissionsTracker(
                project_name="moirais-pipeline",
                output_dir=emissions_dir,
                log_level="error",
            )
            tracker.start()
        except Exception:  # pragma: no cover
            tracker = None

    results = {}
    total = len(selected)

    # Try enlighten progress bars for rich terminal output.
    _manager = None
    _pbar = None
    try:
        import enlighten

        _manager = enlighten.get_manager()
        _pbar = _manager.counter(total=total, desc="Pipeline", unit="modules", color="green")
    except ImportError:
        pass

    for idx, module_name in enumerate(selected, start=1):
        if _pbar:
            _pbar.desc = f"Running: {module_name}"
        else:
            print(f"[{idx}/{total}] Running: {module_name}", flush=True)
        try:
            results[module_name] = run_module(
                module_name,
                cpads_csv=cpads_csv,
                dataset_key=dataset_key,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"  ERROR in {module_name}: {exc}")
        if _pbar:
            _pbar.update()
        else:
            print(f"[{idx}/{total}] Done: {module_name}", flush=True)

    if _manager:
        _manager.stop()

    if tracker is not None:
        try:
            emissions = tracker.stop()
            if emissions is not None:
                print(f"Pipeline CO₂ emissions: {emissions:.6f} kg CO₂eq")
        except Exception:  # pragma: no cover
            pass

    print("Pipeline completed successfully.")
    print("Completed modules:", ", ".join(results.keys()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="MOIRAIS package runner")
    subparsers = parser.add_subparsers(dest="command")

    pipeline = subparsers.add_parser("pipeline", help="Run the MOIRAIS module pipeline")
    pipeline.add_argument("--all", action="store_true", help="Run the current MOIRAIS module surface")
    pipeline.add_argument("--modules", nargs="+", help="Override the default module list")
    pipeline.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    pipeline.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    pipeline.add_argument("--output-dir", help="Optional directory for CSV outputs")
    pipeline.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    pipeline.add_argument(
        "--no-carbon",
        action="store_true",
        default=False,
        help="Disable CodeCarbon emissions tracking for this run",
    )

    parity = subparsers.add_parser(
        "parity-review",
        help=argparse.SUPPRESS,
        description="Internal compatibility audit for maintainers.",
    )
    parity.add_argument("--epiml-root", required=True, help="Path to the internal reference checkout")
    parity.add_argument("--output", help="Optional CSV path for the audit matrix")

    subparsers.add_parser("list-modules", help="List implemented MOIRAIS CPADS modules")

    run_cmd = subparsers.add_parser("run-module", help="Run one MOIRAIS module")
    run_cmd.add_argument("module", help="Module name to run")
    run_cmd.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    run_cmd.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    run_cmd.add_argument("--output-dir", help="Optional directory for CSV outputs")

    run_all = subparsers.add_parser("run-modules", help="Run multiple MOIRAIS modules")
    run_all.add_argument("--modules", nargs="+", help="Module names to run; defaults to all implemented modules")
    run_all.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    run_all.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    run_all.add_argument("--output-dir", help="Optional directory for CSV outputs")

    agent_cmd = subparsers.add_parser("agent", help="Ask Perseus, the MOIRAIS agent")
    agent_cmd.add_argument("question", help="Question to answer")
    agent_cmd.add_argument("--context", help="Optional context string")
    agent_cmd.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming (default: stream to terminal)",
    )

    ask_cmd = subparsers.add_parser(
        "ask",
        help="Ask the MOIRAIS agent (streaming by default)",
    )
    ask_cmd.add_argument("question", help="Question to answer")
    ask_cmd.add_argument("--context", help="Optional context string")
    ask_cmd.add_argument("--model", default=None, help="Override the LLM model name")
    ask_cmd.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming and print the full response at once",
    )

    # ── chat ─────────────────────────────────────────────────────────────
    chat_cmd = subparsers.add_parser(
        "chat",
        help="Launch interactive chat REPL (streaming LLM, slash commands)",
    )
    chat_cmd.add_argument(
        "--agent",
        default=None,
        help="Load a specific agent persona (e.g., moirais-architect)",
    )

    percy_talk = subparsers.add_parser(
        "percy",
        aliases=["perseus"],
        help="Talk to Perseus — MOIRAIS's expert AI (auto-detects best connection)",
    )
    percy_talk.add_argument("question", nargs="?", default=None, help="Question (omit for interactive mode)")
    percy_talk.add_argument("--model", default=None, help="Override model name")
    percy_talk.add_argument("--remote", action="store_true", help="Use Perseus cloud (MOIRAIS-hosted, no local setup)")
    percy_talk.add_argument("--local", action="store_true", help="Force local Ollama only (no network)")
    percy_talk.add_argument("--freeapi", action="store_true", help="Use free community LLM servers (fallback)")
    percy_talk.add_argument("--pi", default=None, help="Connect to Pi (e.g. --pi host or MOIRAIS_PI_HOST)")
    percy_talk.add_argument("--cloud", default=None, help="Custom Perseus relay URL (e.g. https://your-relay:8421)")
    percy_talk.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    relay_cmd = subparsers.add_parser(
        "serve",
        help="Start Perseus relay server (expose Perseus over the network)",
    )
    relay_cmd.add_argument("--port", type=int, default=8421, help="Port to listen on (default: 8421)")
    relay_cmd.add_argument("--token", default=None, help="Require auth token for API access")
    relay_cmd.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")

    subparsers.add_parser(
        "doctor",
        help="Run MOIRAIS environment diagnostics",
    )

    # ── profile-dataset ──────────────────────────────────────────────────
    profile_cmd = subparsers.add_parser(
        "profile-dataset",
        help="Profile a dataset: infer variable types, roles, and suggest analyses",
    )
    profile_cmd.add_argument("--csv", required=True, help="Path to the dataset file (CSV, TSV, Excel, Parquet, JSON)")
    profile_cmd.add_argument("--treatment", default=None, help="Hint: column to use as treatment")
    profile_cmd.add_argument("--outcome", default=None, help="Hint: column to use as outcome")
    profile_cmd.add_argument("--weights", default=None, help="Hint: column to use as survey weight")
    profile_cmd.add_argument("--suggest", action="store_true", help="Also print a suggested analysis plan")

    # ── sample ───────────────────────────────────────────────────────────
    sample_cmd = subparsers.add_parser(
        "sample",
        help="Draw a sample from a dataset (SRS, stratified, cluster, PPS)",
    )
    sample_cmd.add_argument("--csv", required=True, help="Path to the input dataset")
    sample_cmd.add_argument(
        "--method",
        choices=["srs", "stratified", "cluster", "pps"],
        default="srs",
        help="Sampling method (default: srs)",
    )
    sample_cmd.add_argument("--n", type=int, required=True, help="Sample size (or total for proportional stratified)")
    sample_cmd.add_argument("--strata-col", default=None, help="Column for stratified sampling")
    sample_cmd.add_argument("--cluster-col", default=None, help="Column for cluster sampling")
    sample_cmd.add_argument("--size-col", default=None, help="Column for PPS sampling")
    sample_cmd.add_argument("--proportional", action="store_true", help="Use proportional allocation (stratified only)")
    sample_cmd.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    sample_cmd.add_argument("--output", default=None, help="Output CSV path for the sample")

    # ── inspect ──────────────────────────────────────────────────────────
    inspect_cmd = subparsers.add_parser(
        "inspect",
        help="Browse and summarise output CSVs (schema, rows, stats)",
    )
    inspect_cmd.add_argument("path", help="Path to a CSV file or output directory")
    inspect_cmd.add_argument(
        "--module",
        default=None,
        help="Scope inspection to expected outputs of this module",
    )

    # ── verify ───────────────────────────────────────────────────────────
    verify_cmd = subparsers.add_parser(
        "verify",
        help="Validate statistical outputs for correctness",
    )
    verify_cmd.add_argument("path", help="Path to a CSV file or output directory")
    verify_cmd.add_argument(
        "--module",
        default=None,
        help="Scope verification to expected outputs of this module",
    )

    # ── tui ──────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "tui",
        help="Launch the full-screen terminal IDE (requires textual)",
    )

    # ── selftest ─────────────────────────────────────────────────────────
    subparsers.add_parser(
        "selftest",
        help="Run a quick smoke test of all MOIRAIS subsystems",
    )

    # ── percysuits ──────────────────────────────────────────────────────
    percy_cmd = subparsers.add_parser(
        "percysuits",
        help="Pull all Perseus LLM models (auto-detects Ollama)",
    )
    percy_cmd.add_argument(
        "--host",
        default=None,
        help="Ollama host URL (default: OLLAMA_HOST or localhost)",
    )
    percy_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show model status without pulling",
    )
    percy_cmd.add_argument(
        "--ssh",
        default=None,
        help="Pull via SSH (e.g. --ssh user@host)",
    )

    # ── exec ────────────────────────────────────────────────────────────
    exec_cmd = subparsers.add_parser(
        "exec",
        help="Execute code inline, from stdin, or create-and-open (co) a file",
    )
    exec_cmd.add_argument("code", nargs="?", default=None, help="Code string, or 'co' for create-and-open mode")
    exec_cmd.add_argument("filename", nargs="?", default=None, help="Filename for 'co' mode (e.g. test.py)")
    exec_cmd.add_argument("--lang", choices=["python", "r"], default="python", help="Language (default: python)")
    exec_cmd.add_argument("--file", default=None, dest="exec_file", help="Read code from file instead")
    exec_cmd.add_argument("--dir", default=None, dest="co_dir", help="Directory for co files (default: ./cofs)")

    # ── edit ────────────────────────────────────────────────────────────
    edit_cmd = subparsers.add_parser(
        "edit",
        help="Open a file in the built-in scientific editor (Textual)",
    )
    edit_cmd.add_argument("file", help="File to edit (creates if not found)")
    edit_cmd.add_argument("--run", action="store_true", help="Run the file after saving (ctrl+r)")
    edit_cmd.add_argument("--lang", default=None, help="Language hint (python/r/shell)")

    # ── repl ────────────────────────────────────────────────────────────
    repl_cmd = subparsers.add_parser(
        "repl",
        help="Launch headless polyglot REPL (36 languages)",
    )
    repl_cmd.add_argument("--lang", default="python", help="Default language (default: python)")
    repl_cmd.add_argument(
        "--polyglot",
        action="store_true",
        default=True,
        help="Enable cross-language variable bridging (default: on)",
    )
    repl_cmd.add_argument("--no-polyglot", action="store_true", help="Disable cross-language variable bridging")
    repl_cmd.add_argument("--no-detect", action="store_true", help="Disable auto-detection (use --lang only)")

    # ── convert-checkpoint ───────────────────────────────────────────────
    convert_cmd = subparsers.add_parser(
        "convert-checkpoint",
        help="Convert a .pt training checkpoint to GGUF (optionally TurboQuant-compressed)",
    )
    convert_cmd.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    convert_cmd.add_argument("--output", required=True, help="Output .gguf file path")
    convert_cmd.add_argument("--tokenizer-dir", default=None, help="Autoresearch tokenizer directory")
    convert_cmd.add_argument(
        "--turbo-bits",
        type=int,
        default=0,
        choices=[0, 2, 3, 4],
        help="TurboQuant compression bits (0 = no compression, default: 0)",
    )

    # ── crypto ──────────────────────────────────────────────────────────
    crypto_cmd = subparsers.add_parser(
        "crypto",
        help="Post-quantum cryptography utilities (ML-KEM-768 + ChaCha20)",
    )
    crypto_sub = crypto_cmd.add_subparsers(dest="crypto_command")

    crypto_keygen = crypto_sub.add_parser("keygen", help="Generate an ML-KEM-768 key pair")
    crypto_keygen.add_argument("--name", required=True, help="Key pair identifier")
    crypto_keygen.add_argument("--output", default=None, help="Output directory for key files (default: keystore)")

    crypto_encrypt = crypto_sub.add_parser("encrypt", help="Hybrid-encrypt a file")
    crypto_encrypt.add_argument("file", help="File to encrypt")
    crypto_encrypt.add_argument("--to", required=True, dest="recipient", help="Recipient key name or public key file")

    crypto_decrypt = crypto_sub.add_parser("decrypt", help="Decrypt a hybrid-encrypted file")
    crypto_decrypt.add_argument("file", help="File to decrypt")
    crypto_decrypt.add_argument("--key", required=True, help="Key name in keystore")

    # ── list-datasets ────────────────────────────────────────────────────
    subparsers.add_parser(
        "list-datasets",
        help="List all built-in datasets and their cache status",
    )

    # ── download-bootstrap ───────────────────────────────────────────────
    bootstrap_cmd = subparsers.add_parser(
        "download-bootstrap",
        help="Download large bootstrap weight files from CKAN API",
    )
    bootstrap_cmd.add_argument(
        "--survey",
        choices=["csads_2021", "csads_2023", "csus_2019", "csus_2023", "all"],
        default="all",
        help="Which bootstrap file to download (default: all)",
    )
    bootstrap_cmd.add_argument(
        "--limit",
        type=int,
        default=32000,
        help="Max records to fetch from CKAN (default: 32000)",
    )

    # ── verify-pollution (W6 pollution-health CLI) ─────────────────────
    try:
        from moirais.verify_pollution import register_subparser as _vp_register
        _vp_register(subparsers)
    except ImportError:
        pass

    # ── verify-earth-engine (EE auth smoke check) ──────────────────────
    try:
        from moirais.verify_earth_engine import register_subparser as _vee_register
        _vee_register(subparsers)
    except ImportError:
        pass

    return parser


def _load_dotenv_if_present() -> None:
    """Auto-load a nearby .env file into os.environ so subcommands pick up
    GOOGLE_CLOUD_PROJECT, MOIRAIS_EE_KEY_PATH, etc. without requiring the
    user to `source` before calling `moirais`. Zero-dep parser — only simple
    KEY=VALUE lines, ignores comments and blanks, won't overwrite
    already-exported variables (shell wins).

    Tests that need a deterministic "no credentials" baseline set
    MOIRAIS_SKIP_DOTENV=1 so this loader becomes a no-op — otherwise
    a .env with valid keys would defeat the purge."""
    import os
    if os.environ.get("MOIRAIS_SKIP_DOTENV") == "1":
        return
    from pathlib import Path
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[5] / ".env",  # dev/sphinx/project/.env
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            for raw in resolved.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def main() -> int:
    """
    Entry point for the MOIRAIS command line interface.
    """
    _load_dotenv_if_present()

    if len(sys.argv) > 1 and sys.argv[1].startswith("?"):
        question = sys.argv[1][1:].strip()
        if not question and len(sys.argv) > 2:
            question = " ".join(sys.argv[2:])
        elif not question:
            print("Usage: moirais ?<question>")
            return 1
        payload = ask_percy(question, stream=sys.stdout.isatty())
        if "output_stream" in payload:
            for chunk in payload["output_stream"]:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            print(payload.get("output_text", ""))
        return 0

    parser = build_parser()
    args = parser.parse_args()

    # Auto-detect: no subcommand + interactive TTY → launch TUI or chat REPL.
    if args.command is None:
        if sys.stdout.isatty() and sys.stdin.isatty():
            try:
                from .tui import _TEXTUAL_AVAILABLE, launch_tui

                if _TEXTUAL_AVAILABLE:
                    return launch_tui()
            except ImportError:
                pass
            # Textual not available — try chat REPL.
            from .chat import run_chat_repl

            return run_chat_repl()
        parser.print_help()
        return 0

    if args.command == "pipeline":
        if not args.all and not args.modules:
            parser.print_help()
            return 0
        # Use rich progress display when running in an interactive terminal.
        if sys.stdout.isatty():
            from .progress import execute_pipeline_with_progress

            return execute_pipeline_with_progress(
                modules=args.modules,
                cpads_csv=args.cpads_csv,
                dataset_key=getattr(args, "dataset", None),
                output_dir=args.output_dir,
                silent=args.yes,
                track_carbon=not args.no_carbon,
            )
        return execute_pipeline(
            modules=args.modules,
            cpads_csv=args.cpads_csv,
            dataset_key=getattr(args, "dataset", None),
            output_dir=args.output_dir,
            silent=args.yes,
            track_carbon=not args.no_carbon,
        )

    if args.command == "parity-review":
        from ._parity import build_parity_matrix, summarize_parity_matrix, write_parity_matrix

        matrix = build_parity_matrix(args.epiml_root)
        summary = summarize_parity_matrix(matrix)
        if args.output:
            write_parity_matrix(matrix, args.output)
            print(f"Wrote audit matrix to {args.output}")
        print(
            "Audit summary:",
            f"total={summary.total_rows}",
            f"present={summary.already_present}",
            f"scaffolded={summary.scaffolded_incomplete}",
            f"missing={summary.missing}",
            f"deferred={summary.deferred}",
        )
        return 0

    if args.command == "list-modules":
        for spec in list_modules():
            print(spec["name"])
            print(f"  {spec['description']}")
            print(f"  outputs: {', '.join(spec['output_files'])}")
        return 0

    if args.command == "run-module":
        outputs = run_module(
            args.module,
            cpads_csv=args.cpads_csv,
            dataset_key=getattr(args, "dataset", None),
            output_dir=args.output_dir,
        )
        print(f"Completed module: {args.module}")
        print("Generated tables:", ", ".join(outputs.keys()))
        return 0

    if args.command == "run-modules":
        selected = args.modules or [item["name"] for item in list_modules()]
        ds_key = getattr(args, "dataset", None)
        results = {
            module_name: run_module(
                module_name,
                cpads_csv=args.cpads_csv,
                dataset_key=ds_key,
                output_dir=args.output_dir,
            )
            for module_name in selected
        }
        print("Completed modules:", ", ".join(results.keys()))
        return 0

    if args.command in ("percy", "perseus"):
        return _handle_percy(args)

    if args.command == "serve":
        from .perseus_relay import serve
        serve(port=args.port, token=args.token, bind=args.bind)
        return 0

    if args.command in ("agent", "assistant"):
        use_stream = not getattr(args, "no_stream", False)
        try:
            from .agent import create_agent

            agent = create_agent()
            if use_stream:
                for chunk in agent.chat_stream(args.question):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
            else:
                resp = agent.chat(args.question)
                print(resp.text)
                if resp.tool_calls_made:
                    print(f"\n[{len(resp.tool_calls_made)} tool calls in {resp.iterations} iterations]")
            agent.close()
        except Exception:
            payload = ask_percy(
                args.question,
                context=getattr(args, "context", None),
                stream=use_stream,
            )
            if use_stream:
                for chunk in payload["output_stream"]:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
            else:
                print(payload["output_text"])
        return 0

    if args.command == "ask":
        use_stream = not getattr(args, "no_stream", False)
        payload = ask_percy(
            args.question,
            context=args.context,
            model=getattr(args, "model", None),
            stream=use_stream,
        )
        if use_stream:
            for chunk in payload["output_stream"]:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            print(payload["output_text"])
        return 0

    if args.command == "chat":
        from .chat import run_chat_repl

        return run_chat_repl(agent=args.agent)

    if args.command == "tui":
        from .tui import launch_tui

        return launch_tui()

    if args.command == "selftest":
        from .selftest import run_selftest

        return run_selftest()

    if args.command == "percysuits":
        return _handle_percysuits(args)

    if args.command == "exec":
        return _handle_exec(args)

    if args.command == "edit":
        return _handle_edit(args)

    if args.command == "repl":
        return _handle_repl(args)

    if args.command == "list-datasets":
        from .data import list_datasets

        datasets = list_datasets()
        print(f"{'Key':<45} {'Type':<12} {'Rows':>8}  {'Status'}")
        print("-" * 80)
        for d in datasets:
            status = f"{d['rows']:,}" if d["cached"] else "not cached"
            print(f"{d['key']:<45} {d['type']:<12} {status:>8}")
        return 0

    if args.command == "verify-pollution":
        from .verify_pollution import handle_verify_pollution
        return handle_verify_pollution(args)

    if args.command == "verify-earth-engine":
        from .verify_earth_engine import handle_verify_earth_engine
        return handle_verify_earth_engine(args)

    if args.command == "download-bootstrap":
        from .data import DATASET_CATALOG, fetch_ckan_to_cache

        bootstrap_keys = {
            "csads_2021": "oc_csads_2021_bootstrap",
            "csads_2023": "oc_csads_2023_bootstrap",
            "csus_2019": "oc_csus_2019_bootstrap",
            "csus_2023": "oc_csus_2023_bootstrap",
        }
        targets = list(bootstrap_keys.values()) if args.survey == "all" else [bootstrap_keys[args.survey]]
        for key in targets:
            entry = DATASET_CATALOG.get(key)
            if not entry:
                print(f"  Unknown: {key}")
                continue
            rid = entry.get("ckan_resource_id", "")
            if not rid:
                print(f"  {key}: no CKAN resource ID — download the CSV manually to {entry['local_path']}")
                continue
            print(f"  Downloading {key} from CKAN (limit={args.limit})...")
            try:
                df = fetch_ckan_to_cache(key, limit=args.limit)
                print(f"    OK: {len(df):,} rows cached")
            except Exception as e:
                print(f"    ERROR: {e}")
        return 0

    if args.command == "crypto":
        import getpass
        from pathlib import Path as _CryptoPath

        if args.crypto_command == "keygen":
            from .crypto.hybrid import keygen as _hybrid_keygen
            from .crypto.keystore import create_keystore as _ks_create
            from .crypto.keystore import store_keypair as _ks_store

            if args.output:
                pk, sk = _hybrid_keygen()
                out_dir = _CryptoPath(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                pk_path = out_dir / f"{args.name}.moiraispk"
                sk_path = out_dir / f"{args.name}.moiraissk"
                pk_path.write_bytes(pk)
                sk_path.write_bytes(sk)
                print(f"Public key:  {pk_path}")
                print(f"Secret key:  {sk_path}")
            else:
                pw = getpass.getpass("Keystore password: ")
                ks_path = str(_CryptoPath("~/.moirais/keys/keystore.json").expanduser())
                if not _CryptoPath(ks_path).exists():
                    _ks_create(pw, path=ks_path)
                    print(f"Created keystore at {ks_path}")
                pk, sk = _hybrid_keygen()
                _ks_store(args.name, pk, sk, pw, path=ks_path)
                print(f"Key pair '{args.name}' stored in keystore")
            return 0

        if args.crypto_command == "encrypt":
            from .crypto.hybrid import hybrid_encrypt as _h_enc

            src = _CryptoPath(args.file)
            if not src.exists():
                print(f"File not found: {src}")
                return 1
            pk_path = _CryptoPath(args.recipient)
            if pk_path.exists():
                pk = pk_path.read_bytes()
            else:
                from .crypto.keystore import load_keypair as _ks_load

                pw = getpass.getpass("Keystore password: ")
                pk, _ = _ks_load(args.recipient, pw)
            ct = _h_enc(src.read_bytes(), pk)
            out = src.with_suffix(src.suffix + ".moiraisenc")
            out.write_bytes(ct)
            print(f"Encrypted: {out}")
            return 0

        if args.crypto_command == "decrypt":
            from .crypto.hybrid import hybrid_decrypt as _h_dec
            from .crypto.keystore import load_keypair as _ks_load2

            src = _CryptoPath(args.file)
            if not src.exists():
                print(f"File not found: {src}")
                return 1
            pw = getpass.getpass("Keystore password: ")
            _, sk = _ks_load2(args.key, pw)
            plaintext = _h_dec(src.read_bytes(), sk)
            out_name = str(src).replace(".moiraisenc", "")
            if out_name == str(src):
                out_name = str(src) + ".dec"
            _CryptoPath(out_name).write_bytes(plaintext)
            print(f"Decrypted: {out_name}")
            return 0

        crypto_cmd_parser = build_parser()._subparsers._group_actions[0].choices.get("crypto")
        if crypto_cmd_parser:
            crypto_cmd_parser.print_help()
        return 0

    if args.command == "convert-checkpoint":
        from .pt2gguf import convert

        convert(args.checkpoint, args.output, args.tokenizer_dir, args.turbo_bits)
        return 0

    if args.command == "doctor":
        from .doctor import run_doctor

        return run_doctor()

    if args.command == "profile-dataset":
        from .dataset import load_dataset, profile_dataset, suggest_analysis_plan

        df = load_dataset(args.csv)
        profile = profile_dataset(
            df,
            hint_treatment=args.treatment,
            hint_outcome=args.outcome,
            hint_weights=args.weights,
        )
        print(profile.summary_table())
        if profile.suggested_treatment:
            print(f"\nSuggested treatment: {profile.suggested_treatment}")
        if profile.suggested_outcome:
            print(f"Suggested outcome:   {profile.suggested_outcome}")
        if profile.suggested_weights:
            print(f"Suggested weights:   {profile.suggested_weights}")
        if args.suggest:
            plan = suggest_analysis_plan(profile)
            print("\n--- Suggested Analysis Plan ---")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. [{step['analysis']}] {step['rationale']}")
        return 0

    if args.command == "sample":
        from .dataset import load_dataset
        from .sampling import (
            cluster_sample,
            pps_sample,
            simple_random_sample,
            stratified_sample,
        )

        df = load_dataset(args.csv)
        method = args.method

        if method == "srs":
            sample = simple_random_sample(df, args.n, seed=args.seed)
        elif method == "stratified":
            if not args.strata_col:
                print("Error: --strata-col is required for stratified sampling")
                return 1
            sample = stratified_sample(
                df,
                args.strata_col,
                args.n,
                proportional=args.proportional,
                seed=args.seed,
            )
        elif method == "cluster":
            if not args.cluster_col:
                print("Error: --cluster-col is required for cluster sampling")
                return 1
            sample = cluster_sample(df, args.cluster_col, args.n, seed=args.seed)
        elif method == "pps":
            if not args.size_col:
                print("Error: --size-col is required for PPS sampling")
                return 1
            sample = pps_sample(df, args.size_col, args.n, seed=args.seed)
        else:
            print(f"Unknown method: {method}")
            return 1

        print(f"Sampled {len(sample)} rows using {method}")
        if args.output:
            sample.to_csv(args.output, index=False)
            print(f"Wrote sample to {args.output}")
        else:
            print(sample.head(20).to_string())
        return 0

    if args.command == "inspect":
        from pathlib import Path as _Path

        from .inspector import (
            inspect_directory,
            inspect_output,
            render_inspection,
        )

        target = _Path(args.path)
        if target.is_file():
            render_inspection(inspect_output(target))
        elif target.is_dir():
            results = inspect_directory(target, module_name=args.module)
            if not results:
                print(f"No CSV files found in {target}")
                return 1
            for result in results:
                render_inspection(result)
                print()
        else:
            print(f"Path not found: {target}")
            return 1
        return 0

    if args.command == "verify":
        from pathlib import Path as _Path

        from .inspector import (
            render_verification,
            verify_directory,
            verify_statistical_output,
        )

        target = _Path(args.path)
        if target.is_file():
            report = verify_statistical_output(target)
            render_verification(report)
            return 0 if report.passed else 1
        elif target.is_dir():
            reports = verify_directory(target, module_name=args.module)
            if not reports:
                print(f"No CSV files found in {target}")
                return 1
            all_passed = True
            for report in reports:
                render_verification(report)
                print()
                if not report.passed:
                    all_passed = False
            return 0 if all_passed else 1
        else:
            print(f"Path not found: {target}")
            return 1

    parser.print_help()
    return 0


def _handle_exec(args: argparse.Namespace) -> int:
    import os
    import subprocess
    import tempfile

    if args.code == "co":
        return _handle_exec_co(args)

    code = args.code
    if args.exec_file:
        with open(args.exec_file) as f:
            code = f.read()
    elif code is None:
        if sys.stdin.isatty():
            print("Enter code (ctrl+d to execute):")
        code = sys.stdin.read()
    if not code or not code.strip():
        print("No code provided.")
        return 1

    if args.lang == "r":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            return subprocess.call(["Rscript", tmp])
        finally:
            os.unlink(tmp)

    ns = {"__name__": "__moirais_exec__"}
    try:
        import numpy as np
        import pandas as pd

        ns["np"] = np
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        from moirais import fn
        from moirais.fn._registry import REGISTRY

        ns["REGISTRY"] = REGISTRY
        ns["fn"] = fn
    except ImportError:
        pass
    try:
        exec(compile(code, "<moirais-exec>", "exec"), ns)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _handle_exec_co(args: argparse.Namespace) -> int:
    from pathlib import Path

    filename = args.filename
    if not filename:
        print("Usage: moirais exec co <filename>")
        print("  e.g. moirais exec co test.py")
        return 1

    co_dir = Path(args.co_dir) if args.co_dir else Path.cwd() / "cofs"
    co_dir.mkdir(parents=True, exist_ok=True)
    filepath = co_dir / filename

    ext = filepath.suffix.lower()
    lang_map = {
        ".py": "python",
        ".r": "r",
        ".sh": "shell",
        ".js": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".ml": "ocaml",
        ".lua": "lua",
        ".ts": "typescript",
        ".tex": "latex",
    }
    lang = args.lang if args.lang != "python" else lang_map.get(ext, "python")

    if not filepath.exists():
        filepath.write_text("")
        print(f"Created {filepath}")
    else:
        print(f"Opening {filepath}")

    try:
        from .editor import launch_editor

        return launch_editor(str(filepath), run_on_save=True, lang_hint=lang)
    except ImportError:
        import subprocess

        return subprocess.call(["nano", str(filepath)])


def _handle_edit(args: argparse.Namespace) -> int:
    from pathlib import Path

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("")

    try:
        from .editor import launch_editor

        return launch_editor(
            str(filepath),
            run_on_save=args.run,
            lang_hint=args.lang,
        )
    except ImportError:
        import subprocess

        editor = "nano"
        return subprocess.call([editor, str(filepath)])


def _handle_repl(args: argparse.Namespace) -> int:
    from .polyglot import run_headless_repl

    polyglot = not getattr(args, "no_polyglot", False)
    auto_detect = not getattr(args, "no_detect", False)
    return run_headless_repl(
        polyglot=polyglot,
        auto_detect=auto_detect,
        lang=args.lang,
    )


PERCY_MODELS = [
    ("functiongemma:270m", "0.3 GB", "ToolCall", "FunctionGemma 270M — fast tool calling"),
    ("functiongemma", "0.3 GB", "ToolCall", "FunctionGemma (default tag)"),
    ("gemma4:e2b", "7.2 GB", "LLM", "Perseus base model (Gemma 4)"),
    ("gemma4:e2b-it-q4_K_M", "7.2 GB", "LLM", "Gemma 4 instruction-tuned Q4"),
    ("gemma4:e2b-it-q8_0", "8.1 GB", "LLM", "Gemma 4 instruction-tuned Q8"),
    ("gemma4:e4b-it-q4_K_M", "9.6 GB", "LLM", "Gemma 4 large Q4"),
    ("gemma4:e4b-it-q8_0", "11 GB", "LLM", "Gemma 4 large Q8"),
    ("gemma3:4b", "3.3 GB", "LLM", "Gemma 3 lightweight"),
    ("gemma3n:e2b", "5.6 GB", "LLM", "Gemma 3 Nano small"),
    ("gemma3n:e4b", "7.5 GB", "LLM", "Gemma 3 Nano large"),
    ("mistral-nemo", "7.1 GB", "LLM", "Mistral Nemo 12B"),
    ("qwen3.5:0.8b", "1.0 GB", "LLM", "Qwen 3.5 tiny"),
    ("qwen3.5:2b", "2.7 GB", "LLM", "Qwen 3.5 small"),
    ("qwen3.5:4b", "3.4 GB", "LLM", "Qwen 3.5 medium"),
    ("qwen3.5:9b", "6.6 GB", "LLM", "Qwen 3.5 large"),
    ("nemotron-3-nano:4b", "2.8 GB", "LLM", "NVIDIA Nemotron Nano Q4"),
    ("nemotron-3-nano:4b-q8_0", "4.2 GB", "LLM", "NVIDIA Nemotron Nano Q8"),
    ("nemotron-3-nano:4b-bf16", "8.0 GB", "LLM", "NVIDIA Nemotron Nano BF16"),
    ("nemotron-cascade-2", "14 GB", "LLM", "NVIDIA Nemotron Cascade 24B"),
    ("deepseek-r1:1.5b", "1.1 GB", "Reasoning", "DeepSeek R1 tiny"),
    ("deepseek-r1:8b", "5.2 GB", "Reasoning", "DeepSeek R1 medium"),
    ("deepseek-r1:14b", "9.0 GB", "Reasoning", "DeepSeek R1 large"),
    ("magistral:24b", "14 GB", "Reasoning", "Mistral Magistral 24B"),
    ("lfm2", "14 GB", "LLM", "Liquid LFM2"),
    ("lfm2:24b-q4_K_M", "14 GB", "LLM", "Liquid LFM2 24B Q4"),
    ("lfm2.5-thinking", "0.7 GB", "Reasoning", "Liquid LFM2.5 thinking"),
    ("qwen3-vl:8b", "6.1 GB", "Vision", "Qwen 3 Vision 8B"),
    ("glm-ocr", "2.2 GB", "Vision", "GLM OCR model"),
    ("gpt-oss:20b", "12 GB", "LLM", "Open-source GPT 20B (FreeAPI)"),
    ("translategemma", "3.3 GB", "Translation", "Google TranslateGemma"),
    ("phi4-mini", "2.5 GB", "LLM", "Microsoft Phi-4 Mini"),
    ("phi4-mini:3.8b-q8_0", "4.1 GB", "LLM", "Microsoft Phi-4 Mini Q8"),
    ("smollm2:1.7b", "1.0 GB", "LLM", "HuggingFace SmolLM2 tiny"),
    ("llama3.2:3b", "2.0 GB", "LLM", "Meta Llama 3.2 3B"),
    ("granite3.3:8b", "4.9 GB", "LLM", "IBM Granite 3.3 8B"),
    ("granite3.3:2b", "1.6 GB", "LLM", "IBM Granite 3.3 2B"),
    ("kimi-k2.5:cloud", "0 GB", "Cloud", "Moonshot Kimi K2.5 (cloud)"),
    ("minimax-m2.7:cloud", "0 GB", "Cloud", "MiniMax M2.7 (cloud)"),
    ("nomic-embed-text", "0.3 GB", "Embedding", "Nomic text embeddings"),
    ("snowflake-arctic-embed2", "0.4 GB", "Embedding", "Snowflake embeddings"),
]


def _handle_percy(args: argparse.Namespace) -> int:
    import os
    import sys

    from .agent import create_agent

    use_stream = not getattr(args, "no_stream", False)
    model = getattr(args, "model", None)
    provider = None
    base_url = None

    cloud_url = getattr(args, "cloud", None) or os.environ.get("PERSEUS_CLOUD_URL")
    cloud_token = os.environ.get("PERSEUS_CLOUD_TOKEN")

    if getattr(args, "local", False):
        print("Perseus [local mode] — using local Ollama only...")
    elif cloud_url:
        print(f"Perseus [cloud] — connecting to {cloud_url}...")
    elif getattr(args, "remote", False):
        cloud_url = os.environ.get("PERSEUS_CLOUD_URL")
        if cloud_url:
            print(f"Perseus [cloud] — connecting to {cloud_url}...")
        else:
            pi = os.environ.get("MOIRAIS_PI_HOST")
            if pi:
                host_part = pi.split("@")[-1] if "@" in pi else pi
                cloud_url = f"http://{host_part}:8421"
                print(f"Perseus [cloud via Pi] — connecting to {cloud_url}...")
            else:
                provider = "freeapi"
                print("Perseus [internet fallback] — no cloud relay configured, using community servers...")
                print("  Tip: Set PERSEUS_CLOUD_URL or run `moirais serve` on your Pi for the real Perseus.")
    elif getattr(args, "freeapi", False):
        provider = "freeapi"
        print("Perseus [community servers] — using OllamaFreeAPI...")
    elif getattr(args, "pi", None):
        pi_host = args.pi
        base_url = f"http://{pi_host}:11434" if "://" not in pi_host else pi_host
        print(f"Perseus [Pi] — connecting to {base_url}...")
    elif os.environ.get("MOIRAIS_PI_HOST"):
        pi = os.environ["MOIRAIS_PI_HOST"]
        host_part = pi.split("@")[-1] if "@" in pi else pi
        base_url = f"http://{host_part}:11434"

    agent = create_agent(model=model, base_url=base_url, provider=provider, cloud_url=cloud_url, cloud_token=cloud_token)
    model_name = getattr(agent, "_model", "freeapi")
    print(f"Perseus [{model_name}] ready.\n")

    question = getattr(args, "question", None)
    if question is None:
        print("Type your question (or 'quit' to exit):\n")
        while True:
            try:
                q = input("percy> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                print("We suffer more often in imagination than in reality. — Seneca")
                break
            _percy_answer(agent, q, use_stream, sys.stdout)
            print()
    else:
        _percy_answer(agent, question, use_stream, sys.stdout)

    if hasattr(agent, "close"):
        agent.close()
    return 0


def _percy_answer(agent, question, use_stream, out):
    if use_stream and hasattr(agent, "chat_stream"):
        for chunk in agent.chat_stream(question):
            out.write(chunk)
            out.flush()
        out.write("\n")
    else:
        resp = agent.chat(question)
        out.write(resp.text + "\n")
        if resp.tool_calls_made:
            out.write(f"\n[{len(resp.tool_calls_made)} tool calls in {resp.iterations} iterations]\n")


def _handle_percysuits(args: argparse.Namespace) -> int:
    import os

    ssh_target = getattr(args, "ssh", None)
    host = args.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    dry_run = args.dry_run

    if ssh_target:
        print("Perseus Model Suite")
        print(f"Target: {ssh_target} (SSH)")
        print()
        installed, installed_sizes = _percysuits_get_installed_ssh(ssh_target)
    else:
        print("Perseus Model Suite")
        print(f"Target: {host}")
        print()
        installed, installed_sizes = _percysuits_get_installed(host)
    if installed is None:
        return 1

    to_pull = []
    have = []
    for name, size, category, desc in PERCY_MODELS:
        if name in installed:
            have.append((name, size, category, desc))
        else:
            to_pull.append((name, size, category, desc))

    print(f"{'Model':<30} {'Size':>7}  {'Type':<12} Status")
    print("-" * 72)
    for name, size, category, desc in PERCY_MODELS:
        tag = "INSTALLED" if name in installed else "MISSING"
        sym = "+" if tag == "INSTALLED" else "-"
        print(f"  {sym} {name:<28} {size:>7}  {category:<12} {desc}")
    print()
    print(f"Installed: {len(have)}/{len(PERCY_MODELS)} | Missing: {len(to_pull)}")

    if installed_sizes:
        total = sum(installed_sizes.values()) / (1024**3)
        print(f"Total installed: {total:.1f} GB")
    print()

    if not to_pull:
        print("All Perseus models installed!")
        return 0

    if dry_run:
        print("Would pull:")
        for name, size, _cat, desc in to_pull:
            print(f"  {name} ({size}) — {desc}")
        return 0

    if ssh_target:
        return _percysuits_pull_ssh(ssh_target, to_pull)
    return _percysuits_pull(host, to_pull)


def _percysuits_get_installed_ssh(ssh_target: str):
    import json as _json
    import subprocess

    try:
        result = subprocess.run(
            ["ssh", ssh_target, "curl -s http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        models_data = _json.loads(result.stdout).get("models", [])
        installed = set()
        sizes = {}
        for m in models_data:
            name = m.get("name", "")
            installed.add(name)
            if ":" in name:
                installed.add(name.split(":")[0])
            sz = m.get("size", 0)
            if sz:
                sizes[name] = sz
        return installed, sizes
    except Exception as exc:
        print(f"ERROR: cannot reach Ollama via SSH {ssh_target}: {exc}")
        return None, None


def _percysuits_pull_ssh(ssh_target: str, to_pull: list) -> int:
    import subprocess

    pulled = 0
    failed = []
    for i, (name, size, _cat, _desc) in enumerate(to_pull, 1):
        print(f"[{i}/{len(to_pull)}] {name} ({size}) ...")
        try:
            proc = subprocess.Popen(
                ["ssh", ssh_target, f"ollama pull {name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    print(f"  {stripped}")
            proc.wait(timeout=600)
            if proc.returncode == 0:
                pulled += 1
                print(f"  Done: {name}")
            else:
                failed.append(name)
                print(f"  FAILED: {name} (exit {proc.returncode})")
        except Exception as exc:
            failed.append(name)
            print(f"  FAILED: {name} ({exc})")

    print()
    print(f"Pulled: {pulled} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 0 if not failed else 1


def _percysuits_get_installed(host: str):
    try:
        import httpx

        resp = httpx.get(f"{host}/api/tags", timeout=15)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
        installed = set()
        sizes = {}
        for m in models_data:
            name = m.get("name", "")
            installed.add(name)
            if ":" in name:
                installed.add(name.split(":")[0])
            sz = m.get("size", 0)
            if sz:
                sizes[name] = sz
        return installed, sizes
    except Exception as exc:
        print(f"ERROR: cannot reach Ollama at {host}: {exc}")
        print()
        print("Make sure Ollama is running:")
        print("  ollama serve")
        print()
        print("Or set OLLAMA_HOST if Ollama is on another machine:")
        print("  export OLLAMA_HOST=http://your-server.local:11434")
        return None, None


def _percysuits_pull(host: str, to_pull: list) -> int:
    import json as _json

    import httpx

    pulled = 0
    failed = []
    for i, (name, size, _cat, _desc) in enumerate(to_pull, 1):
        print(f"[{i}/{len(to_pull)}] {name} ({size}) ...")
        try:
            with httpx.stream(
                "POST",
                f"{host}/api/pull",
                json={"name": name, "stream": True},
                timeout=httpx.Timeout(
                    connect=15,
                    read=600,
                    write=15,
                    pool=15,
                ),
            ) as resp:
                resp.raise_for_status()
                last_status = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except ValueError:
                        continue
                    status = data.get("status", "")
                    if status != last_status:
                        print(f"  {status}")
                        last_status = status
                    if "error" in data:
                        raise RuntimeError(data["error"])
            pulled += 1
            print(f"  Done: {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAILED: {name} ({exc})")

    print()
    print(f"Pulled: {pulled} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
