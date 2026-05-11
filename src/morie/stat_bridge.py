"""
Bridge between Go TIDE and Python morie statistical functions.

Three modes:
  python -m morie.stat_bridge registry-json   -> JSON dump for Go tree
  python -m morie.stat_bridge help             -> formatted help text
  python -m morie.stat_bridge exec "cmd args"  -> run command, print result
"""

from __future__ import annotations

import json
import sys


def registry_json() -> str:
    from morie.fn._registry import REGISTRY

    entries = []
    for _key, entry in sorted(REGISTRY.items()):
        entries.append(
            {
                "short": entry.short,
                "full": entry.full,
                "category": entry.category,
                "description": entry.description,
                "quote": entry.quote,
            }
        )
    return json.dumps(entries)


def help_text() -> str:
    from morie.fn._registry import REGISTRY

    cats: dict[str, list] = {}
    for entry in REGISTRY.values():
        cats.setdefault(entry.category, []).append(entry.short)

    lines = ["MORIE Statistical Functions", "=" * 40, ""]
    for cat in sorted(cats):
        fns = sorted(cats[cat])
        lines.append(f"{cat} ({len(fns)})")
        for i in range(0, len(fns), 8):
            chunk = fns[i : i + 8]
            lines.append("  " + ", ".join(chunk))
        lines.append("")
    lines.append(f"Total: {len(REGISTRY)} functions")
    return "\n".join(lines)


class _BridgeLog:
    """File-like log that stat_commands handlers can .write() or call()."""

    def __init__(self):
        self._parts: list[str] = []

    def __call__(self, msg):
        self._parts.append(str(msg))

    def write(self, msg):
        self._parts.append(str(msg))

    def getvalue(self) -> str:
        return "\n".join(self._parts)


def exec_command(cmd_str: str) -> str:
    parts = cmd_str.strip().split()
    if not parts:
        return "Error: empty command"

    name = parts[0]
    args = parts[1:]

    try:
        from morie.stat_commands import resolve

        cmd = resolve(name)
        if cmd is not None:
            log = _BridgeLog()
            store: dict = {}
            cmd.handler_stat(args, log, store)
            return log.getvalue().rstrip()
    except Exception as e:
        return f"Error: {e}"

    try:
        from morie.fn._registry import REGISTRY

        if name in REGISTRY:
            mod = __import__(f"morie.fn.{name}", fromlist=[name])
            fn = getattr(mod, name)
            result = fn(*args) if args else fn()
            return str(result)
    except Exception as e:
        return f"Error running {name}: {e}"

    return f"Unknown command: {name}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m morie.stat_bridge <registry-json|help|exec 'cmd'>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "registry-json":
        print(registry_json())
    elif mode == "help":
        print(help_text())
    elif mode == "exec":
        if len(sys.argv) < 3:
            print("Error: exec requires a command string")
            sys.exit(1)
        cmd_str = " ".join(sys.argv[2:])
        print(exec_command(cmd_str))
    elif mode == "load-dataset":
        if len(sys.argv) < 3:
            print("Error: load-dataset requires a dataset name")
            sys.exit(1)
        name = sys.argv[2]
        try:
            from morie.data import load_dataset

            df = load_dataset(name)
            print(df.head().to_string())
            print(f"\nShape: {df.shape}")
        except Exception as e:
            print(f"Error loading '{name}': {e}")
    elif mode == "fn-info":
        if len(sys.argv) < 3:
            print("Error: fn-info requires a function name")
            sys.exit(1)
        name = sys.argv[2]
        from morie.fn._registry import REGISTRY

        e = REGISTRY.get(name)
        if e:
            print(f"{e.short} ({e.full})")
            print(f"Category: {e.category}")
            print(e.description)
            if e.quote:
                print(e.quote)
        else:
            print(f"Not found: {name}")
    elif mode == "fn-search":
        if len(sys.argv) < 3:
            print("Error: fn-search requires a query")
            sys.exit(1)
        q = " ".join(sys.argv[2:]).lower()
        from morie.fn._registry import REGISTRY

        matches = [
            (k, v)
            for k, v in REGISTRY.items()
            if q in k or q in v.full.lower() or q in v.description.lower() or q in v.category.lower()
        ]
        for k, v in matches[:20]:
            print(f"  {k:8s} {v.category:15s} {v.description}")
        if not matches:
            print("No matches.")
        print(f"\n{len(matches)} results")
    elif mode == "migration-json":
        try:
            if len(sys.argv) >= 4 and sys.argv[2] == "--config":
                from morie._migration import build_migration_inventory, load_mapping_from_json

                mapping = load_mapping_from_json(sys.argv[3])
                legacy_root = sys.argv[4] if len(sys.argv) >= 5 else "."
                df = build_migration_inventory(legacy_root=legacy_root, mapping=mapping)
                print(json.dumps(df.to_dict(orient="records"), default=str))
            else:
                from morie._migration import MIGRATION_INVENTORY

                print(json.dumps(MIGRATION_INVENTORY, default=str))
        except ImportError:
            print("[]")
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "parity-json":
        try:
            if len(sys.argv) >= 4 and sys.argv[2] == "--config":
                from morie._parity import build_parity_matrix, load_parity_config

                script_map, surface_map = load_parity_config(sys.argv[3])
                old_root = sys.argv[4] if len(sys.argv) >= 5 else "."
                df = build_parity_matrix(old_root, script_map=script_map, surface_map=surface_map)
                print(json.dumps(df.to_dict(orient="records"), default=str))
            else:
                from morie._parity import PARITY_MATRIX

                print(json.dumps(PARITY_MATRIX, default=str))
        except ImportError:
            print("[]")
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "verify":
        try:
            from morie.selftest import run_selftest

            results = run_selftest()
            for r in results:
                status = "PASS" if r.get("ok") else "FAIL"
                print(f"  {status}  {r.get('name', '?')}")
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "profile":
        if len(sys.argv) < 3:
            print("Error: profile requires a dataset name")
            sys.exit(1)
        name = sys.argv[2]
        try:
            from morie.data import load_dataset

            df = load_dataset(name)
            print(df.describe().to_string())
            print(f"\nShape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Dtypes:\n{df.dtypes.to_string()}")
            nulls = df.isnull().sum()
            if nulls.any():
                print(f"\nMissing:\n{nulls[nulls > 0].to_string()}")
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "compare-dirs":
        if len(sys.argv) < 4:
            print("Usage: python -m morie.stat_bridge compare-dirs <source> <target>")
            sys.exit(1)
        source, target = sys.argv[2], sys.argv[3]
        try:
            from morie._migration import compare_trees

            df = compare_trees(source, target)
            print(json.dumps(df.to_dict(orient="records"), default=str))
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "function-parity":
        if len(sys.argv) < 4:
            print("Usage: python -m morie.stat_bridge function-parity <source> <target>")
            sys.exit(1)
        source, target = sys.argv[2], sys.argv[3]
        try:
            from morie._parity import compare_function_parity, summarize_function_parity

            df = compare_function_parity(source, target)
            summary = summarize_function_parity(df)
            print(
                json.dumps(
                    {
                        "parity": df.to_dict(orient="records"),
                        "summary": {
                            "matched": summary.matched_functions,
                            "source_only": summary.source_only_functions,
                            "target_only": summary.target_only_functions,
                            "ratio": round(summary.parity_ratio, 3),
                        },
                    },
                    default=str,
                )
            )
        except Exception as e:
            print(f"Error: {e}")
    elif mode == "tide-parity":
        try:
            from morie._parity import TIDE_PARITY, summarize_tide_parity

            print(json.dumps({"features": TIDE_PARITY, "summary": summarize_tide_parity()}, default=str))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
