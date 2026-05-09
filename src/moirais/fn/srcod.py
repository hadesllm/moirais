"""AST depth analysis. 'Everything begins with choice.' -- The Merovingian"""

from __future__ import annotations

import ast

from ._containers import DescriptiveResult


def ast_depth(
    source: str,
) -> DescriptiveResult:
    """Compute the depth and complexity metrics of a Python source AST.

    Parses Python source code into an Abstract Syntax Tree and reports
    structural metrics: max depth, node count, function/class counts,
    and cyclomatic complexity estimate.

    Parameters
    ----------
    source : str
        Python source code string.

    Returns
    -------
    DescriptiveResult
        ``value`` is the maximum AST depth; ``extra`` has node counts
        and complexity metrics.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse source: {exc}") from exc

    def _depth(node):
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_depth(c) for c in children)

    max_d = _depth(tree)
    n_nodes = sum(1 for _ in ast.walk(tree))
    n_funcs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))

    branch_nodes = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert, ast.BoolOp)
    n_branches = sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))
    cyclomatic = n_branches + 1

    n_imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))

    return DescriptiveResult(
        name="AST Depth Analysis",
        value=max_d,
        extra={
            "max_depth": max_d,
            "n_nodes": n_nodes,
            "n_functions": n_funcs,
            "n_classes": n_classes,
            "cyclomatic_complexity": cyclomatic,
            "n_imports": n_imports,
            "n_branches": n_branches,
        },
    )


srcod = ast_depth


def cheatsheet() -> str:
    return "ast_depth({}) -> AST depth analysis. 'Everything begins with choice.' -- The "
