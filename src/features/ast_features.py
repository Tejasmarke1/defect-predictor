"""
ast_features.py
---------------
Extract structural complexity features from Python source code using the
built-in ``ast`` module and the ``radon`` library.

All public functions handle gracefully:
  - None or empty source strings
  - Files with syntax errors
  - Missing radon installation
"""

from __future__ import annotations

import ast
from typing import Optional

import numpy as np
from loguru import logger

# Optional radon import — degrade gracefully if not installed.
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit  # noqa: F401 (availability check only)
    _RADON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RADON_AVAILABLE = False
    logger.warning("radon not installed — cyclomatic_complexity will be 0.")

try:
    from radon.complexity import SCORE  # noqa: F401
    from cognitive_complexity.api import get_cognitive_complexity  # type: ignore
    _COG_AVAILABLE = True
except ImportError:
    _COG_AVAILABLE = False


# ---------------------------------------------------------------------------
# helpers / AST visitors
# ---------------------------------------------------------------------------

class _NestingDepthVisitor(ast.NodeVisitor):
    """Walk an AST tree and track maximum nesting depth."""

    NESTING_NODES = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.AsyncFor, ast.AsyncWith,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )

    def __init__(self) -> None:
        self.max_depth = 0
        self._current_depth = 0

    def _visit_nesting(self, node: ast.AST) -> None:
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    # Attach the same handler to all nesting node types dynamically.
    def visit_If(self, node):           self._visit_nesting(node)   # noqa: E704
    def visit_For(self, node):          self._visit_nesting(node)   # noqa: E704
    def visit_While(self, node):        self._visit_nesting(node)   # noqa: E704
    def visit_With(self, node):         self._visit_nesting(node)   # noqa: E704
    def visit_Try(self, node):          self._visit_nesting(node)   # noqa: E704
    def visit_AsyncFor(self, node):     self._visit_nesting(node)   # noqa: E704
    def visit_AsyncWith(self, node):    self._visit_nesting(node)   # noqa: E704
    def visit_FunctionDef(self, node):  self._visit_nesting(node)   # noqa: E704
    def visit_AsyncFunctionDef(self, node): self._visit_nesting(node)  # noqa: E704
    def visit_ClassDef(self, node):     self._visit_nesting(node)   # noqa: E704


class _FunctionMetricsVisitor(ast.NodeVisitor):
    """Collect per-function metrics: line length and argument count."""

    def __init__(self) -> None:
        self.function_lengths: list[int] = []
        self.arg_counts: list[int] = []

    def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        end_line = getattr(node, "end_lineno", node.lineno)
        length = max(1, end_line - node.lineno + 1)
        self.function_lengths.append(length)

        args = node.args
        n_args = (
            len(args.args)
            + len(args.posonlyargs)
            + len(args.kwonlyargs)
            + (1 if args.vararg else 0)
            + (1 if args.kwarg else 0)
        )
        self.arg_counts.append(n_args)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):       self._handle_func(node)  # noqa: E704
    def visit_AsyncFunctionDef(self, node):  self._handle_func(node)  # noqa: E704


# ---------------------------------------------------------------------------
# feature extractors
# ---------------------------------------------------------------------------

def _safe_parse(source: str) -> Optional[ast.Module]:
    """
    Parse *source* into an AST module, returning None on failure.

    Parameters
    ----------
    source : Python source code string.

    Returns
    -------
    ast.Module or None.
    """
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        logger.debug(f"SyntaxError during AST parse: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Unexpected parse error: {exc}")
        return None


def _default_features() -> dict:
    """Return a zero-filled feature dict used when source is unparseable."""
    return {
        "cyclomatic_complexity": 0,
        "cognitive_complexity": 0,
        "max_nesting_depth": 0,
        "num_functions": 0,
        "num_classes": 0,
        "avg_function_length": 0.0,
        "num_imports": 0,
        "ast_node_count": 0,
        "has_try_except": 0,
        "max_args_per_function": 0,
    }


def compute_cyclomatic_complexity(source: str) -> int:
    """
    Compute the total cyclomatic complexity for all functions/methods in *source*.

    Uses radon's ``cc_visit``. Returns 0 if radon is unavailable or source is invalid.

    Parameters
    ----------
    source : Python source code.

    Returns
    -------
    int — sum of cyclomatic complexity across all blocks.
    """
    if not _RADON_AVAILABLE or not source:
        return 0
    try:
        blocks = cc_visit(source)
        return int(sum(b.complexity for b in blocks))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"cc_visit failed: {exc}")
        return 0


def compute_cognitive_complexity(source: str, tree: Optional[ast.Module]) -> int:
    """
    Compute cognitive complexity if the ``cognitive_complexity`` package is available.

    Falls back to 0 otherwise.

    Parameters
    ----------
    source : Python source code (unused if library unavailable).
    tree   : Pre-parsed AST (unused; kept for API consistency).

    Returns
    -------
    int — aggregate cognitive complexity score.
    """
    if not _COG_AVAILABLE or not source:
        return 0
    try:
        import ast as _ast
        from cognitive_complexity.api import get_cognitive_complexity as _gcc  # type: ignore
        module = _safe_parse(source)
        if module is None:
            return 0
        total = 0
        for node in _ast.walk(module):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                total += _gcc(node)
        return int(total)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"cognitive_complexity failed: {exc}")
        return 0


def compute_max_nesting_depth(tree: ast.Module) -> int:
    """
    Walk *tree* and return the maximum nesting depth of control-flow structures.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    int — maximum nesting depth (0 if no nesting found).
    """
    visitor = _NestingDepthVisitor()
    visitor.visit(tree)
    return visitor.max_depth


def compute_function_and_class_counts(tree: ast.Module) -> dict:
    """
    Count functions (sync + async) and classes in *tree*.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    dict with keys num_functions, num_classes.
    """
    num_functions = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    num_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    return {"num_functions": num_functions, "num_classes": num_classes}


def compute_function_length_and_args(tree: ast.Module) -> dict:
    """
    Compute average function length (lines) and max argument count across all functions.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    dict with keys avg_function_length, max_args_per_function.
    """
    visitor = _FunctionMetricsVisitor()
    visitor.visit(tree)
    avg_len = float(np.mean(visitor.function_lengths)) if visitor.function_lengths else 0.0
    max_args = int(max(visitor.arg_counts)) if visitor.arg_counts else 0
    return {
        "avg_function_length": round(avg_len, 4),
        "max_args_per_function": max_args,
    }


def compute_import_count(tree: ast.Module) -> int:
    """
    Count the number of import statements (``import`` + ``from … import``).

    Acts as a coupling proxy — more imports generally means more dependencies.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    int — total import statement count.
    """
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    )


def compute_ast_node_count(tree: ast.Module) -> int:
    """
    Count total AST nodes as a proxy for file size/complexity.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    int — total node count.
    """
    return sum(1 for _ in ast.walk(tree))


def compute_has_try_except(tree: ast.Module) -> int:
    """
    Detect whether the file uses any try/except block.

    Parameters
    ----------
    tree : Parsed AST module.

    Returns
    -------
    int (0 or 1).
    """
    return int(any(isinstance(n, ast.Try) for n in ast.walk(tree)))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def extract_ast_features(source: Optional[str]) -> dict:
    """
    Extract all AST-based structural features from *source*.

    Handles None, empty strings, and syntax errors gracefully by returning
    zero-filled default features.

    Parameters
    ----------
    source : Python source code string, or None.

    Returns
    -------
    dict mapping feature name → value.
    """
    if not source or not source.strip():
        logger.debug("Empty or None source — returning default AST features.")
        return _default_features()

    tree = _safe_parse(source)
    if tree is None:
        logger.debug("Unparseable source — returning default AST features.")
        return _default_features()

    features: dict = {}

    features["cyclomatic_complexity"] = compute_cyclomatic_complexity(source)
    features["cognitive_complexity"] = compute_cognitive_complexity(source, tree)
    features["max_nesting_depth"] = compute_max_nesting_depth(tree)
    features.update(compute_function_and_class_counts(tree))
    features.update(compute_function_length_and_args(tree))
    features["num_imports"] = compute_import_count(tree)
    features["ast_node_count"] = compute_ast_node_count(tree)
    features["has_try_except"] = compute_has_try_except(tree)

    return features