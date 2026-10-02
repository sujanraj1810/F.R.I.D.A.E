# executor.py

from __future__ import annotations

import ast
import builtins
import io
import math
import multiprocessing
import os
import statistics
from collections import Counter, defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from typing import Any, Callable

import json
import numpy as np
import pandas as pd

from config import MAX_CODE_CHARS, MAX_OUTPUT_CHARS, PY_TIMEOUT
from observability import log_event


# Generated Python is executed in a short-lived child process on Unix.  This
# gives us a real kill boundary for timeouts.  It is deliberately NOT called a
# security sandbox: the AST policy below is still the security boundary.

ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "io",
    "json",
    "math",
    "statistics",
    "re",
    "datetime",
    "collections",
}

# Only these public attributes may be reached from an imported module alias.
# In particular, this prevents gadgets such as bs4.sys.modules["os"] from
# being reachable.  The generated-code namespace already exposes the common
# modules directly, so generated code rarely needs imports at all.
MODULE_ATTRIBUTE_ALLOWLIST: dict[str, set[str]] = {
    "pandas": {
        "DataFrame", "Series", "Index", "Timestamp", "Timedelta", "Grouper",
        "Categorical", "CategoricalDtype", "NA", "NaT", "isna", "notna",
        "isnull", "notnull", "to_datetime", "to_numeric", "to_timedelta",
        "concat", "merge", "join", "crosstab", "pivot_table", "cut", "qcut",
        "get_dummies", "factorize", "unique", "value_counts", "read_csv",
        "read_json", "read_fwf", "read_table", "options",
    },
    "numpy": {
        "array", "asarray", "arange", "linspace", "zeros", "ones", "full",
        "eye", "mean", "median", "std", "var", "sum", "min", "max",
        "percentile", "quantile", "nanmean", "nanmedian", "nanstd", "nanvar",
        "nansum", "nanmin", "nanmax", "isnan", "isfinite", "where", "unique",
        "sort", "argsort", "corrcoef", "cov", "sqrt", "log", "exp", "abs",
        "pi", "e", "inf", "nan",
    },
    "io": {"StringIO", "BytesIO"},
    "json": {"loads", "dumps", "load", "dump"},
    "math": {
        "ceil", "floor", "sqrt", "log", "log10", "exp", "pow", "fabs",
        "sin", "cos", "tan", "pi", "e", "isfinite", "isnan",
    },
    "statistics": {
        "mean", "median", "mode", "multimode", "pstdev", "pvariance",
        "stdev", "variance", "quantiles",
    },
    "re": {"compile", "search", "match", "fullmatch", "findall", "finditer", "split", "sub", "escape"},
    "datetime": {"date", "datetime", "timedelta", "timezone"},
    "collections": {"Counter", "defaultdict", "deque"},
}

BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins",
    "importlib", "ctypes", "signal", "pickle", "marshal", "resource",
    "eval", "exec", "compile", "__import__", "input", "breakpoint", "help",
    "open", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr",
}

BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "input", "breakpoint", "help",
    "open", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "read_html", "read_excel", "read_sql", "read_sql_query",
    "read_sql_table", "read_parquet", "read_feather", "read_pickle", "read_orc",
    "read_hdf", "read_stata", "read_spss", "read_sas", "read_gbq",
}

BLOCKED_WRITE_CALLS = {
    "to_csv", "to_excel", "to_json", "to_pickle", "to_parquet", "to_feather",
    "to_orc", "to_hdf", "to_stata", "to_html", "to_xml", "to_clipboard",
}

BLOCKED_ATTRIBUTE_CALLS = {"loadtxt", "genfromtxt", "fromfile", "frombuffer"}

RESTRICTED_TO_BUFFER = {"read_csv", "read_json"}
BUFFER_CONSTRUCTORS = {"StringIO", "BytesIO"}

ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "len", "list", "map", "max", "min", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
}

DUUNDER_ALLOWLIST = {"__name__", "__doc__"}

SECRET_ENV_KEYS = {
    "BOT_TOKEN", "AIPIPE_TOKEN", "EVAL_TOKEN", "OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN",
}


class ValidationError(ValueError):
    """Raised when generated Python violates the execution policy."""


class ExecutionResult:
    def __init__(self, *, ok: bool, output: str = "", error: str | None = None, timed_out: bool = False) -> None:
        self.ok = bool(ok)
        self.output = output
        self.error = error
        self.timed_out = bool(timed_out)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "output": self.output, "error": self.error, "timed_out": self.timed_out}


class _CappedStringIO(io.StringIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = max(1024, int(limit))
        self._size = 0

    def write(self, s: str) -> int:
        if not s:
            return 0
        remaining = self.limit - self._size
        if remaining <= 0:
            return len(s)
        piece = s[:remaining]
        self._size += len(piece)
        super().write(piece)
        return len(s)


def _attribute_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                local = alias.asname or root
                aliases[local] = root
    return aliases


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if alias.name != root or root not in ALLOWED_IMPORTS:
                raise ValidationError(f"Import not allowed: {alias.name}")
    else:
        if node.level or not node.module:
            raise ValidationError("Relative imports are not allowed.")
        root = node.module.split(".", 1)[0]
        if node.module != root or root not in ALLOWED_IMPORTS:
            raise ValidationError(f"Import not allowed: {node.module}")
        allowed = MODULE_ATTRIBUTE_ALLOWLIST.get(root, set())
        for alias in node.names:
            if alias.name == "*" or alias.name not in allowed:
                raise ValidationError(f"Imported symbol not allowed: {node.module}.{alias.name}")


def _is_buffer_construction(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in BUFFER_CONSTRUCTORS
    chain = _attribute_chain(node.func)
    return bool(chain and chain[-1] in BUFFER_CONSTRUCTORS)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _validate_call(node: ast.Call) -> None:
    name = _call_name(node)
    if name in BLOCKED_CALLS or name in BLOCKED_WRITE_CALLS or name in BLOCKED_ATTRIBUTE_CALLS:
        raise ValidationError(f"Call not allowed: {name}")
    if name in RESTRICTED_TO_BUFFER:
        if not node.args or not _is_buffer_construction(node.args[0]):
            raise ValidationError(f"{name} is restricted to inline StringIO/BytesIO buffers.")


def validate_python(code: str) -> None:
    if not isinstance(code, str):
        raise ValidationError("Code must be a string.")
    if len(code) > MAX_CODE_CHARS:
        raise ValidationError("Generated code exceeds the size limit.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValidationError(f"Syntax error: {exc}") from exc

    aliases = _module_aliases(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                raise ValidationError(f"Name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            # No private/dunder attribute traversal. This closes common Python
            # object-introspection chains such as func.__globals__ and
            # obj.__class__.__subclasses__.
            if node.attr.startswith("_") and node.attr not in DUUNDER_ALLOWLIST:
                raise ValidationError(f"Private/dunder attribute not allowed: {node.attr}")

            chain = _attribute_chain(node)
            if chain and chain[0] in aliases:
                module = aliases[chain[0]]
                if len(chain) >= 2:
                    allowed = MODULE_ATTRIBUTE_ALLOWLIST.get(module, set())
                    if chain[1] not in allowed:
                        raise ValidationError(f"Module attribute not allowed: {module}.{chain[1]}")
        elif isinstance(node, ast.Call):
            _validate_call(node)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > MAX_CODE_CHARS:
                raise ValidationError("Oversized string literal.")


def _truncate_output(value: str) -> str:
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    suffix = "\n...[output truncated]"
    return value[: max(0, MAX_OUTPUT_CHARS - len(suffix))] + suffix


def _build_namespace(*, dataset_loader: Callable[[], pd.DataFrame] | None = None,
                     retrieval_helpers: dict[str, Any] | None = None,
                     artifact_helpers: dict[str, Any] | None = None) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "pd": pd, "np": np, "io": io, "json": json, "math": math,
        "statistics": statistics, "Counter": Counter, "defaultdict": defaultdict,
        "deque": deque, "date": date, "datetime": datetime, "timedelta": timedelta,
    }
    namespace["__builtins__"] = {
        name: getattr(builtins, name) for name in ALLOWED_BUILTINS if hasattr(builtins, name)
    }
    namespace["__builtins__"]["__import__"] = _safe_import
    if dataset_loader is not None:
        namespace["load_uploaded_dataset"] = dataset_loader
    if retrieval_helpers:
        namespace.update(retrieval_helpers)
    if artifact_helpers:
        namespace.update(artifact_helpers)
    return namespace


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("Relative imports are not allowed")
    root = name.split(".", 1)[0]
    if name != root or root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    module = __import__(name, globals, locals, fromlist or ["*"])
    if fromlist:
        allowed = MODULE_ATTRIBUTE_ALLOWLIST.get(root, set())
        for item in fromlist:
            if item == "*" or item not in allowed:
                raise ImportError(f"Imported symbol not allowed: {root}.{item}")
    return module


def _scrub_child_environment() -> None:
    for key in list(os.environ):
        upper = key.upper()
        if key in SECRET_ENV_KEYS or upper.endswith("_TOKEN") or "API_KEY" in upper or "SECRET" in upper:
            os.environ.pop(key, None)


def _execute_process(code: str, namespace: dict[str, Any], result_conn, rpc_recv, rpc_send, artifact_names: set[str]) -> None:
    _scrub_child_environment()
    output = _CappedStringIO(MAX_OUTPUT_CHARS)
    error_output = _CappedStringIO(MAX_OUTPUT_CHARS)
    request_id = 0

    def rpc_artifact(name: str):
        nonlocal request_id
        def _call(*args, **kwargs):
            nonlocal request_id
            request_id += 1
            rid = request_id
            rpc_send.send(("rpc", rid, name, args, kwargs))
            while True:
                message = rpc_recv.recv()
                if message[0] != "rpc_result" or message[1] != rid:
                    continue
                if message[2] is False:
                    raise RuntimeError(str(message[3]))
                return message[3]
        return _call

    for name in artifact_names:
        namespace[name] = rpc_artifact(name)

    try:
        with redirect_stdout(output), redirect_stderr(error_output):
            exec(code, namespace, namespace)
        stdout = output.getvalue()
        stderr = error_output.getvalue()
        if stderr:
            stdout = f"{stdout}\n{stderr}" if stdout else stderr
        result_conn.send(("result", ExecutionResult(ok=True, output=_truncate_output(stdout)).as_dict()))
    except Exception as exc:
        stdout = output.getvalue()
        stderr = error_output.getvalue()
        details = f"{type(exc).__name__}: {exc}"
        if stderr:
            details = f"{details}\n{stderr}"
        result_conn.send(("result", ExecutionResult(ok=False, output=_truncate_output(stdout), error=_truncate_output(details)).as_dict()))
    finally:
        result_conn.close()
        rpc_recv.close()
        rpc_send.close()


def _run_forked(code: str, namespace: dict[str, Any], timeout: float,
                artifact_helpers: dict[str, Any] | None = None) -> ExecutionResult:
    if "fork" not in multiprocessing.get_all_start_methods():
        return ExecutionResult(ok=False, error="Process-isolated execution requires a Unix host with fork support.")

    ctx = multiprocessing.get_context("fork")
    result_recv, result_send = ctx.Pipe(duplex=False)
    rpc_recv, rpc_send = ctx.Pipe(duplex=False)
    response_recv, response_send = ctx.Pipe(duplex=False)
    artifact_names = set(artifact_helpers or {})
    process = ctx.Process(
        target=_execute_process,
        args=(code, namespace, result_send, response_recv, rpc_send, artifact_names),
        daemon=True,
    )
    process.start()
    result_send.close()
    response_recv.close()

    deadline = __import__("time").monotonic() + max(0.1, float(timeout))
    final_payload = None
    try:
        while process.is_alive() and __import__("time").monotonic() < deadline:
            if rpc_recv.poll(0.02):
                try:
                    message = rpc_recv.recv()
                    if message[0] == "rpc":
                        _, rid, name, args, kwargs = message
                        helper = (artifact_helpers or {}).get(name)
                        if helper is None:
                            response_send.send(("rpc_result", rid, False, f"Artifact helper not available: {name}"))
                        else:
                            try:
                                value = helper(*args, **kwargs)
                                response_send.send(("rpc_result", rid, True, value))
                            except Exception as exc:
                                response_send.send(("rpc_result", rid, False, f"{type(exc).__name__}: {exc}"))
                except (EOFError, OSError):
                    break
            if result_recv.poll(0):
                try:
                    message = result_recv.recv()
                    if message[0] == "result":
                        final_payload = message[1]
                        break
                except (EOFError, OSError):
                    break
            __import__("time").sleep(0.005)

        if final_payload is None and result_recv.poll(0.05):
            try:
                message = result_recv.recv()
                if message[0] == "result":
                    final_payload = message[1]
            except (EOFError, OSError):
                pass

        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join(1.0)
            return ExecutionResult(ok=False, error=f"Execution timed out after {timeout} seconds.", timed_out=True)

        if isinstance(final_payload, dict):
            return ExecutionResult(**final_payload)
        return ExecutionResult(ok=False, error=f"Execution process exited without producing a result (exit code {process.exitcode}).")
    finally:
        for conn in (result_recv, rpc_recv, response_send):
            try:
                conn.close()
            except Exception:
                pass

def run_python(code: str, *, dataset_loader: Callable[[], pd.DataFrame] | None = None,
               retrieval_helpers: dict[str, Any] | None = None,
               artifact_helpers: dict[str, Any] | None = None,
               timeout: float = PY_TIMEOUT, timeout_seconds: float | None = None,
               extra_globals: dict[str, Any] | None = None) -> ExecutionResult:
    try:
        validate_python(code)
    except ValidationError as exc:
        result = ExecutionResult(ok=False, error=f"ValidationError: {exc}")
        log_event("python_validation_error", error=str(exc))
        return result

    if timeout_seconds is not None:
        timeout = timeout_seconds

    merged_retrieval = dict(retrieval_helpers or {})
    merged_artifacts = dict(artifact_helpers or {})
    if extra_globals:
        for key, value in extra_globals.items():
            if key == "load_uploaded_dataset":
                dataset_loader = value
            elif key in {"fetch_url", "fetch_soup", "fetch_table", "fetch_csv", "fetch_excel", "fetch_json"}:
                merged_retrieval[key] = value
            else:
                merged_artifacts[key] = value

    namespace = _build_namespace(dataset_loader=dataset_loader, retrieval_helpers=merged_retrieval, artifact_helpers=merged_artifacts)
    log_event("python_execution_start", code_chars=len(code), timeout=timeout)

    result = _run_forked(code, namespace, timeout, artifact_helpers=merged_artifacts)
    log_event("python_execution_timeout" if result.timed_out else "python_execution_complete",
              timeout=timeout if result.timed_out else None, ok=result.ok, error=result.error,
              output_chars=len(result.output))
    return result
