"""
Safe query execution over the CSV files of a task workspace.

The model writes ONE query; the wrapper runs it and turns the result into an
answer string. Two query languages are supported:

  sql    - SQLite in memory, one table per CSV (table name = file stem).
           An authorizer allows only reading and functions; one SELECT/WITH
           statement; a progress handler aborts runaway queries.
  pandas - one Python expression over DataFrames (`df` = the only/first CSV,
           plus one name per file stem). Only a whitelisted AST is evaluated:
           no imports, no dunders, no I/O methods (to_*/read_*), no eval/query.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

MAX_ROWS_PREVIEW = 3


DELIMS = [",", ";", "\t", "|"]
TABULAR = {".csv", ".tsv", ".psv", ".txt", ".json", ".jsonl", ".ndjson"}


def _delimiter(header: str) -> str | None:
    counts = {d: header.count(d) for d in DELIMS}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 1 else None


def table_name(p: Path) -> str:
    n = re.sub(r"\W", "_", p.stem)
    return n if not n[:1].isdigit() else f"t_{n}"


def _records(p: Path) -> list | None:
    text = p.read_text(errors="replace").strip()
    if p.suffix in (".jsonl", ".ndjson"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, dict):  # {"orders": [...]} or a single record
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        data = lists[0] if len(lists) == 1 else [data]
    return data if isinstance(data, list) and data and isinstance(data[0], dict) else None


def _frames_of(p: Path) -> dict[str, pd.DataFrame]:
    """One file -> its tables. JSON nested lists of records become child tables."""
    name = table_name(p)
    if p.suffix in (".json", ".jsonl", ".ndjson"):
        recs = _records(p)
        if recs is None:
            return {}
        out = {}
        nested = {k for r in recs for k, v in r.items()
                  if isinstance(v, list) and v and isinstance(v[0], dict)}
        flat = [{k: v for k, v in r.items() if k not in nested} for r in recs]
        parent = pd.json_normalize(flat, sep="_")
        out[name] = parent
        key = next((c for c in ("id", f"{p.stem}_id", "order_id", "key") if c in parent.columns), None)
        for k in sorted(nested):
            rows = []
            for i, r in enumerate(recs):
                for child in r.get(k) or []:
                    if isinstance(child, dict):
                        rows.append({f"{name}_{key or 'row'}": r.get(key) if key else i, **child})
            out[f"{name}_{k}"] = pd.json_normalize(rows, sep="_")
        return out
    with p.open(errors="replace") as f:
        header = f.readline()
    sep = _delimiter(header)
    if sep is None:
        return {}
    return {name: pd.read_csv(p, sep=sep, engine="python")}


def data_files(ws: Path) -> list[Path]:
    return sorted(p for p in ws.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR)


def load_frames(ws: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in data_files(ws):
        try:
            out.update(_frames_of(p))
        except (ValueError, json.JSONDecodeError, pd.errors.ParserError, UnicodeError):
            continue
    return out


def csv_files(ws: Path) -> list[Path]:  # kept for callers that only want delimited files
    return [p for p in data_files(ws) if p.suffix.lower() not in (".json", ".jsonl", ".ndjson")]


def preview(ws: Path) -> str:
    """Columns + first rows of each table, as compact as possible."""
    out = []
    for name, df in load_frames(ws).items():
        out.append(f"{name} ({len(df)} rows):\n" + df.head(MAX_ROWS_PREVIEW).to_csv(index=False).strip())
    return "\n\n".join(out)


def strip_code(reply: str) -> str:
    m = re.search(r"```(?:\w+)?\n(.*?)```", reply, re.S)
    text = (m.group(1) if m else reply).strip()
    text = re.sub(r"^(sql|pandas|python|query|answer)\s*:\s*", "", text, flags=re.I)
    return text.strip().rstrip(";").strip()


def to_answer(v) -> str | None:
    """Normalize a query result to the answer string."""
    if isinstance(v, pd.DataFrame):
        if v.empty:
            return None
        v = v.iloc[0, 0] if v.shape[1] >= 1 else None
    if isinstance(v, pd.Series):
        if v.empty:
            return None
        v = v.iloc[0] if len(v) == 1 else v.iloc[0]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6g}" if abs(v) < 1e6 else str(v)
    if isinstance(v, (int, str)):
        return str(v).strip()
    if isinstance(v, (list, tuple)) and v:
        return to_answer(v[0])
    return str(v).strip()


# ------------------------------------------------------------------ SQL
_SQL_OK = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def run_sql(ws: Path, reply: str, timeout_s: float = 5.0) -> str | None:
    q = strip_code(reply)
    if not re.match(r"^\s*(select|with)\b", q, re.I) or ";" in q:
        raise ValueError("expected one SELECT statement")
    con = sqlite3.connect(":memory:")
    try:
        for name, df in load_frames(ws).items():
            df.to_sql(name, con, index=False)
        con.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in _SQL_OK else sqlite3.SQLITE_DENY)
        t0 = time.monotonic()
        con.set_progress_handler(lambda: int(time.monotonic() - t0 > timeout_s), 10000)
        row = con.execute(q).fetchone()
        if row is None:
            return None
        if row[0] is None and re.search(r"\b(sum|total|count)\s*\(", q, re.I):
            return "0"  # SQL aggregates over an empty selection give NULL, the answer is 0
        return to_answer(row[0])
    finally:
        con.close()


# ------------------------------------------------------------------ pandas
_ALLOWED_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.Call, ast.keyword,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Lambda, ast.arguments, ast.arg,
)
_BANNED_ATTR = re.compile(r"^(_|to_|read_|eval$|query$|pipe$|plot|hist$|style$|boxplot$|"
                          r"from_|save|dump|load|open|exec|compile|globals|locals|getattr|setattr|"
                          r"delattr|import|system|popen|remove|unlink)")
_SAFE_BUILTINS = {"len": len, "int": int, "float": float, "round": round, "sum": sum, "min": min,
                  "max": max, "abs": abs, "sorted": sorted, "str": str, "list": list, "set": set}


def _check(tree: ast.AST, names: set[str]) -> None:
    lambda_args = {a.arg for n in ast.walk(tree) if isinstance(n, ast.Lambda) for a in n.args.args}
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"not allowed: {type(n).__name__}")
        if isinstance(n, ast.Attribute) and _BANNED_ATTR.match(n.attr):
            raise ValueError(f"attribute not allowed: {n.attr}")
        if isinstance(n, ast.Name) and n.id not in names | lambda_args | set(_SAFE_BUILTINS) | {"pd"}:
            raise ValueError(f"unknown name: {n.id}")
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "__" in n.value:
            raise ValueError("dunder string")


class _PdProxy:
    """Only harmless pandas helpers are reachable as `pd.<name>`."""
    to_numeric = staticmethod(pd.to_numeric)
    isna = staticmethod(pd.isna)
    notna = staticmethod(pd.notna)
    concat = staticmethod(pd.concat)


def run_pandas(ws: Path, reply: str) -> str | None:
    expr = strip_code(reply)
    lines = [l for l in expr.splitlines() if l.strip() and not l.strip().startswith(("import ", "#"))]
    expr = lines[-1].strip() if lines else ""
    expr = re.sub(r"^\s*(result|answer|ans|res)\s*=\s*", "", expr)
    frames = load_frames(ws)
    if not frames:
        raise ValueError("no csv files")
    env = dict(frames)
    env["df"] = next(iter(frames.values()))
    tree = ast.parse(expr, mode="eval")
    _check(tree, set(env))
    env["pd"] = _PdProxy
    return to_answer(eval(compile(tree, "<pandas>", "eval"), {"__builtins__": _SAFE_BUILTINS}, env))


RUNNERS = {"sql": run_sql, "pandas": run_pandas}
