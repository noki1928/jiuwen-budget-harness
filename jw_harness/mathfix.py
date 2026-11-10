"""
Robust evaluation of a model's math reply (expression or short program).

The wrapper, not the model, is responsible for syntax: unbalanced
parentheses, unicode operators, "x = ..." prefixes, trailing "= 123",
code fences and prose lines are repaired here. Only a whitelisted subset
of Python AST is executed.
"""
from __future__ import annotations

import ast
import math
import re

ALLOWED_CALLS = {"int", "min", "max", "abs", "round", "ceil", "floor", "sum", "pow"}
ALLOWED = (ast.Module, ast.Expression, ast.Expr, ast.Assign, ast.AugAssign, ast.Name,
           ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call,
           ast.Attribute, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
           ast.USub, ast.UAdd,
           ast.If, ast.IfExp, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not,
           ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
UNICODE = {"×": "*", "·": "*", "÷": "//", "−": "-", "–": "-", "^": "**", " ": " "}
EXPR_LINE = re.compile(r"^[\s\w().+\-*/%,=\[\]'\"]*\d[\s\w().+\-*/%,=\[\]'\"]*$")


def _normalize(line: str) -> str:
    for a, b in UNICODE.items():
        line = line.replace(a, b)
    line = line.strip().rstrip(".;").strip("`").strip()
    line = re.sub(r"(?<=\d),(?=\d{3}\b)", "", line)          # 1,234 -> 1234
    line = re.sub(r"^(print|int)\s*\((.*)\)$", r"\2", line)   # print(expr) -> expr
    return line


def _balance_variants(e: str) -> list[str]:
    opens, closes = e.count("("), e.count(")")
    out = []
    if opens > closes:
        out.append(e + ")" * (opens - closes))
    elif closes > opens:
        out.append("(" * (closes - opens) + e)
    # drop unmatched closing parens, then close what is left open
    depth, buf = 0, []
    for ch in e:
        if ch == ")":
            if depth == 0:
                continue
            depth -= 1
        elif ch == "(":
            depth += 1
        buf.append(ch)
    out.append("".join(buf) + ")" * depth)
    return [e] + out


def _safe(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if not isinstance(n, ALLOWED):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name not in ALLOWED_CALLS:
                return False
        if isinstance(n, ast.Attribute) and not (isinstance(n.value, ast.Name) and n.value.id == "math"):
            return False
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            if not (isinstance(n.right, ast.Constant) and abs(n.right.value) <= 10):
                return False
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "pow" and len(n.args) != 3:
            return False  # only modular pow(a, b, m): bounded result
    return True


ENV = {"__builtins__": {}, "pow": pow, "int": int, "min": min, "max": max, "abs": abs, "round": round,
       "sum": sum, "ceil": math.ceil, "floor": math.floor, "math": math}


def _as_int(v) -> str | None:
    if isinstance(v, str):  # word answers, e.g. a weekday picked from a list
        v = v.strip()
        return v if 0 < len(v) <= 60 else None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        if abs(v - round(v)) > 1e-9:
            return str(int(v))  # truncation, same as int()
        v = round(v)
    return str(int(v))


def eval_expr(e: str) -> str | None:
    e = re.sub(r"^\s*[A-Za-z_]\w*\s*=(?!=)\s*", "", e)  # "answer = ..." -> "..."
    e = re.sub(r"\s*=\s*-?\d+(\.\d+)?\s*$", "", e)      # "... = 389" -> "..."
    for cand in _balance_variants(e):
        try:
            tree = ast.parse(cand, mode="eval")
        except SyntaxError:
            continue
        if not _safe(tree):
            continue
        try:
            return _as_int(eval(compile(tree, "<e>", "eval"), dict(ENV)))
        except Exception:  # noqa: BLE001
            continue
    return None


def eval_program(lines: list[str], parsed: ast.Module | None = None) -> str | None:
    if parsed is not None:
        return _run(parsed)
    fixed = []
    for l in lines:
        for cand in _balance_variants(l):
            try:
                ast.parse(cand)
                fixed.append(cand)
                break
            except SyntaxError:
                continue
    if not fixed:
        return None
    src = "\n".join(fixed)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if not _safe(tree):
        return None
    return _run(tree)


def _run(tree: ast.Module) -> str | None:
    env = dict(ENV)
    try:
        exec(compile(tree, "<p>", "exec"), env)
    except Exception:  # noqa: BLE001
        return None
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        try:
            return _as_int(eval(compile(ast.Expression(last.value), "<l>", "eval"), env))
        except Exception:  # noqa: BLE001
            return None
    if "x" in env:
        return _as_int(env["x"])
    names = [t.id for n in tree.body for t in
             (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign) else [])
             if isinstance(t, ast.Name)]
    return _as_int(env.get(names[-1])) if names else None


def solve_reply(reply: str) -> str | None:
    text = re.sub(r"```(?:python)?", "\n", reply)
    # whole reply is a valid program (may contain if/else blocks): run it as is
    code = "\n".join(l for l in text.splitlines()
                     if not re.match(r"\s*(print\s*\(|import math|from math import)", l)).strip()
    if "\n" in code:
        try:
            tree = ast.parse(code)
            if _safe(tree) and any(isinstance(n, (ast.Assign, ast.AugAssign)) for n in tree.body):
                r = eval_program(code.splitlines(), parsed=tree)
                if r is not None:
                    return r
        except SyntaxError:
            pass
    lines = [_normalize(l) for l in text.splitlines()]
    lines = [l for l in lines if l and EXPR_LINE.match(l) and not l.startswith("#")]
    if not lines:
        return None
    if len(lines) > 1 and any(re.match(r"^[A-Za-z_]\w*\s*([-+*/%]|//)?=(?!=)", l) for l in lines):
        r = eval_program(lines)
        if r is not None:
            return r
    for l in reversed(lines):  # the last expression-looking line wins
        r = eval_expr(l)
        if r is not None:
            return r
    return None
