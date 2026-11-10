"""
Task solvers. The model never gets a shell for coding/math: the wrapper runs
the tests, builds a minimal prompt, parses the reply, applies it, re-checks and
writes answer.txt itself. Other task types go to a small capped agent loop.
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import textwrap
import sys
from pathlib import Path

from .llm import CHEAP, LLM, SECOND
from .mathfix import solve_reply

# ---------------------------------------------------------------- prompts
# Measured on 104 tasks (bench/): see README for the numbers behind each choice.
SYS_LINE = "Fix the bugs. Output only changed lines as N: code (indent auto). Nothing else."
SYS_LINE_MULTI = "Fix the bugs. Output only changed lines as file:N: code (indent auto). Nothing else."
SYS_FUNC = ("You fix bugs in Python. Reply ONLY with the corrected function definitions "
            "in one ```python block. No prose.")
SYS_STEPS = (
    "Translate the word problem into Python statements, ONE line per sentence, "
    "updating variable x. Do NOT compute anything yourself. Use // for rounding down, "
    "ceil(a/b) for rounding up, % for remainder.\n"
    "Percent loss p%: x -= x*p//100; percent growth: x += x*p//100. "
    "The final value must be exactly the quantity asked in the question. "
    "The last line must set x to the answer (a string if the answer is a word, e.g. a weekday).\n"
    "Example:\nx = 1000\nx -= 3*20\nx //= 2\nx -= x*10//100\n"
    "Reply ONLY with the code."
)
SYS_AGENT = (
    "You solve a task in a sandbox directory. Each turn reply with ONE json object only:\n"
    '{"cmd": "<bash command>"} to run a command (output is truncated), or\n'
    '{"answer": "<final answer>"} when done (for edit tasks answer DONE after checking).\n'
    'If the task cannot be solved from the available data, reply {"answer": "IMPOSSIBLE"}.\n'
    "Prefer one python3 -c / grep command that computes the answer directly. No network, no installs."
)


# ---------------------------------------------------------------- helpers
class Result:
    def __init__(self, ok: bool, answer: str = "", cost: float = 0.0, note: str = ""):
        self.ok, self.answer, self.cost, self.note = ok, answer, cost, note


def write_answer(task_dir: Path, answer: str) -> None:
    """answer.txt goes to the task dir and, if present, to its workspace (the task's cwd)."""
    for d in (task_dir, task_dir / "workspace"):
        if d.is_dir():
            (d / "answer.txt").write_text(answer.strip() + "\n")


def same_answer(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a.strip().casefold() == b.strip().casefold()


async def cross_check(first: str, alternatives: list, log: list) -> tuple[str, float]:
    """Second independent answer; on disagreement a third one decides by majority.

    `alternatives` are async callables returning (answer | None, cost). Returns the
    chosen answer (the first one when nothing agrees) and the extra cost.
    """
    cost, seen = 0.0, [first]
    for alt in alternatives[:2]:
        ans, c = await alt()
        cost += c
        if same_answer(ans, first):
            log.append({"verify": "agree", "answer": first})
            return first, cost
        if any(same_answer(ans, x) for x in seen[1:]):
            log.append({"verify": "majority", "answer": ans, "first": first})
            return ans, cost
        seen.append(ans)
    log.append({"verify": "no majority, keep first", "answers": seen})
    return first, cost


def instruction_core(instr: str) -> str:
    """Drop the shared boilerplate tail (tools hint, answer.txt rules)."""
    return re.split(r"\s(?:Compute it with|Use bash|Use the file tools|Work only inside)", instr)[0]


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


# ---------------------------------------------------------------- coding
TEST_FILE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$")


def coding_target(task: dict, ws: Path) -> tuple[str, list[str]] | None:
    """(test file, source files to fix) parsed from the instruction, or None if not this task shape.

    "Fix the bugs in mylib.py" names the file; otherwise every non-test .py file in the
    workspace is a candidate (packages like shop/cart.py + shop/pricing.py).
    """
    instr = task.get("instruction", "")
    m = re.search(r"python3?\s+([\w./-]+\.py)", instr)
    if not m or not (ws / m.group(1)).is_file():
        return None
    test = m.group(1)
    m = re.search(r"bugs in\s+([\w./-]+\.py)", instr)
    if m and (ws / m.group(1)).is_file():
        return test, [m.group(1)]
    libs = sorted(str(p.relative_to(ws)) for p in ws.rglob("*.py")
                  if not TEST_FILE.search(str(p.relative_to(ws))) and "__pycache__" not in p.parts)
    return (test, libs) if libs else None


_CONST_NODES = (ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.BinOp, ast.UnaryOp,
                ast.IfExp, ast.Compare, ast.BoolOp, ast.Load, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


def _const_value(node: ast.AST):
    """Value of an expected-value expression built from literals only (`[1, 3] * 2`, `x if False else y`)."""
    expr = ast.Expression(node)
    if not all(isinstance(n, _CONST_NODES) for n in ast.walk(expr)):
        raise ValueError("not a constant expression")
    if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(expr)):
        raise ValueError("pow")
    return eval(compile(expr, "<expected>", "eval"), {"__builtins__": {}}, {})


def _pure_call(node: ast.AST, modules: set[str]) -> bool:
    """module.func(<literals>): the same expression always means the same result.

    Attributes of objects (inv.items['nut']) or calls on instances change between asserts,
    so they can not prove a contradiction.
    """
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    while isinstance(f, ast.Attribute):
        f = f.value
    if not (isinstance(f, ast.Name) and f.id in modules):
        return False
    try:
        for a in list(node.args) + [k.value for k in node.keywords]:
            _const_value(a)
    except (ValueError, TypeError, SyntaxError, ArithmeticError):
        return False
    return True


def contradictory_tests(ws: Path, test: str) -> str:
    """The same pure call is expected to give different values -> no fix can pass all tests."""
    try:
        tree = ast.parse((ws / test).read_text())
    except (SyntaxError, OSError):
        return ""
    modules = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            modules |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            modules |= {a.asname or a.name for a in n.names if a.name[:1].islower() and "." not in a.name}
    seen: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args):
            continue
        attr = n.func.attr
        try:
            if attr == "assertEqual" and len(n.args) >= 2:
                a, b = n.args[0], n.args[1]
                if not _pure_call(a, modules):
                    a, b = b, a
                if _pure_call(a, modules):
                    seen.setdefault(ast.unparse(a), set()).add(repr(_const_value(b)))
            elif attr in ("assertTrue", "assertFalse") and _pure_call(n.args[0], modules):
                seen.setdefault(f"bool({ast.unparse(n.args[0])})", set()).add(repr(attr == "assertTrue"))
        except (ValueError, TypeError, SyntaxError, ArithmeticError):
            continue
    bad = [k for k, v in seen.items() if len(v) > 1]
    return f"contradictory asserts for {bad[0]}" if bad else ""


def run_tests(ws: Path, test: str) -> str:
    """'' when the tests pass, otherwise the report."""
    p = _run([sys.executable, test], ws)
    if p is None:
        return "TIMEOUT"
    return "" if p.returncode == 0 else (p.stdout + p.stderr)


def units(srcs: dict[str, str]) -> list[tuple[str, str, int, int]]:
    """(file, name, first line, last line) of every function and method."""
    out = []
    for f, src in srcs.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = n.decorator_list[0].lineno if n.decorator_list else n.lineno
                out.append((f, n.name, start, n.end_lineno))
    return out


def failing(report: str, srcs: dict[str, str]) -> tuple[list[str], list[tuple[str, str, int, int]]]:
    """Short failure descriptions and the code units to show."""
    us = units(srcs)
    picked: list = []

    def add(u):
        if u not in picked:
            picked.append(u)

    descs = []
    blocks = re.split(r"^={10,}$", report, flags=re.M)[1:] or [report]
    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        call = next((l.strip() for l in lines if "self.assert" in l or l.strip().startswith("assert ")), "")
        err = next((l.strip() for l in reversed(lines) if re.match(r"[\w.]*(Error|Exception)\b", l.strip())), "")
        descs.append(f"{call}  -> {err}" if call else err)
        # frames inside the sources: the innermost unit around the reported line
        for fpath, line in re.findall(r'File "([^"]+)", line (\d+)', block):
            for f in srcs:
                if fpath.endswith("/" + f) or fpath == f:
                    inside = [u for u in us if u[0] == f and u[2] <= int(line) <= u[3]]
                    if inside:
                        add(min(inside, key=lambda u: u[3] - u[2]))
        for name in re.findall(r"\b(\w+)\(", call):
            for u in us:
                if u[1] == name:
                    add(u)
    if not picked and report:  # import error / unknown failure shape: show everything (bounded)
        picked = us[:12]
        descs = [d for d in descs if d] or report.strip().splitlines()[-4:]
    return [d for d in descs if d][:8], picked


def coding_prompt(mode: str, srcs: dict[str, str], descs: list[str], picked: list) -> str:
    many = len({u[0] for u in picked}) > 1
    chunks = []
    for f in dict.fromkeys(u[0] for u in picked):
        lines = srcs[f].splitlines()
        body = []
        for u in sorted((u for u in picked if u[0] == f), key=lambda u: u[2]):
            if any(o is not u and o[0] == f and o[2] <= u[2] and u[3] <= o[3] for o in picked):
                continue  # nested inside another shown unit
            rng = range(u[2], u[3] + 1)
            body.append("\n".join((f"{i}: {lines[i - 1]}" if mode == "line" else lines[i - 1]) for i in rng))
        chunks.append((f"# file: {f}\n" if many else "") + "\n\n".join(body))
    return "Failing:\n" + "\n".join(descs) + "\n\nCode:\n" + "\n\n".join(chunks)


def apply_line(srcs: dict[str, str], reply: str, shown: list | None = None) -> dict[str, str]:
    """Apply 'N: code' / 'file:N: code' replacements; every touched file must still compile."""
    files = list(srcs)
    shown_files = list(dict.fromkeys(u[0] for u in (shown or []))) or files
    lines = {f: srcs[f].splitlines() for f in files}
    for m in re.finditer(r"^\s*(?:([\w./-]+\.py)\s*:\s*)?(\d+)\s*[:|]\s?(.*)$", reply, flags=re.M):
        f, i, code = m.group(1), int(m.group(2)), m.group(3).strip()
        if f:
            f = next((x for x in files if x == f or x.endswith("/" + f) or f.endswith("/" + x)), None)
        elif len(shown_files) == 1:
            f = shown_files[0]
        else:  # several files shown, no file given: the shown unit that contains the line
            cand = {u[0] for u in (shown or []) if u[2] <= i <= u[3] + 1}
            f = cand.pop() if len(cand) == 1 else None
        if f is None or not (1 <= i <= len(lines[f])):
            continue
        ls = lines[f]
        old = ls[i - 1]
        if old.strip():
            # indentation is always taken from the original line: models miscount spaces
            ls[i - 1] = old[:len(old) - len(old.lstrip())] + code
        elif code:
            # the model grew a function into the blank line after it: use the body's indent,
            # keep the blank separator
            prev = next((l for l in reversed(ls[:i - 1]) if l.strip()), "")
            indent = prev[:len(prev) - len(prev.lstrip())] or "    "
            ls[i - 1] = indent + code + "\n"
    out = dict(srcs)
    for f in files:
        new = "\n".join(lines[f]) + "\n"
        if new.strip() == srcs[f].strip():
            continue
        try:
            compile(new, f, "exec")  # ast.parse misses e.g. 'return' outside a function
        except SyntaxError:
            continue
        out[f] = new
    return out


def apply_func(srcs: dict[str, str], reply: str, shown: list) -> dict[str, str]:
    """Replace the shown functions/methods by the definitions in the reply (re-indented)."""
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = textwrap.dedent(m.group(1) if m else reply)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return srcs
    code_lines = code.splitlines()
    new_defs = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = n.decorator_list[0].lineno if n.decorator_list else n.lineno
            new_defs.setdefault(n.name, textwrap.dedent("\n".join(code_lines[start - 1:n.end_lineno])))
    out = dict(srcs)
    for f in dict.fromkeys(u[0] for u in shown):
        ls = out[f].splitlines()
        for (_, name, a, b) in sorted((u for u in shown if u[0] == f), key=lambda u: -u[2]):
            if name not in new_defs:
                continue
            indent = ls[a - 1][:len(ls[a - 1]) - len(ls[a - 1].lstrip())]
            ls[a - 1:b] = textwrap.indent(new_defs[name], indent).splitlines()
        new = "\n".join(ls) + "\n"
        try:
            compile(new, f, "exec")
            out[f] = new
        except SyntaxError:
            pass
    return out


async def solve_coding(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    ws = task_dir / "workspace"
    tgt = coding_target(task, ws)
    if tgt is None:
        return None  # not the "fix the code until tests pass" shape: let the agent handle it
    test, libs = tgt
    cost = 0.0
    # 3 cheap line-edit rounds, then one full-function rewrite by the second model
    plan = [(CHEAP, "line")] * 3 + [(SECOND, "func")]
    cap = max(4 * task.get("_est", 100), 200)
    prev = None
    for model, mode in plan:
        if cost > cap:
            break
        report = await asyncio.to_thread(run_tests, ws, test)
        if not report:
            break
        srcs = {f: (ws / f).read_text() for f in libs}
        descs, picked = failing(report, srcs)
        if not picked:
            break
        if prev == descs and mode == "line":
            continue  # no progress with the cheap model: jump to the escalation step
        prev = descs
        prompt = coding_prompt(mode, srcs, descs, picked)
        system = SYS_LINE if len({u[0] for u in picked}) == 1 else SYS_LINE_MULTI
        reply, c = await llm.ask(system if mode == "line" else SYS_FUNC, prompt, model=model,
                                 max_tokens=300 if mode == "line" else 500)
        cost += c
        log.append({"model": model, "mode": mode, "prompt": prompt, "reply": reply, "cost": c})
        new = apply_line(srcs, reply, picked) if mode == "line" else apply_func(srcs, reply, picked)
        for f, txt in new.items():
            if txt != srcs[f]:
                (ws / f).write_text(txt)
    ok = await asyncio.to_thread(run_tests, ws, test) == ""
    if ok:
        write_answer(task_dir, "DONE")
    return Result(ok, "DONE" if ok else "", cost)


# ---------------------------------------------------------------- math
def math_missing_numbers(instr: str) -> bool:
    """The first sentence (the starting state) has no number, e.g. 'A library owns some books.'"""
    first = re.split(r"(?<=[.?!])\s", instruction_core(instr), maxsplit=1)[0]
    return not re.search(r"\d", first)


def is_math_word_problem(task: dict) -> bool:
    return task.get("task_type") == "math" and not (Path(task["_dir"]) / "workspace").is_dir()


SYS_EXPR = ("Translate the word problem into ONE Python expression, do NOT compute it yourself. "
            "Use // for rounding down, ceil(a/b) for rounding up, % for remainder. Reply ONLY with the expression.")


async def _math_try(llm: LLM, q: str, model: str, system: str, log: list) -> tuple[str | None, float, str]:
    """(answer, cost, problem) for one translation of the question into code."""
    reply, c = await llm.ask(system, q, model=model, max_tokens=300)
    ans = solve_reply(reply)
    log.append({"model": model, "prompt": q, "reply": reply, "answer": ans, "cost": c})
    start = re.search(r"^\s*x\s*=\s*(-?\d+)\s*$", reply, flags=re.M)
    if start and start.group(1) == "1000" and "1000" not in re.findall(r"\d+", q):
        # "x = 1000" copied from the prompt example: the question has no starting value.
        # Other constants (365, 7, 0) are legitimate parts of a translation.
        return None, c, "start value copied from the prompt example"
    if ans is None or ans.startswith("-"):
        return None, c, "no evaluable program"
    return ans, c, ""


async def solve_math(llm: LLM, task_dir: Path, task: dict, log: list) -> Result:
    q = instruction_core(task["instruction"])
    cost, ans, problem = 0.0, None, ""
    for model in (CHEAP, SECOND):
        ans, c, problem = await _math_try(llm, q, model, SYS_STEPS, log)
        cost += c
        if ans is not None or problem.startswith("start value copied"):
            break
    if ans is None:
        return Result(False, "", cost, problem)
    if task.get("_verify"):
        async def alt(model, system):
            a, c, _ = await _math_try(llm, q, model, system, log)
            return a, c
        ans, c = await cross_check(ans, [lambda: alt(SECOND, SYS_STEPS), lambda: alt(CHEAP, SYS_EXPR)], log)
        cost += c
    write_answer(task_dir, ans)
    return Result(True, ans, cost)


# ---------------------------------------------------------------- data_analysis
# One query through a safe executor (dataq.py). Measured on 62 tasks: SQL 61/62 at ~33
# units, pandas 60-62/62 at ~37, the agent loop 61/62 at ~214. Failures of the two
# models never overlapped, so the second step uses the other model and language.
SYS_QUERY = {
    "sql": "Reply with one SQLite SELECT query that answers the question. Tables are shown below.",
    "pandas": "Reply with one pandas expression that answers the question. "
              "DataFrames are shown below by name; df is the first one.",
}


def data_question(instr: str) -> str:
    return re.sub(r"^The file \S+ has columns \S+ \(\d+ rows\)\.\s*", "", instruction_core(instr))


def has_csv(ws: Path) -> bool:
    """The workspace has tables the query executors can load (csv/tsv/psv/json/jsonl)."""
    if not ws.is_dir():
        return False
    from . import dataq
    try:
        return bool(dataq.load_frames(ws))
    except Exception:  # noqa: BLE001
        return False


async def _data_try(llm: LLM, ws: Path, user: str, model: str, mode: str, log: list):
    """(answer, error, cost): one query, one retry with the execution error."""
    from . import dataq  # pandas import is heavy: only when a data task shows up
    msgs = [{"role": "user", "content": user}]
    cost, err, ans = 0.0, "", None
    for _ in range(2):
        reply, c = await llm.ask(SYS_QUERY[mode], msgs, model=model, max_tokens=300)
        cost += c
        err, ans = "", None
        try:
            ans = await asyncio.to_thread(dataq.RUNNERS[mode], ws, reply)
            if ans is None:
                err = "query returned no value"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
        if not err and mode == "sql" and not re.search(r"\bfrom\b", reply, re.I):
            err = "constant query"  # "SELECT 0": the model gave up on the data
        log.append({"model": model, "mode": mode, "reply": reply, "answer": ans, "error": err, "cost": c})
        if not err or err in ("query returned no value", "constant query") or _missing_column(err, ws):
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Error: {err}\nReply with a corrected query only."}]
    return (None if err else ans), err, cost


async def solve_data(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    from . import dataq
    ws = task_dir / "workspace"
    user = f"{dataq.preview(ws)}\n\nQ: {data_question(task['instruction'])}"
    cost, ans = 0.0, None
    for model, mode in ((CHEAP, "sql"), (SECOND, "pandas")):
        ans, err, c = await _data_try(llm, ws, user, model, mode, log)
        cost += c
        if ans is not None:
            break
        if err in ("query returned no value", "constant query") or _missing_column(err, ws):
            return Result(False, "", cost, f"infeasible: {err}")
    if ans is None:
        return None  # let the agent loop try
    if task.get("_verify"):
        async def alt(model, mode):
            a, _, c = await _data_try(llm, ws, user, model, mode, log)
            return a, c
        ans, c = await cross_check(ans, [lambda: alt(SECOND, "pandas"), lambda: alt(SECOND, "sql")], log)
        cost += c
    write_answer(task_dir, ans)
    return Result(True, ans, cost)


def _missing_column(err: str, ws: Path) -> bool:
    """'no such column: discount' / KeyError 'discount' for a name that is in no loaded table."""
    m = re.search(r"no such column: (\w+)|KeyError: \W*(\w+)", err)
    if not m:
        return False
    from . import dataq
    col = (m.group(1) or m.group(2)).lower()
    cols = {str(c).lower() for df in dataq.load_frames(ws).values() for c in df.columns}
    return col not in cols


# ---------------------------------------------------------------- search
# The kb is parsed into an entity graph by the wrapper (kbgraph.py). Measured on 62+25 tasks
# (1-3 hops): path query 87/87 (35b) at ~21 units, subgraph 85/87 (35b; misses 3-hop chains),
# whole kb 49/62 at ~89, the agent loop 48/55 at ~294.
SYS_SUBGRAPH = "Answer from the facts below with the value only. If they do not contain the answer, reply IMPOSSIBLE."
SYS_PATH = ("Reply with a path that answers the question: the name from the question, then fields from the list. "
            "Example: Anna Kim > Employer > Headquarters. If the data cannot answer it, reply IMPOSSIBLE.")


AGGREGATE = re.compile(r"\b(how many|number of|count|oldest|youngest|earliest|latest|most|least|largest|"
                       r"smallest|first|last|total|average|highest|lowest)\b", re.I)


def search_question(instr: str) -> str:
    return re.sub(r"^The directory \S+ contains [^.]*\.\s*", "", instruction_core(instr))


def load_kb(ws: Path):
    from . import kbgraph
    return kbgraph.load(ws) if ws.is_dir() else None


def _value(reply: str) -> str | None:
    v = reply.strip().strip("`*\"'. ").splitlines()[0].strip() if reply.strip() else ""
    v = re.sub(r"^(answer|value)\s*:\s*", "", v, flags=re.I)
    return None if not v or v.upper().startswith("IMPOSSIBLE") else v


async def _path_try(llm: LLM, kb, q: str, model: str, log: list) -> tuple[str | None, float]:
    ment = kb.mentioned(q)
    msgs = [{"role": "user", "content": f"Fields:\n{kb.schema()}\n\nQ: {q}"}]
    cost = 0.0
    for _ in range(2):  # one retry with the lookup error
        reply, c = await llm.ask(SYS_PATH, msgs, model=model, max_tokens=60)
        cost += c
        try:
            ans, err = kb.follow(reply, ment[0] if len(ment) == 1 else None), ""
        except LookupError as e:
            ans, err = None, str(e)
        log.append({"model": model, "mode": "path", "reply": reply, "answer": ans, "error": err, "cost": c})
        if ans or err == "IMPOSSIBLE":
            return ans, cost
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Error: {err}\nReply with a corrected path or IMPOSSIBLE."}]
    return None, cost


async def _subgraph_try(llm: LLM, kb, q: str, model: str, log: list) -> tuple[str | None, float]:
    facts = kb.subgraph(q)
    if not facts:
        return None, 0.0
    reply, c = await llm.ask(SYS_SUBGRAPH, f"{facts}\n\nQ: {q}", model=model, max_tokens=60)
    ans = _value(reply)
    log.append({"model": model, "mode": "subgraph", "reply": reply, "answer": ans, "cost": c})
    return ans, c


async def solve_search(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    kb = load_kb(task_dir / "workspace")
    if kb is None:
        return None  # not a field-based kb: the agent loop handles it
    q = search_question(task["instruction"])
    if not kb.mentioned(q):
        # the entity is not a page of the graph (e.g. facts written as prose in one wiki
        # file); recon already skipped names absent from the raw text, so read the files
        return None
    ans, cost = await _path_try(llm, kb, q, CHEAP, log)
    if not ans:  # second opinion with the other model and the other view before giving up
        ans, c = await _subgraph_try(llm, kb, q, SECOND, log)
        cost += c
    if not ans and AGGREGATE.search(q) and len(kb.whole()) <= 8000:
        # "oldest", "how many": no single path answers it, the whole small kb does
        reply, c = await llm.ask(SYS_SUBGRAPH, f"{kb.whole()}\n\nQ: {q}", model=CHEAP, max_tokens=60)
        cost += c
        ans = _value(reply)
        log.append({"model": CHEAP, "mode": "whole", "reply": reply, "answer": ans, "cost": c})
    if not ans:
        return Result(False, "", cost, "kb has no answer")
    if task.get("_verify"):
        ans, c = await cross_check(ans, [lambda: _subgraph_try(llm, kb, q, SECOND, log),
                                         lambda: _path_try(llm, kb, q, SECOND, log)], log)
        cost += c
    write_answer(task_dir, ans)
    return Result(True, ans, cost)


# ---------------------------------------------------------------- generic agent
def workspace_preview(ws: Path, budget: int = 3500) -> str:
    """File list + text files: whole when they are small (a 3 KB wiki page costs ~100 units
    and saves the steps of reading it), otherwise their first lines."""
    if not ws.is_dir():
        return "(no files)"
    files = sorted(p for p in ws.rglob("*") if p.is_file())
    listing = "\n".join(str(p.relative_to(ws)) for p in files[:60])
    out = ["Files:\n" + listing + ("\n..." if len(files) > 60 else "")]
    texts = []
    for p in files:
        if p.suffix not in (".csv", ".md", ".txt", ".json", ".jsonl", ".py", ".tsv", ".log", ".yaml", ".yml", ".ini"):
            continue
        try:
            texts.append((p, p.read_text(errors="replace")))
        except OSError:
            continue
    whole = sum(len(t) for _, t in texts) <= budget
    used = len(out[0])
    for p, t in texts:
        if used > budget:
            break
        body = t if whole else "\n".join(t.splitlines()[:5])[:400]
        chunk = f"\n--- {'' if whole else 'head '}{p.relative_to(ws)} ---\n{body}"
        out.append(chunk)
        used += len(chunk)
    return "".join(out)


def parse_act(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for m in re.finditer(r"\{.*\}", t, re.S):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("cmd" in obj or "answer" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None


BANNED = re.compile(r"(\bsudo\b|pip\s+install|\bcurl\b|\bwget\b|\bssh\b|\brm\s+-rf\s+/(?!\S))")


def run_shell(cmd: str, ws: Path) -> str:
    if BANNED.search(cmd):
        return "refused: command not allowed"
    p = _run(["bash", "-lc", cmd], ws, timeout=60)
    if p is None:
        return "timeout"
    out = (p.stdout + p.stderr).strip()
    return (out[:1500] + "\n...[truncated]") if len(out) > 1500 else (out or "(no output)")


async def solve_agent(llm: LLM, task_dir: Path, task: dict, log: list,
                      max_steps: int = 8, cost_cap: float = 2500) -> Result:
    ws = task_dir / "workspace"
    ws_cwd = ws if ws.is_dir() else task_dir
    first = (f"Task: {instruction_core(task['instruction'])}\n\n{workspace_preview(ws)}")
    hist: list[dict] = []
    cost = 0.0
    for step in range(max_steps):
        msgs = [{"role": "user", "content": first}] + hist[-6:]  # keep the last 3 exchanges
        reply, c = await llm.ask(SYS_AGENT, msgs, max_tokens=350)
        cost += c
        log.append({"step": step, "reply": reply, "cost": c})
        act = parse_act(reply)
        if act is None:
            hist += [{"role": "assistant", "content": reply[:300]},
                     {"role": "user", "content": 'Reply with one json object: {"cmd": ...} or {"answer": ...}'}]
        elif "answer" in act:
            ans = str(act["answer"]).strip()
            if (not ans or ans.upper().startswith("IMPOSSIBLE") or len(ans) > 80
                    or re.search(r"\b(not (found|contain|available|present|provided)|no (information|data|record))\b", ans, re.I)):
                return Result(False, "", cost, "model says impossible")
            write_answer(task_dir, ans)
            return Result(True, ans, cost)
        else:
            out = await asyncio.to_thread(run_shell, str(act["cmd"]), ws_cwd)
            hist += [{"role": "assistant", "content": json.dumps(act)[:400]},
                     {"role": "user", "content": out}]
        if cost > cost_cap:
            return Result(False, "", cost, "cost cap")
    return Result(False, "", cost, "step cap")
