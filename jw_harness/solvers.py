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
import sys
from pathlib import Path

from .llm import CHEAP, LLM, SECOND
from .mathfix import solve_reply

# ---------------------------------------------------------------- prompts
# Measured on 104 tasks (bench/): see README for the numbers behind each choice.
SYS_LINE = "Fix the bugs. Output only changed lines as N: code (indent auto). Nothing else."
SYS_FUNC = ("You fix bugs in Python. Reply ONLY with the corrected function definitions "
            "in one ```python block. No prose.")
SYS_STEPS = (
    "Translate the word problem into Python statements, ONE line per sentence, "
    "updating variable x. Do NOT compute anything yourself. Use // for rounding down, "
    "ceil(a/b) for rounding up, % for remainder.\n"
    "Percent loss p%: x -= x*p//100; percent growth: x += x*p//100. "
    "The final value must be exactly the quantity asked in the question. "
    "The last line must set x to the answer.\n"
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


def instruction_core(instr: str) -> str:
    """Drop the shared boilerplate tail (tools hint, answer.txt rules)."""
    return re.split(r"\s(?:Compute it with|Use bash|Use the file tools|Work only inside)", instr)[0]


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


# ---------------------------------------------------------------- coding
def coding_target(task: dict, ws: Path) -> tuple[str, str] | None:
    """(test file, file to fix) parsed from the instruction, or None if not this task shape."""
    instr = task.get("instruction", "")
    m = re.search(r"python3?\s+([\w./-]+\.py)", instr)
    test = m.group(1) if m else "test_mylib.py"
    m = re.search(r"bugs in\s+([\w./-]+\.py)", instr)
    lib = m.group(1) if m else "mylib.py"
    if (ws / test).is_file() and (ws / lib).is_file():
        return test, lib
    return None


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


def contradictory_tests(ws: Path, test: str) -> str:
    """The same call is expected to give different values -> no fix can pass all tests."""
    try:
        tree = ast.parse((ws / test).read_text())
    except (SyntaxError, OSError):
        return ""
    seen: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args):
            continue
        attr = n.func.attr
        try:
            if attr == "assertEqual" and len(n.args) >= 2:
                a, b = n.args[0], n.args[1]
                if isinstance(a, ast.Constant) and not isinstance(b, ast.Constant):
                    a, b = b, a
                seen.setdefault(ast.unparse(a), set()).add(repr(_const_value(b)))
            elif attr in ("assertTrue", "assertFalse"):
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


def failing(report: str, src: str, mod: str) -> tuple[list[str], set[str]]:
    funcs = {n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    descs, names = [], set()
    for block in re.split(r"^={10,}$", report, flags=re.M)[1:]:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        call = next((l.strip() for l in lines if "self.assert" in l), "")
        err = next((l.strip() for l in reversed(lines)
                    if re.match(r"\w*(Error|Exception)\b", l.strip())), "")
        names.update(n for n in re.findall(r"\b\w+\.(\w+)\(", call) if n in funcs)
        names.update(n for n in re.findall(rf'{re.escape(mod)}", line \d+, in (\w+)', block) if n in funcs)
        descs.append(f"{call}  -> {err}" if call else err)
    if not names and report:  # import error / unknown failure shape: show everything
        names = funcs
        descs = descs or report.strip().splitlines()[-4:]
    return descs[:8], names


def spans(src: str) -> dict[str, tuple[int, int]]:
    return {n.name: (n.lineno, n.end_lineno) for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def coding_prompt(mode: str, src: str, descs: list[str], names: set[str]) -> str:
    lines, sp = src.splitlines(), spans(src)
    chunks = []
    for n in sorted(names, key=lambda k: sp[k][0]):
        a, b = sp[n]
        if mode == "line":
            chunks.append("\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1)))
        else:
            chunks.append("\n".join(lines[a - 1:b]))
    return "Failing:\n" + "\n".join(descs) + "\n\nCode:\n" + "\n\n".join(chunks)


def apply_line(src: str, reply: str) -> str:
    lines = src.splitlines()
    for m in re.finditer(r"^\s*(\d+)\s*[:|]\s?(.*)$", reply, flags=re.M):
        i = int(m.group(1))
        if 1 <= i <= len(lines):
            old = lines[i - 1]
            code = m.group(2).strip()
            if old.strip():
                # indentation is always taken from the original line: models miscount spaces
                lines[i - 1] = old[:len(old) - len(old.lstrip())] + code
            elif code:
                # the model grew a function into the blank line after it: use the body's indent,
                # keep the blank separator
                prev = next((l for l in reversed(lines[:i - 1]) if l.strip()), "")
                indent = prev[:len(prev) - len(prev.lstrip())] or "    "
                lines[i - 1] = indent + code + "\n"
    out = "\n".join(lines) + "\n"
    try:
        compile(out, "mylib.py", "exec")  # ast.parse misses e.g. 'return' outside a function
    except SyntaxError:
        return src
    return out


def apply_func(src: str, reply: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    code = m.group(1) if m else reply
    try:
        new = {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    except SyntaxError:
        return src
    code_lines, lines = code.splitlines(), src.splitlines()
    for name, (a, b) in sorted(spans(src).items(), key=lambda kv: -kv[1][0]):
        if name in new:
            n = new[name]
            start = n.decorator_list[0].lineno if n.decorator_list else n.lineno
            lines[a - 1:b] = code_lines[start - 1:n.end_lineno]
    return "\n".join(lines) + "\n"


async def solve_coding(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    ws = task_dir / "workspace"
    tgt = coding_target(task, ws)
    if tgt is None:
        return None  # not the "fix lib until tests pass" shape: let the agent handle it
    test, lib = tgt
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
        src = (ws / lib).read_text()
        try:
            descs, names = failing(report, src, lib)
        except SyntaxError:
            descs, names = report.strip().splitlines()[-4:], set()
        if not names:
            break
        if prev == descs and mode == "line":
            continue  # no progress with the cheap model: jump to the escalation steps
        prev = descs
        prompt = coding_prompt(mode, src, descs, names)
        reply, c = await llm.ask(SYS_LINE if mode == "line" else SYS_FUNC, prompt, model=model,
                                 max_tokens=300 if mode == "line" else 400)
        cost += c
        log.append({"model": model, "mode": mode, "prompt": prompt, "reply": reply, "cost": c})
        (ws / lib).write_text(apply_line(src, reply) if mode == "line" else apply_func(src, reply))
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


async def solve_math(llm: LLM, task_dir: Path, task: dict, log: list) -> Result:
    q = instruction_core(task["instruction"])
    cost = 0.0
    for model in (CHEAP, SECOND):
        reply, c = await llm.ask(SYS_STEPS, q, model=model, max_tokens=300)
        cost += c
        ans = solve_reply(reply)
        log.append({"model": model, "prompt": q, "reply": reply, "answer": ans, "cost": c})
        start = re.search(r"^\s*x\s*=\s*(-?\d+)\s*$", reply, flags=re.M)
        if start and start.group(1) not in re.findall(r"\d+", q):
            # e.g. "x = 1000" copied from the prompt example: the question has no starting value
            return Result(False, "", cost, f"start value {start.group(1)} not in the question")
        if ans is not None and not ans.startswith("-"):
            write_answer(task_dir, ans)
            return Result(True, ans, cost)
    return Result(False, "", cost, "no evaluable program")


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
    return ws.is_dir() and any(ws.rglob("*.csv"))


async def solve_data(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    from . import dataq  # pandas import is heavy: only when a data task shows up
    ws = task_dir / "workspace"
    user = f"{dataq.preview(ws)}\n\nQ: {data_question(task['instruction'])}"
    cost = 0.0
    for model, mode in ((CHEAP, "sql"), (SECOND, "pandas")):
        msgs = [{"role": "user", "content": user}]
        for _ in range(2):  # one retry with the execution error
            reply, c = await llm.ask(SYS_QUERY[mode], msgs, model=model, max_tokens=300)
            cost += c
            err, ans = "", None
            try:
                ans = await asyncio.to_thread(dataq.RUNNERS[mode], ws, reply)
                if ans is None:
                    err = "query returned no value"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:200]}"
            log.append({"model": model, "mode": mode, "reply": reply, "answer": ans, "error": err, "cost": c})
            if not err and mode == "sql" and not re.search(r"\bfrom\b", reply, re.I):
                err = "constant query"  # "SELECT 0": the model gave up on the data
            if not err:
                write_answer(task_dir, ans)
                return Result(True, ans, cost)
            if err in ("query returned no value", "constant query") or _missing_column(err, ws):
                return Result(False, "", cost, f"infeasible: {err}")
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Error: {err}\nReply with a corrected query only."}]
    return None  # let the agent loop try


def _missing_column(err: str, ws: Path) -> bool:
    """'no such column: discount' / KeyError 'discount' for a name that is in no CSV header."""
    m = re.search(r"no such column: (\w+)|KeyError: \W*(\w+)", err)
    if not m:
        return False
    col = (m.group(1) or m.group(2)).lower()
    headers = set()
    for p in ws.rglob("*.csv"):
        with p.open(errors="replace") as f:
            headers |= {h.strip().lower() for h in f.readline().split(",")}
    return col not in headers


# ---------------------------------------------------------------- search
# The kb is parsed into an entity graph by the wrapper (kbgraph.py). Measured on 62+25 tasks
# (1-3 hops): path query 87/87 (35b) at ~21 units, subgraph 85/87 (35b; misses 3-hop chains),
# whole kb 49/62 at ~89, the agent loop 48/55 at ~294.
SYS_SUBGRAPH = "Answer from the facts below with the value only. If they do not contain the answer, reply IMPOSSIBLE."
SYS_PATH = ("Reply with a path that answers the question: the name from the question, then fields from the list. "
            "Example: Anna Kim > Employer > Headquarters. If the data cannot answer it, reply IMPOSSIBLE.")


def search_question(instr: str) -> str:
    return re.sub(r"^The directory \S+ contains [^.]*\.\s*", "", instruction_core(instr))


def load_kb(ws: Path):
    from . import kbgraph
    return kbgraph.load(ws) if ws.is_dir() else None


def _value(reply: str) -> str | None:
    v = reply.strip().strip("`*\"'. ").splitlines()[0].strip() if reply.strip() else ""
    v = re.sub(r"^(answer|value)\s*:\s*", "", v, flags=re.I)
    return None if not v or v.upper().startswith("IMPOSSIBLE") else v


async def solve_search(llm: LLM, task_dir: Path, task: dict, log: list) -> Result | None:
    kb = load_kb(task_dir / "workspace")
    if kb is None:
        return None  # not a field-based kb: the agent loop handles it
    q = search_question(task["instruction"])
    cost = 0.0
    ment = kb.mentioned(q)
    msgs = [{"role": "user", "content": f"Fields:\n{kb.schema()}\n\nQ: {q}"}]
    for _ in range(2):  # one retry with the lookup error
        reply, c = await llm.ask(SYS_PATH, msgs, model=CHEAP, max_tokens=60)
        cost += c
        try:
            ans, err = kb.follow(reply, ment[0] if len(ment) == 1 else None), ""
        except LookupError as e:
            ans, err = None, str(e)
        log.append({"model": CHEAP, "mode": "path", "reply": reply, "answer": ans, "error": err, "cost": c})
        if ans:
            write_answer(task_dir, ans)
            return Result(True, ans, cost)
        if err == "IMPOSSIBLE":
            break
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Error: {err}\nReply with a corrected path or IMPOSSIBLE."}]
    # second opinion with the other model and the other view before giving up
    facts = kb.subgraph(q)
    if facts:
        reply, c = await llm.ask(SYS_SUBGRAPH, f"{facts}\n\nQ: {q}", model=SECOND, max_tokens=60)
        cost += c
        ans = _value(reply)
        log.append({"model": SECOND, "mode": "subgraph", "reply": reply, "answer": ans, "cost": c})
        if ans:
            write_answer(task_dir, ans)
            return Result(True, ans, cost)
    return Result(False, "", cost, "kb has no answer")


# ---------------------------------------------------------------- generic agent
def workspace_preview(ws: Path, budget: int = 1800) -> str:
    if not ws.is_dir():
        return "(no files)"
    out, used = [], 0
    files = sorted(p for p in ws.rglob("*") if p.is_file())
    listing = "\n".join(str(p.relative_to(ws)) for p in files[:60])
    out.append("Files:\n" + listing + ("\n..." if len(files) > 60 else ""))
    used = len(out[0])
    for p in files:
        if p.suffix not in (".csv", ".md", ".txt", ".json", ".py", ".tsv") or used > budget:
            continue
        try:
            head = "\n".join(p.read_text(errors="replace").splitlines()[:5])[:400]
        except OSError:
            continue
        chunk = f"\n--- head {p.relative_to(ws)} ---\n{head}"
        out.append(chunk)
        used += len(chunk)
    return "".join(out)[:budget + 400]


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
            if not ans or ans.upper() == "IMPOSSIBLE":
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
