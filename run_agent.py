#!/usr/bin/env python3
"""
Budget-aware harness on openJiuwen.

  python run_agent.py --tasks-dir /data/tasks [--out solutions.json]

1. Recon (no LLM, free): classify each task, run its tests, count failures,
   look for signs that the task is infeasible.
2. Plan: estimate the cost of each task in budget units (input/10 + output),
   order by reward / estimated cost, suspected-infeasible tasks last.
3. Solve with up to 5 concurrent sessions; after every finished task write
   answer.txt and rewrite the solutions file atomically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from jw_harness import solvers
from jw_harness.llm import BudgetExhausted, from_env

# Measured mean cost per task in budget units (bench/runs, qwen3.6-35b-a3b-noreason).
EST = {"math": 70, "coding_per_failure": 35, "coding_base": 20,
       "data": 40, "search_kb": 25, "data_analysis": 250, "search": 300, "other": 1500}
TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", 52 * 60))  # the container is killed at 60 min


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks(root: Path) -> list[dict]:
    tasks = []
    for d in sorted(root.iterdir()):
        f = d / "task.json"
        if not f.is_file():
            continue
        try:
            t = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        t["_dir"] = str(d)
        t.setdefault("task_id", d.name)
        try:
            t["_reward"] = float(t.get("reward", 1))
        except (TypeError, ValueError):
            t["_reward"] = 1.0
        tasks.append(t)
    return tasks


# ------------------------------------------------------------------ recon
NAME_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
QUOTED_RE = re.compile(r"'([^']{2,40})'")


def workspace_text(ws: Path, limit: int = 3_000_000) -> str:
    out, used = [], 0
    for p in ws.rglob("*"):
        if p.is_file() and used < limit:
            try:
                s = p.read_text(errors="ignore")
            except OSError:
                continue
            out.append(s)
            used += len(s)
    return "\n".join(out)


def recon(t: dict) -> None:
    """Fill t['_kind'], t['_est'], t['_suspect'] without any LLM call."""
    d = Path(t["_dir"])
    ws = d / "workspace"
    ttype = t.get("task_type", "")
    instr = t.get("instruction", "")
    t["_suspect"] = ""
    t["_impossible"] = ""
    tgt = solvers.coding_target(t, ws) if ws.is_dir() else None
    if tgt:
        t["_kind"] = "coding"
        report = solvers.run_tests(ws, tgt[0])
        n = len(re.findall(r"^(FAIL|ERROR):", report, flags=re.M)) or (1 if report else 0)
        t["_failures"] = n
        t["_impossible"] = solvers.contradictory_tests(ws, tgt[0])
        t["_est"] = EST["coding_base"] + EST["coding_per_failure"] * max(n, 1)
        if not report:
            t["_est"] = 0  # tests already pass
        return
    if ttype == "math" and not ws.is_dir():
        t["_kind"], t["_est"] = "math", EST["math"]
        if solvers.math_missing_numbers(instr):
            t["_impossible"] = "no number in the starting sentence"
        return
    kb = solvers.load_kb(ws) if ttype == "search" else None
    if kb is not None:
        t["_kind"], t["_est"] = "search", EST["search_kb"]
        q = solvers.search_question(instr)
        if not kb.mentioned(q) and kb.unknown_names(q):
            t["_impossible"] = f"names {kb.unknown_names(q)} not in kb"
        return
    if ttype == "data_analysis" and solvers.has_csv(ws):
        t["_kind"], t["_est"] = "data", EST["data"]
    else:
        t["_kind"] = "agent"
        t["_est"] = EST.get(ttype, EST["other"])
    if not ws.is_dir():
        if ttype in ("search", "data_analysis"):
            t["_suspect"] = "no workspace"
        return
    text = workspace_text(ws)
    names = [n for n in NAME_RE.findall(instr) if not n.startswith(("The ", "In ", "How ", "Use ", "What "))]
    if ttype == "search" and names and not any(n in text for n in names):
        t["_suspect"] = f"names {names} not in kb"
    quoted = QUOTED_RE.findall(instr)
    if ttype == "data_analysis" and quoted and not any(q in text for q in quoted):
        if t["_kind"] == "data":  # the query would only select nothing
            t["_impossible"] = f"values {quoted} not in data"
        else:
            t["_suspect"] = f"values {quoted} not in data"


# ------------------------------------------------------------------ output
class Solutions:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def record(self, task_id: str, entry: dict) -> None:
        async with self.lock:
            self.data[task_id] = entry
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))
                os.replace(tmp, self.path)
            except OSError as e:
                log(f"cannot write solutions file {self.path}: {e}")


def solutions_path(arg: str | None, tasks_dir: Path) -> Path:
    raw = arg or os.environ.get("OUT")
    if raw:
        p = Path(raw)
        if p.is_dir() or raw.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
            return p / "solutions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return tasks_dir / "solutions.json"


# ------------------------------------------------------------------ main loop
async def solve_one(llm, t: dict, sols: Solutions, traj_dir: Path, deadline: float) -> None:
    tid = t["task_id"]
    d = Path(t["_dir"])
    if time.monotonic() > deadline:
        return
    trace: list = []
    t0 = time.monotonic()
    res = None
    try:
        if t["_kind"] == "coding":
            res = await solvers.solve_coding(llm, d, t, trace)
        elif t["_kind"] == "math":
            res = await solvers.solve_math(llm, d, t, trace)
        elif t["_kind"] == "data":
            res = await solvers.solve_data(llm, d, t, trace)
        elif t["_kind"] == "search":
            res = await solvers.solve_search(llm, d, t, trace)
        if res is None:
            if t["_suspect"]:  # likely infeasible: a short, cheap look only
                res = await solvers.solve_agent(llm, d, t, trace, max_steps=3, cost_cap=300)
            else:
                res = await solvers.solve_agent(llm, d, t, trace)
    except BudgetExhausted as e:
        res = solvers.Result(False, "", 0.0, f"budget: {e}")
    except Exception as e:  # noqa: BLE001
        res = solvers.Result(False, "", 0.0, f"error: {e}")
        trace.append({"traceback": traceback.format_exc()[-2000:]})
    spent = sum(x.get("cost", 0) for x in trace if isinstance(x, dict))
    entry = {"task_id": tid, "answer": res.answer, "submitted": res.ok, "reward": t["_reward"],
             "kind": t["_kind"], "est": t["_est"], "spent": round(spent, 1),
             "seconds": round(time.monotonic() - t0, 1), "note": res.note}
    if res.ok:
        await sols.record(tid, entry)
    try:
        (traj_dir / f"{tid}.json").write_text(json.dumps({**entry, "trace": trace}, ensure_ascii=False, indent=1))
    except OSError:
        pass
    log(f"{tid:24} {t['_kind']:6} {'OK ' if res.ok else '-- '} spent={spent:7.1f} est={t['_est']:5} "
        f"total={llm.spend.units:9.1f} {res.note}")


async def main_async(a) -> int:
    tasks_dir = Path(a.tasks_dir)
    tasks = load_tasks(tasks_dir)
    log(f"{len(tasks)} tasks in {tasks_dir}")
    start = time.monotonic()
    deadline = start + TIME_LIMIT_S

    await asyncio.gather(*(asyncio.to_thread(recon, t) for t in tasks))
    budget = float(os.environ["TOKEN_BUDGET"]) if os.environ.get("TOKEN_BUDGET") else None
    for t in tasks:
        t["_prio"] = t["_reward"] / max(t["_est"], 1)
    order = sorted(tasks, key=lambda t: (bool(t["_suspect"]), -t["_prio"]))
    for t in order:
        log(f"plan {t['task_id']:24} kind={t['_kind']:6} reward={t['_reward']:<4} est={t['_est']:<5}"
            f" {('SUSPECT: ' + t['_suspect']) if t['_suspect'] else ''}")

    sols = solutions_path(a.out, tasks_dir)
    log(f"solutions file: {sols}")
    traj_dir = Path(os.environ.get("TRAJ_DIR", "/tmp/trajectories"))
    traj_dir.mkdir(parents=True, exist_ok=True)
    solutions = Solutions(sols)

    # tests that already pass cost nothing
    for t in order:
        if t["_kind"] == "coding" and t["_est"] == 0:
            solvers.write_answer(Path(t["_dir"]), "DONE")
            await solutions.record(t["task_id"], {"task_id": t["task_id"], "answer": "DONE", "submitted": True,
                                                  "kind": "coding", "spent": 0})
    queue = [t for t in order if not (t["_kind"] == "coding" and t["_est"] == 0)]
    for t in queue:
        if t["_impossible"]:
            log(f"skip {t['task_id']}: {t['_impossible']}")
    queue = [t for t in queue if not t["_impossible"]]
    if a.skip_suspect:
        queue = [t for t in queue if not t["_suspect"]]

    llm = from_env(budget)
    q: asyncio.Queue = asyncio.Queue()
    for t in queue:
        q.put_nowait(t)

    async def worker():
        while not q.empty() and not llm.dead:
            t = q.get_nowait()
            rem = llm.remaining()
            if rem is not None and rem < t["_est"] * 0.5:
                log(f"skip {t['task_id']}: est {t['_est']} > remaining {rem:.0f}")
                continue
            await solve_one(llm, t, solutions, traj_dir, deadline)

    await asyncio.gather(*(worker() for _ in range(a.workers)))
    ok = sum(1 for v in solutions.data.values() if v.get("submitted"))
    log(f"done: submitted {ok}/{len(tasks)}, spent {llm.spend.units:.1f} units "
        f"({llm.spend.calls} calls, in={llm.spend.input_tokens} out={llm.spend.output_tokens}), "
        f"{time.monotonic() - start:.0f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", default=None, help="solutions file (default: $OUT or <tasks-dir>/solutions.json)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip-suspect", action="store_true", help="never spend on suspected-infeasible tasks")
    a = ap.parse_args()
    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY is not set", file=sys.stderr)
        return 2
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
