"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import httpx

AGENT_TIMEOUT = 180.0  # CPU stand-in can be slow; the H100 is fast.

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _iteration_sqls(history: list[dict]) -> list[str]:
    """Ordered SQL the agent held at each attempt (generate, then each revise)."""
    return [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and h.get("sql")]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question via execution accuracy, per agent iteration.

    Calls the agent, then re-runs the SQL it held at *each* iteration against
    the target DB and compares canonicalized rows to the gold result. This lets
    summarize() report what the pass rate would have been had we stopped after
    iteration 0, 1, 2, ...
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    # Gold reference rows (run once). If gold itself errors, we can't score.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    record: dict = {
        "question": q_text,
        "db_id": db_id,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
    }

    try:
        resp = httpx.post(agent_url, json={"question": q_text, "db": db_id}, timeout=AGENT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        record.update({
            "agent_ok": False,
            "agent_error": f"{type(e).__name__}: {e}",
            "final_sql": "",
            "iterations": 0,
            "per_iteration": [],
            "final_correct": False,
        })
        return record

    sqls = _iteration_sqls(data.get("history", []))
    if not sqls and data.get("sql"):  # fallback if history lacked sql entries
        sqls = [data["sql"]]

    per_iteration: list[dict] = []
    for i, sql in enumerate(sqls):
        ok, rows, err = run_sql(db_id, sql)
        correct = matches(gold_rows, rows) if gold_ok else False
        per_iteration.append({"iter": i, "sql": sql, "exec_ok": ok, "error": err, "correct": correct})

    record.update({
        "agent_ok": data.get("ok", False),
        "agent_error": data.get("error"),
        "final_sql": data.get("sql", ""),
        "iterations": data.get("iterations", len(sqls)),
        "per_iteration": per_iteration,
        "final_correct": per_iteration[-1]["correct"] if per_iteration else False,
    })
    return record


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    max_iters = max((len(r["per_iteration"]) for r in results), default=0) or 1

    def correct_at(r: dict, k: int) -> bool:
        per = r["per_iteration"]
        if not per:
            return False
        idx = k if k < len(per) else len(per) - 1  # carry-forward last attempt
        return bool(per[idx]["correct"])

    pass_rate_at_iteration = {
        str(k): round(sum(correct_at(r, k) for r in results) / n, 4)
        for k in range(max_iters)
    }

    counts = Counter(r["iterations"] for r in results)
    return {
        "n": n,
        "overall_pass_rate": round(sum(r["final_correct"] for r in results) / n, 4),
        "pass_rate_at_iteration": pass_rate_at_iteration,
        "iteration_counts": {str(k): counts[k] for k in sorted(counts)},
        "avg_iterations": round(sum(r["iterations"] for r in results) / n, 3),
        "gold_sql_errors": sum(1 for r in results if not r.get("gold_ok", True)),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
