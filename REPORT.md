# LLM Inference + Observability — Report

Text-to-SQL PoC: Qwen3-30B-A3B served with vLLM on 1×H100, a LangGraph
verify→revise agent on top, observed with Prometheus/Grafana (serving) and
Langfuse (per-request), evaluated on a 30-question BIRD subset.

---

## 1. Serving configuration (Phase 1)

Model fixed at `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE: 30B total / ~3B active
per token — VRAM-heavy, compute-light per token). Hardware fixed at 1×H100 80GB.

| Flag | Value | Why (for this workload) |
|---|---|---|
| `--max-model-len` | `8192` | Prompts are ~1.5–3K tokens with short SQL outputs. The model's 256K default context would reserve huge KV-cache per sequence we never use; capping at 8K covers our prompts with margin and frees KV cache for concurrency. |
| `--max-num-batched-tokens` | `8192` | Kept ≥ `max-model-len` to satisfy the scheduler (otherwise vLLM refuses to start), and lets a full-length prompt prefill in a single step. |

Operational note: vLLM's `torch.compile` path JIT-compiles a CUDA helper at
startup and needs the Python dev headers (`python3-dev` / `build-essential`) —
installing them was required to boot on the VM. Kept `torch.compile`/CUDA-graphs
enabled (no `--enforce-eager`) to preserve decode performance for the SLO.

**Deliverable:** `screenshots/vllm_manual_query.png`.

---

## 2. Observability dashboard (Phase 2)

Grafana dashboard `infra/grafana/provisioning/dashboards/serving.json`, three
categories, built from vLLM's `/metrics` (current, non-deprecated metric names):

- **Latency (where in the lifecycle):** E2E latency P50/P95/P99 (SLO threshold
  line at 5s), TTFT percentiles (prefill), queue-wait P95 (capacity-bound
  signal), inter-token latency P50/P95 (decode).
- **Throughput:** prompt + generation tokens/s, completed requests/s,
  running-vs-waiting requests (saturation signal).
- **KV cache:** `kv_cache_usage_perc` (headroom, thresholds at 0.8/0.95),
  preemptions/s (eviction cliff), prefix-cache hit rate (schema-prompt reuse).

All panels confirmed to bind against the live H100 endpoint.

**Deliverable:** `screenshots/grafana_serving.png and infra/grafana/provisioning/dashboards/serving.json`.

---

## 3. Agent design (Phase 3)

LangGraph: `generate_sql → execute → verify → (revise → execute → verify)*`,
capped at `MAX_ITERATIONS = 3`.

- **verify** — deterministically rejects SQL that errored (no LLM call needed,
  guarantees broken SQL routes to revise) and otherwise asks the LLM for a
  `{"ok", "issue"}` verdict on plausibility (zero rows when results expected,
  columns that don't answer the question). Unparseable verdict defaults to `ok`
  so the loop can't hang.
- **revise** — feeds the failing SQL, its result, and the verifier's complaint
  back to the model for a fix; bumps the iteration counter.
- **route_after_verify** — ends on `verify_ok` or at the iteration cap, else
  revises.

Interactive testing confirmed the verify→revise loop fires (e.g. multi-join
questions where the first SQL omits a required join and verify routes to revise).
Langfuse traces show the `generate_sql / verify / (revise)` waterfall.

---

## 4. Agent tracing (Phase 4)

Langfuse (local, from docker-compose) captures the LangGraph spans via the
callback handler in `agent/server.py`. Each trace shows per-span prompt,
response, latency, and token count. Static metadata on every trace:
`project: mlops-hw2` (merged with optional per-request `tags` from the API).

**Deliverables:** `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png`.

---

## 5. Baseline eval (Phase 5)

Execution accuracy: agent's final SQL vs. gold SQL, compared on canonicalized
row sets (sorted, stringified, None→""). Per-iteration pass rate recorded with
carry-forward.

Command: `uv run python evals/run_eval.py --out results/eval_baseline.json`

| Metric | Value |
|---|---|
| Overall pass rate | **30.0%** (9/30) |
| Pass rate @ iter 0 | **26.7%** |
| Pass rate @ iter 1 | **30.0%** |
| Pass rate @ iter 2 | **30.0%** |
| Avg iterations | **1.43** |

Iteration distribution: 21 questions finished in 1 pass, 5 in 2, 4 in 3 (max cap).
The revise loop triggered on 9/30 questions; pass rate improved **+3.3 pp**
from iter 0 → iter 1, then flat (gains came from the first revise, not later ones).

**Deliverable:** `screenshots/grafana_eval_run.png`, `results/eval_baseline.json`.

---

## 6. Hitting the SLO (Phase 6)

**Target:** P95 end-to-end **agent** latency < 5s at 10+ RPS over a 5-min window.

Load test: `uv run python load_test/driver.py --rps 10 --duration 300`
(measures client-side latency to `POST /answer`, not single vLLM calls).

### After iteration 1 (`uvicorn --workers 4`)

| Metric | Before (1 worker) | After (4 workers) | SLO |
|---|---|---|---|
| Agent latency P50 | 75.0s | **3.04s** | — |
| Agent latency P95 | 100.8s | **60.2s** | **< 5s** |
| Agent latency P99 | 106.3s | 73.6s | — |
| Achieved RPS | 8.33 | 8.33 | 10+ |
| OK / errors | 2573 / 427 err | 2603 / 397 err | — |

**Grafana after workers (~18:46–18:51):** Steady state looked similar to before
(vLLM E2E P95 ~2–3s, queue-wait ~0.25s, waiting=0, prefix-cache ~85–90%). Near
**18:50:30** all panels spiked together: `num_requests_running` → ~140, gen
tokens/s and completed req/s jumped, KV cache → ~50% then dropped sharply, vLLM
E2E P99 >6s and TTFT P99 >200ms. Preemptions stayed 0. Interpretation: four
agent workers increased concurrent vLLM load; a late burst saturated running
slots and widened the **tail** of agent latency even while the median collapsed.

Key insight: Grafana E2E latency is **per vLLM request** (~3s steady state). The
load driver measures **full agent runs** (2–3 sequential LLM calls). With one
worker, agent-side queuing dominated (P50 75s). With four workers, **P50 met
the spirit of the SLO** (~3s) but **P95 still missed** (~60s) due to tail
requests queuing behind concurrent agent runs and periodic vLLM saturation spikes.

### Iteration log

1. **Saw** vLLM E2E P95 ~3s, queue-wait ~0.25s, waiting=0, KV cache ~50%,
   preemptions=0; agent load-test P95 ~101s, P50 ~75s, achieved RPS 8.3.
   **Hypothesized** sync single-worker agent serializes full agent runs while
   the load driver fires 10 RPS concurrently. **Changed** `--workers 4`.
   **Result** P50 75s → **3.0s** (targeted metric moved); P95 101s → **60s**
   (better but still 12× over SLO); achieved RPS unchanged at 8.3.

2. **Saw** late-run Grafana spike: running ~140, KV cache peak then flush, vLLM
   E2E/TTFT tail widened; agent P95 still ~60s with ~380 HTTP errors.
   **Hypothesized** four workers expose vLLM to too many concurrent sequences
   (2–3× agent RPS in LLM calls), creating tail latency without vLLM queue
   backlog (waiting still 0). **Did not change further** within GPU time budget;
   next lever would be `--max-num-seqs` cap, lower `--max-model-len` (4096), or
   FP8 weights to raise safe concurrency.

**Final config:** vLLM flags unchanged (§1); agent served with
`uvicorn agent.server:app --host 0.0.0.0 --port 8001 --workers 4`.

**SLO verdict:** **Missed.** P95 agent latency 60s (need <5s); achieved RPS 8.3
(need ≥10). Median latency and diagnosis quality improved substantially; tail
latency and effective throughput remain the gap.

**Quality after tuning:** Worker count does not change SQL logic. Re-use baseline
eval (`results/eval_baseline.json`, 30% pass) as post-tuning quality — no
regression from the latency fix.

**Deliverables:** `screenshots/grafana_before.png`, `screenshots/grafana_after.png`,
`results/eval_baseline.json` (quality unchanged; copy to `eval_after_tuning.json`
if required by rubric).

---

## 7. Did the agent loop earn its keep?

Yes, **modestly**. Baseline eval shows pass rate **26.7% at iter 0 → 30.0% at
iter 1/2** (+3.3 pp). Nine of 30 questions used multiple iterations (5 stopped
at iter 2, 4 at iter 3); the gain appears on the **first revise**, not on
further loops — iter 1 and iter 2 pass rates are identical, so extra revises
rarely rescued questions that the first fix couldn't. The architecture is doing
real work (verify catches bad SQL and revise fixes some of it), but diminishing
returns after one revision on this 30B + prompt setup. Worth the cost in quality
terms; each revise adds another full vLLM call, which hurts the latency SLO.

---

## 8. What I'd do with more time (specific)

- Run the agent with async invocation or a bounded request queue so load-test
  concurrency maps to vLLM concurrency instead of piling up at a sync worker.
- Schema linking / column pruning to shrink prompts (lower TTFT, higher
  prefix-cache efficiency on the non-schema tail).
- FP8 weights for Qwen3-30B to increase KV-cache headroom if vLLM becomes the
  limiter at higher agent worker counts.
- A smaller/faster verifier model (or rule-based checks for obvious SQL errors)
  to cut LLM calls per agent run from 2–3 toward 2.
- Few-shot exemplars per DB for hard join patterns in BIRD.
