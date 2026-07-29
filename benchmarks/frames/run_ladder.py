"""Corpus-independent ladder rungs: naive (no retrieval) and oracle.

These two mirror the paper's calibration rows and involve no retrieval at
all, so they run without any index:

- ``--mode naive``: the question alone — measures the generator's parametric
  knowledge (the paper: 40.8% for Gemini-Pro-1.5);
- ``--mode oracle``: the question plus the *gold* Wikipedia articles
  (``wiki_links``) as context — the retrieval-quality ceiling (the paper:
  72.9%). Gold articles come from the snapshot corpus; the few post-snapshot
  ones are absent here exactly as they were for the paper.

Direct OpenAI calls (no serviette server involved), the Appendix-B judge,
and built-in cost accounting: an estimate is printed *before* anything is
spent, actual usage is accumulated from every response and reported in the
summary.

    OPENAI_API_KEY=... python run_ladder.py --mode naive --generator gpt-5 \
        --questions $FRAMES_DATA/questions.jsonl [--limit 20]
    OPENAI_API_KEY=... python run_ladder.py --mode oracle --generator gpt-5 \
        --questions $FRAMES_DATA/questions.jsonl --gold-dir $FRAMES_DATA/corpus-gold
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from run_bench import JUDGE_PROMPT
from run_retrieval_eval import gold_slugs

# $/1M tokens (input, output); used for the pre-run estimate and the actual
# spend report. Update when prices move.
PRICES = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context is insufficient, say so."
)


class Usage:
    def __init__(self) -> None:
        self.lock = Lock()
        self.prompt = 0
        self.completion = 0

    def add(self, usage: dict) -> None:
        with self.lock:
            self.prompt += usage.get("prompt_tokens", 0)
            self.completion += usage.get("completion_tokens", 0)

    def cost(self, model: str) -> float:
        p_in, p_out = PRICES.get(model, (0, 0))
        return (self.prompt * p_in + self.completion * p_out) / 1e6


def chat(model: str, messages: list[dict], api_key: str, *, temperature=None) -> tuple[str, dict]:
    body: dict = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if model.startswith(("gpt-5", "o")):
        body["max_completion_tokens"] = 8000  # bound reasoning runaways
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"]["content"] or "", resp.get("usage", {})
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def oracle_context(item: dict, gold_dir: Path, max_chars: int) -> str:
    parts = []
    for slug in sorted(gold_slugs(item["wiki_links"])):
        # Files are stored with original-case slugs; resolve case-insensitively.
        matches = [p for p in gold_dir.glob("*.txt") if p.stem.casefold() == slug]
        if matches:
            parts.append(matches[0].read_text(errors="replace"))
    context = "\n\n====\n\n".join(parts)
    return context[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "oracle"], required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--gold-dir", default=None)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-context-chars", type=int, default=400_000)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    questions = [json.loads(line) for line in open(args.questions)]
    if args.limit:
        questions = questions[: args.limit]
    gold_dir = Path(args.gold_dir) if args.gold_dir else None
    if args.mode == "oracle" and gold_dir is None:
        raise SystemExit("--mode oracle requires --gold-dir")

    # ---- pre-run cost estimate (printed BEFORE any spend) -------------------
    p_in, p_out = PRICES.get(args.generator, (0, 0))
    if args.mode == "naive":
        est_in = len(questions) * 80
    else:
        est_in = len(questions) * 22_000
    est_out = len(questions) * (1200 if args.generator.startswith(("gpt-5", "o")) else 100)
    est = (est_in * p_in + est_out * p_out) / 1e6
    judge_est = len(questions) * 450 * 0.15 / 1e6 + len(questions) * 110 * 0.6 / 1e6
    print(
        f"ESTIMATE [{args.mode}/{args.generator}/{len(questions)}q]: "
        f"~{est_in/1e6:.1f}M in + ~{est_out/1e6:.2f}M out ≈ ${est:.2f} "
        f"(+judge ≈ ${judge_est:.2f})",
        flush=True,
    )

    gen_usage, judge_usage = Usage(), Usage()

    def ask(item: dict) -> dict:
        if args.mode == "naive":
            messages = [{"role": "user", "content": item["prompt"]}]
        else:
            context = oracle_context(item, gold_dir, args.max_context_chars)
            messages = [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {item['prompt']}"},
            ]
        temperature = None if args.generator.startswith(("gpt-5", "o")) else 0
        answer, usage = chat(args.generator, messages, api_key, temperature=temperature)
        gen_usage.add(usage)
        return {**item, "predicted": answer}

    def judge(item: dict) -> dict:
        verdict, usage = chat(
            args.judge_model,
            [{"role": "user", "content": JUDGE_PROMPT.format(
                question=item["prompt"], predicted=item["predicted"], gold=item["answer"])}],
            api_key,
            temperature=0,
        )
        judge_usage.add(usage)
        correct = '"TRUE"' in verdict or verdict.rstrip().endswith("TRUE")
        return {**item, "judge": verdict[-400:], "correct": bool(correct)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        answered = list(ex.map(ask, questions))
    print(
        f"answered {len(answered)} in {time.time()-t0:.0f}s | gen spend so far: "
        f"${gen_usage.cost(args.generator):.2f}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        judged = list(ex.map(judge, answered))

    accuracy = statistics.mean(1.0 if r["correct"] else 0.0 for r in judged)
    label = f"{args.mode}-{args.generator}"
    summary = {
        "label": label,
        "questions": len(judged),
        "accuracy": round(accuracy, 4),
        "generator_usage": {"prompt": gen_usage.prompt, "completion": gen_usage.completion},
        "generator_cost_usd": round(gen_usage.cost(args.generator), 2),
        "judge_cost_usd": round(judge_usage.cost(args.judge_model), 2),
    }
    print(json.dumps(summary), flush=True)
    out = Path(args.results) / f"ladder-{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
