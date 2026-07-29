"""End-to-end FRAMES run: ask every question via serviette ``/rag``, judge the
answers with the paper's autorater prompt (Appendix B, judge_prompt.txt).

Reports accuracy (the paper's headline metric) **and** gold-article recall of
the contexts actually used (the paper's retrieval metric), plus token usage of
the judge. Generation cost lives inside the serviette server (its ``llm``
config section); the judge model is set here.

    OPENAI_API_KEY=... python run_bench.py --questions data/questions.jsonl \
        --port 8987 --k 8 --label e5small-hybrid-gpt4o \
        [--judge-model gpt-4o-mini] [--limit 0] [--concurrency 6]
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

from run_retrieval_eval import gold_slugs, hit_slug, post_json

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text()


def openai_chat(model: str, prompt: str, api_key: str) -> tuple[str, dict]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"]["content"], resp.get("usage", {})
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--port", type=int, default=8987)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--label", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    questions = [json.loads(line) for line in open(args.questions)]
    if args.limit:
        questions = questions[: args.limit]
    rag_url = f"http://127.0.0.1:{args.port}/api/v1/rag"

    usage_lock = Lock()
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def ask(item: dict) -> dict:
        # Reasoning generators can spend minutes per answer; a tight timeout
        # here is worse than useless — the server keeps generating after the
        # client gives up, so every retry pays for a duplicate generation.
        resp = post_json(
            rag_url, {"query": item["prompt"], "k": args.k}, timeout=900, retries=2
        )
        gold = gold_slugs(item["wiki_links"])
        got = {hit_slug(h) for h in resp.get("sources", [])}
        return {
            **item,
            "predicted": resp.get("answer", ""),
            "context_recall": len(gold & got) / len(gold) if gold else 1.0,
        }

    def judge(item: dict) -> dict:
        verdict, usage = openai_chat(
            args.judge_model,
            JUDGE_PROMPT.format(
                question=item["prompt"],
                predicted=item["predicted"],
                gold=item["answer"],
            ),
            api_key,
        )
        with usage_lock:
            judge_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            judge_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        correct = '"TRUE"' in verdict or verdict.rstrip().endswith("TRUE")
        return {**item, "judge": verdict[-400:], "correct": bool(correct)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        answered = list(ex.map(ask, questions))
    t_answer = time.time() - t0
    print(f"answered {len(answered)} in {t_answer:.0f}s", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        judged = list(ex.map(judge, answered))
    print(f"judged in {time.time() - t0:.0f}s", flush=True)

    accuracy = statistics.mean(1.0 if r["correct"] else 0.0 for r in judged)
    recall = statistics.mean(r["context_recall"] for r in judged)
    summary = {
        "label": args.label,
        "k": args.k,
        "questions": len(judged),
        "accuracy": round(accuracy, 4),
        "context_article_recall": round(recall, 4),
        "judge_model": args.judge_model,
        "judge_usage": judge_usage,
        "answer_seconds": round(t_answer),
    }
    print(json.dumps(summary))

    out = Path(args.out or f"results/bench-{args.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
