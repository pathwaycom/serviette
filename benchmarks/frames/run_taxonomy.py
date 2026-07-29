"""Classify the failures of a bench run into an error taxonomy.

For every question the run got wrong, an LLM classifier (given the question,
the gold answer, the model's answer, and the retrieval coverage flag) assigns
exactly one category:

- ``retrieval_miss``   — needed facts absent from the context and the answer
                         shows the model didn't know them either;
- ``reasoning_error``  — facts were available (context or evidently known)
                         but chained/aggregated incorrectly;
- ``judge_error``      — the prediction is actually equivalent to the gold
                         answer (the autorater was too strict);
- ``dataset_issue``    — the question/gold answer is ambiguous, outdated for
                         the 2023-06-01 snapshot, or self-inconsistent.

Judge-error rows double as an estimate of autorater noise, so the taxonomy
also bounds measurement error of the headline accuracies.

    OPENAI_API_KEY=... python run_taxonomy.py \
        --run results/bench-serviette-fullwiki-gpt5-permissive.jsonl \
        [--classifier gpt-4o]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_ladder import Usage, chat

PROMPT = """\
You are auditing a RAG benchmark. A model answered a multi-hop question \
incorrectly according to an automatic judge. Classify the failure into \
exactly one category:

- retrieval_miss: the answer indicates required facts were not available \
(refusal, "context doesn't say", or a guess unsupported by any fact)
- reasoning_error: the necessary facts appear known or provided, but they \
were combined, compared, counted, or converted incorrectly
- judge_error: the prediction is substantively equivalent to the gold answer \
(formatting, units, naming variants, added detail) and was misgraded
- dataset_issue: the question or its gold answer is ambiguous, time-\
dependent beyond the 2023-06-01 snapshot, or internally inconsistent

Question: {question}
Gold answer: {gold}
Model's answer: {predicted}
Share of gold articles present in the retrieved context: {recall}

Reply with the category name alone on the first line, then one short \
sentence of justification."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--classifier", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.run)]
    summary_in = rows[0].get("summary", {})
    failures = [r for r in rows[1:] if not r.get("correct")]
    print(f"failures to classify: {len(failures)} of {summary_in.get('questions')}")

    api_key = os.environ["OPENAI_API_KEY"]
    usage = Usage()
    categories = [
        "retrieval_miss", "reasoning_error", "judge_error", "dataset_issue"
    ]

    def classify(item: dict) -> dict:
        reply, u = chat(
            args.classifier,
            [{"role": "user", "content": PROMPT.format(
                question=item["prompt"],
                gold=item["answer"],
                predicted=item["predicted"][:2000],
                recall=item.get("context_recall", "n/a"),
            )}],
            api_key,
            temperature=0,
        )
        usage.add(u)
        first = reply.strip().splitlines()[0].strip().lower()
        category = next(
            (c for c in categories if c in first or c in re.sub(r"\W", "_", first)),
            "unparsed",
        )
        return {**item, "taxonomy": category, "taxonomy_note": reply[:300]}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        classified = list(ex.map(classify, failures))

    counts = Counter(r["taxonomy"] for r in classified)
    total = len(classified)
    summary = {
        "run": Path(args.run).name,
        "classifier": args.classifier,
        "failures": total,
        "taxonomy": {
            c: {"n": counts.get(c, 0), "share": round(counts.get(c, 0) / total, 3)}
            for c in categories + ["unparsed"]
            if counts.get(c)
        },
        "classifier_cost_usd": round(usage.cost(args.classifier), 2),
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = Path(args.out or args.run.replace(".jsonl", "-taxonomy.jsonl"))
    with out.open("w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for row in classified:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
