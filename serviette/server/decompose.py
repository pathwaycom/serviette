"""Query decomposition for multi-hop questions.

One LLM call turns a question that chains several facts into standalone
search queries, one per fact. Retrieval then runs per sub-query and the
lists are fused (see :mod:`serviette.server.ranking`), so chunks belonging
to different hops stop competing for the same top-k slots — the failure
mode single-query retrieval hits on multi-hop benchmarks like FRAMES.
"""

from __future__ import annotations

import logging
import re

from serviette.server.llm import AsyncLLM

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
Split the question below into at most {n} standalone search queries, one per \
line, each retrieving ONE fact needed to answer it. Queries must be \
self-contained (no pronouns referring to other lines). If the question needs \
only one fact, output the question itself as the single query. Output only \
the queries, no numbering and no commentary.

Question: {query}"""

# A list marker the model may still prepend despite the prompt ("1. ", "12) ",
# "- ", "* "). After digits a dot or bracket is required, so a query that
# merely starts with a number ("1984 novel author") keeps it — that number is
# usually the very fact the retrieval needs.
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+")


async def decompose_query(
    llm: AsyncLLM, query: str, max_subqueries: int
) -> list[str]:
    """Return retrieval queries for ``query``: itself plus LLM sub-queries.

    The original question always stays first — it carries cross-hop context
    no sub-query has. Degrades to just the original on any LLM hiccup or
    unparseable output (never fails retrieval over an optimization).
    """

    try:
        reply = await llm.raw(
            _DECOMPOSE_PROMPT.format(n=max_subqueries, query=query)
        )
    except Exception:  # noqa: BLE001 - decomposition must never break /rag
        logger.warning("query decomposition failed; using the original query only")
        return [query]
    subqueries = []
    for line in reply.splitlines():
        cleaned = _LIST_MARKER.sub("", line, count=1).strip()
        if cleaned and cleaned.lower() != query.strip().lower():
            subqueries.append(cleaned)
    return [query] + subqueries[:max_subqueries]
