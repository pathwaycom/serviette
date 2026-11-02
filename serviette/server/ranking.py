"""Rank fusion and diversification — pure functions over hit dicts.

Hits are the accessor result dicts (``text`` / ``metadata`` / ``score``,
plus ``embedding`` when the accessor was asked for it). Scores coming from
different queries or different scorers are not comparable, so fusion uses
ranks: RRF for hybrid search, round-robin for decomposed queries.
"""

from __future__ import annotations

import math
from typing import Any

# The standard RRF dampening constant (Cormack et al.): small enough that
# top ranks dominate, large enough that a #1 in one list does not drown a
# consistent #3 in every list.
RRF_K = 60


def rrf_merge(
    result_lists: list[list[dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    """Fuse ranked result lists with reciprocal-rank fusion.

    A hit appearing in several lists (same ``text``) sums its 1/(RRF_K + rank)
    contributions, so chunks that answer *several* sub-queries rise. The fused
    RRF value replaces ``score`` — original per-query similarities are not
    comparable across lists.
    """

    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            key = hit["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            # Keep the first-seen dict (embedding included when present).
            fused.setdefault(key, hit)
    merged = [
        {**fused[key], "score": score}
        for key, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]
    return merged[:limit]


def interleave_merge(
    result_lists: list[list[dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    """Round-robin fusion for multi-query (decomposed) retrieval.

    Takes every list's rank-1 hit, then every rank-2, and so on, deduplicating
    by ``text``. Unlike summed RRF this *guarantees* the top hit of each
    sub-query a slot — for multi-hop questions a chunk that is perfect for one
    hop must beat a chunk that is mediocre for every hop. Hits keep their own
    per-query scores (not comparable across lists; order is the contract).
    """

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank in range(max((len(hits) for hits in result_lists), default=0)):
        for hits in result_lists:
            if rank >= len(hits) or len(merged) >= limit:
                continue
            hit = hits[rank]
            if hit["text"] not in seen:
                seen.add(hit["text"])
                merged.append(hit)
    return merged[:limit]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mmr_select(
    hits: list[dict[str, Any]],
    k: int,
    diversity: float,
    *,
    relevance: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Greedy maximal-marginal-relevance selection of ``k`` hits.

    Hits arrive in relevance order. By default their scores must be comparable
    across the pool and are min-max-normalized. Callers with incomparable
    scores supply one ``relevance`` value in [0, 1] per hit, aligned with that
    order. This selection signal is independent of the returned hit scores.

    Redundancy is the max cosine similarity to the already-selected set (via
    each hit's ``embedding``). ``diversity`` weighs redundancy: 0 preserves
    the incoming top-k, 1 ignores relevance entirely. When the whole pool
    fits, its incoming order is kept. Hits missing an embedding are treated
    as maximally novel rather than dropped.
    """

    if relevance is not None and len(relevance) != len(hits):
        raise ValueError("MMR relevance must have one value per hit")
    if k <= 0:
        return []
    if diversity == 0.0:
        return hits[:k]
    if k >= len(hits) or not hits:
        return list(hits)
    if relevance is None:
        lo = min(h["score"] for h in hits)
        hi = max(h["score"] for h in hits)
        span = (hi - lo) or 1.0
        relevance = [(h["score"] - lo) / span for h in hits]

    selected: list[int] = []
    remaining = list(range(len(hits)))
    while remaining and len(selected) < k:
        best_i, best_val = remaining[0], -math.inf
        for i in remaining:
            emb = hits[i].get("embedding")
            redundancy = 0.0
            if emb is not None and selected:
                sims = [
                    _cosine(emb, hits[j]["embedding"])
                    for j in selected
                    if hits[j].get("embedding") is not None
                ]
                redundancy = max(sims) if sims else 0.0
            value = (1.0 - diversity) * relevance[i] - diversity * redundancy
            if value > best_val:
                best_i, best_val = i, value
        selected.append(best_i)
        remaining.remove(best_i)
    return [hits[i] for i in selected]
