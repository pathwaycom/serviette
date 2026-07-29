"""Minimal in-process BM25 index shared by hybrid-capable accessors.

Kept dependency-free on purpose: hybrid search must not pull in a search
library for backends (embedded DuckDB, Qdrant) whose native keyword scoring
serviette does not use yet. Tens of thousands of chunks tokenize in well
under a second; the accessors cap corpus size and fall back to pure vector
search beyond it — at that scale a backend-native sparse index is the right
tool (native sparse-vector hybrid is planned).
"""

from __future__ import annotations

import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over pre-built hit dicts.

    ``hits`` are accessor result dicts (``text`` / ``metadata`` and optionally
    ``embedding``); :meth:`search` returns the same dicts with a BM25
    ``score``. ``doc_count`` and ``has_embeddings`` let the owning accessor
    decide when a rebuild is due.
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, hits: list[dict[str, Any]], has_embeddings: bool) -> None:
        self.has_embeddings = has_embeddings
        self.doc_count = len(hits)
        self._hits = hits
        self._doc_terms: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        total_len = 0
        for hit in hits:
            terms: dict[str, int] = {}
            for token in _tokenize(hit["text"]):
                terms[token] = terms.get(token, 0) + 1
            self._doc_terms.append(terms)
            total_len += sum(terms.values())
            for token in terms:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._avg_len = (total_len / self.doc_count) if self.doc_count else 0.0
        n = self.doc_count
        self._idf = {
            token: math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        query_tokens = [t for t in _tokenize(query) if t in self._idf]
        if not query_tokens or not self.doc_count:
            return []
        scored: list[tuple[float, int]] = []
        for i, terms in enumerate(self._doc_terms):
            doc_len = sum(terms.values())
            score = 0.0
            for token in query_tokens:
                tf = terms.get(token)
                if not tf:
                    continue
                denom = tf + self._K1 * (
                    1 - self._B + self._B * doc_len / (self._avg_len or 1.0)
                )
                score += self._idf[token] * tf * (self._K1 + 1) / denom
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda pair: -pair[0])
        return [{**self._hits[i], "score": score} for score, i in scored[:k]]
