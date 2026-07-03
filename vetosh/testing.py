"""Test-support helpers (not part of the public API; not user-documented).

Provides a deterministic, dependency-free fake embedding shared by the test
suite and by the test-only ``mock`` embedder type, so the indexer can run
end-to-end without any embedding provider or credentials.
"""

from __future__ import annotations

import hashlib

# numpy is a transitive dependency of pathway; importing it at module scope keeps
# it visible in this module's globals so Pathway can resolve the mock embedder's
# ``-> np.ndarray`` return annotation.
import numpy as np

EMBED_DIM = 384


def fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding: SHA-256 of ``text`` tiled to 384 floats."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
    repeats = (EMBED_DIM // len(digest)) + 1
    raw = (digest * repeats)[:EMBED_DIM]
    return [b / 255.0 for b in raw]


def build_mock_embedder():
    """Return a Pathway UDF mock embedder (imports pathway lazily)."""

    import pathway as pw

    @pw.udf(deterministic=True)  # pure hash; the engine need not memoize it
    def embed(text: str) -> np.ndarray:
        return np.array(fake_embedding(text), dtype=float)

    return embed
