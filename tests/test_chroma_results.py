"""chromadb result-shape handling in the Chroma accessor (no server needed).

chromadb >= 0.5 returns result embeddings as numpy arrays; ``x or []`` on
such a value raises, which used to break the hybrid+MMR path.
"""

from __future__ import annotations

import numpy as np

from serviette.server.accessors.chroma import _first_or_empty, _or_empty


def test_get_embeddings_as_ndarray():
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = _or_empty(arr)
    assert out is arr
    assert _or_empty(None) == []


def test_query_embeddings_list_of_ndarray():
    per_query = [np.array([[0.1, 0.2]])]
    assert _first_or_empty(per_query) is per_query[0]
    assert _first_or_empty(None) == []
    assert _first_or_empty([]) == []
    assert _first_or_empty(np.zeros((0, 2))) == []
