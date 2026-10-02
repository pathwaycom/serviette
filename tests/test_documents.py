"""Tests for document-level questions on ``/rag`` (``rag.documents``)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serviette.config.schema import (
    DuckDbConfig,
    EmbedderConfig,
    RagConfig,
    ServerConfig,
    ServietteConfig,
)
from serviette.server import documents as docmode
from serviette.server.accessors.abstract import AsyncVectorAccessor
from serviette.server.accessors.duckdb import DuckDbAccessor
from serviette.server.main import create_app
from tests.conftest import fake_embedding, write_duckdb_rows

SHARED = [f"Article {i}. Unchanged provision number {i}." for i in range(6)]
# path -> chunk texts. The two "law" files are versions of one text: six
# identical articles, one edited passage, one passage only in the newer file.
CORPUS = {
    "/docs/law_2023.txt": [*SHARED, "The notice period is 30 days."],
    "/docs/law_2024.txt": [
        *SHARED,
        "The notice period is 14 days.",
        "Article 9. A new appeals procedure is introduced.",
    ],
    "/docs/journal.pdf": ["Monday: the pump was replaced.", "Tuesday: no incidents."],
    "/archive/journal.pdf": ["An older journal kept in the archive."],
    "/docs/menu.md": ["Soup of the day is borscht."],
}
# Force the two edited passages to be near-identical vectors (a pair) —
# fake embeddings of different strings are otherwise unrelated.
EDITED = {"The notice period is 30 days.", "The notice period is 14 days."}


def _embedding(text: str) -> list[float]:
    return fake_embedding("notice period" if text in EDITED else text)


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "store.duckdb"
    rows = []
    for doc, (doc_path, texts) in enumerate(CORPUS.items()):
        for i, text in enumerate(texts):
            rows.append(
                {
                    "id": f"{doc}-{i}",
                    "text": text,
                    "metadata": {
                        "path": doc_path,
                        "modified_at": 1_700_000_000 + doc,
                        "size": 100 * (doc + 1),
                    },
                    "embedding": _embedding(text),
                }
            )
    write_duckdb_rows(path, rows)
    return path


class _ScriptedLLM:
    """Refuses ordinary questions (the no-answer marker) unless ``answers``;
    ``route`` is its reply to the routing call. Records every call."""

    def __init__(self, route: str = "", answers: bool = False):
        self.route = route
        self.answers = answers
        self.calls: list[dict] = []
        self.raw_prompts: list[str] = []

    async def complete(self, query, context, *, system_prompt=None):
        self.calls.append({"context": list(context), "system_prompt": system_prompt})
        ordinary = system_prompt is not None and "reply exactly" in system_prompt
        if ordinary and not self.answers:
            return "No information found"
        return f"answer from {len(context)} items"

    async def raw(self, prompt):
        self.raw_prompts.append(prompt)
        return self.route

    async def close(self):
        return None


def _ask(store_path, embedder, llm, query, *, rag=None, k=2):
    config = ServietteConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        rag=rag,
        server=ServerConfig(serve_frontend=False),
    )
    accessor = DuckDbAccessor(config.vector_db)
    app = create_app(config, embedder=embedder, accessor=accessor, llm=llm)
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": query, "k": k})
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Catalog (accessor)
# ---------------------------------------------------------------------------


async def test_duckdb_catalog_lists_documents_and_their_chunks(store_path):
    accessor = DuckDbAccessor(DuckDbConfig(type="duckdb", path=str(store_path)))
    catalog = {e["id"]: e for e in await accessor.list_documents()}
    assert set(catalog) == set(CORPUS)
    assert catalog["/docs/law_2024.txt"]["chunks"] == 8
    assert catalog["/docs/menu.md"]["metadata"]["path"] == "/docs/menu.md"

    chunks = await accessor.document_chunks("/docs/journal.pdf", with_embeddings=True)
    assert {c["text"] for c in chunks} == set(CORPUS["/docs/journal.pdf"])
    assert all(len(c["embedding"]) > 0 for c in chunks)
    await accessor.close()


async def test_generic_catalog_matches_the_native_one(store_path):
    """Backends without a native query build the catalog from the full scan
    (KeywordHybridMixin); it must agree with DuckDB's SQL."""

    from serviette.server.hybrid import KeywordHybridMixin

    accessor = DuckDbAccessor(DuckDbConfig(type="duckdb", path=str(store_path)))
    native = {e["id"]: e["chunks"] for e in await accessor.list_documents()}
    generic = {
        e["id"]: e["chunks"] for e in await KeywordHybridMixin._catalog_scan(accessor)
    }
    assert generic == native
    chunks = await KeywordHybridMixin.document_chunks(accessor, "/docs/menu.md")
    assert [c["text"] for c in chunks] == CORPUS["/docs/menu.md"]
    await accessor.close()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

CATALOG = [{"id": path, "metadata": {"path": path}, "chunks": 1} for path in CORPUS]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("what does menu.md say?", ["menu.md"]),
        ("Расскажи, что написано в файле MENU.MD", ["menu.md"]),
        ("compare law_2024.txt with law_2023.txt", ["law_2024.txt", "law_2023.txt"]),
        ("open /archive/journal.pdf please", ["/archive/journal.pdf"]),
        # A file that is not indexed is still a mention (reported as missing).
        ("что написано в Журнал.pdf и menu.md?", ["Журнал.pdf", "menu.md"]),
        ("is node.js mentioned anywhere?", []),
        # A bare stem is an ordinary word, not a file reference.
        ("what is on the menu today?", []),
        # A longer name is its own (unknown) file, not a mention of menu.md.
        ("see oldmenu.md", ["oldmenu.md"]),
    ],
)
def test_mentioned_names_needs_a_full_file_name(query, expected):
    assert docmode.mentioned_names(query, CATALOG) == expected


def test_resolve_names_exact_stem_close_ambiguous_and_missing():
    resolution = docmode.resolve_names(
        ["menu.md", "law_2023", "law_2024.tx", "journal.pdf", "budget.xlsx"], CATALOG
    )
    assert [e["id"] for e in resolution.found] == [
        "/docs/menu.md",
        "/docs/law_2023.txt",
        "/docs/law_2024.txt",
    ]
    assert resolution.ambiguous == {
        "journal.pdf": ["/docs/journal.pdf", "/archive/journal.pdf"]
    }
    assert "budget.xlsx" in resolution.missing


@pytest.mark.parametrize(
    "reply, mode, files",
    [
        ("MODE: catalog\nFILES:", "catalog", []),
        ("mode: documents\nfiles: a.pdf; 'b.pdf'", "documents", ["a.pdf", "b.pdf"]),
        ("MODE: search\nFILES: a.pdf", "search", []),
        ("", "search", []),
        ("I think this is about files", "search", []),
    ],
)
def test_parse_route_degrades_to_search(reply, mode, files):
    route = docmode.parse_route(reply)
    assert (route.mode, route.files) == (mode, files)


# ---------------------------------------------------------------------------
# /rag flow
# ---------------------------------------------------------------------------


def test_answered_question_costs_one_call(store_path, mock_server_embedder):
    llm = _ScriptedLLM(answers=True)
    body = _ask(store_path, mock_server_embedder, llm, "what is the soup?")
    assert body["mode"] == "search" and body["notice"] is None
    assert len(llm.calls) == 1 and llm.raw_prompts == []
    # The corpus card states the exact document count.
    assert "holds 5 documents" in llm.calls[0]["system_prompt"]


def test_corpus_card_can_be_turned_off(store_path, mock_server_embedder):
    llm = _ScriptedLLM(answers=True)
    rag = RagConfig(documents={"corpus_card": False})
    _ask(store_path, mock_server_embedder, llm, "what is the soup?", rag=rag)
    assert "holds 5 documents" not in llm.calls[0]["system_prompt"]


def test_disabled_mode_never_routes(store_path, mock_server_embedder):
    llm = _ScriptedLLM(route="MODE: catalog")
    rag = RagConfig(documents={"enabled": False})
    body = _ask(store_path, mock_server_embedder, llm, "see menu.md", rag=rag)
    assert body["mode"] == "search"
    assert llm.raw_prompts == [] and len(llm.calls) == 1


def test_failed_search_routes_to_the_catalog(store_path, mock_server_embedder):
    llm = _ScriptedLLM(route="MODE: catalog\nFILES:")
    body = _ask(store_path, mock_server_embedder, llm, "which files do you have?")
    assert body["mode"] == "catalog" and body["sources"] == []
    # failed attempt + routing + catalog answer; routing sees names, not chunks
    assert len(llm.calls) == 2 and len(llm.raw_prompts) == 1
    assert "- menu.md" in llm.raw_prompts[0]
    assert "borscht" not in llm.raw_prompts[0]
    listing = "\n".join(llm.calls[1]["context"])
    assert "Indexed documents: 5" in listing
    assert all(path in listing for path in CORPUS)
    assert llm.calls[1]["system_prompt"] == docmode.CATALOG_SYSTEM_PROMPT


def test_catalog_listing_respects_the_budget():
    catalog = [
        {"id": f"/docs/file_{i:04d}.txt", "metadata": {"modified_at": i}, "chunks": 1}
        for i in range(500)
    ]
    totals, listing = docmode.catalog_context(catalog, budget=2_000)
    assert "Indexed documents: 500" in totals
    assert len(listing) < 2_300
    assert "of 500" in listing and "incomplete" in listing
    # newest first
    assert listing.index("file_0499") < listing.index("file_0498")


def test_search_verdict_hands_back_to_the_adaptive_loop(
    store_path, mock_server_embedder
):
    llm = _ScriptedLLM(route="MODE: search")
    rag = RagConfig(adaptive={"factor": 2, "max_iterations": 3})
    body = _ask(store_path, mock_server_embedder, llm, "who?", rag=rag, k=1)
    assert body["mode"] == "search"
    # Three growing attempts, but the routing call happened only once.
    assert [len(c["context"]) for c in llm.calls] == [1, 2, 4]
    assert len(llm.raw_prompts) == 1
    assert body["notice"] == "No answer was found in the 5 indexed documents."


def test_document_verdict_stops_the_adaptive_loop(store_path, mock_server_embedder):
    llm = _ScriptedLLM(route="MODE: documents\nFILES: menu")
    rag = RagConfig(adaptive={"factor": 2, "max_iterations": 4})
    body = _ask(store_path, mock_server_embedder, llm, "what's on the menu?", rag=rag)
    assert body["mode"] == "document"
    assert body["documents"] == ["/docs/menu.md"]
    assert len(llm.calls) == 2  # one failed attempt, one document answer


def test_named_file_is_answered_without_a_search(store_path, mock_server_embedder):
    llm = _ScriptedLLM()
    body = _ask(store_path, mock_server_embedder, llm, "что написано в menu.md?")
    assert body["mode"] == "document" and body["documents"] == ["/docs/menu.md"]
    assert len(llm.calls) == 1 and llm.raw_prompts == []
    header, *excerpts = llm.calls[0]["context"]
    assert "menu.md" in header and "whole document is shown" in header
    assert excerpts == CORPUS["/docs/menu.md"]
    assert [s["text"] for s in body["sources"]] == CORPUS["/docs/menu.md"]
    assert body["notice"] is None


def test_large_document_is_cut_to_the_budget():
    entry = {"id": "/docs/big.txt", "metadata": {"path": "/docs/big.txt"}, "chunks": 50}
    chunks = [
        {"text": f"part {i} " + "x" * 200, "metadata": {}, "embedding": fake_embedding(f"p{i}")}
        for i in range(50)
    ]
    chunks[17]["embedding"] = fake_embedding("the question")
    built = docmode.single_document_context(
        entry, chunks, fake_embedding("the question"), budget=1_000
    )
    assert sum(len(text) for text in built.context[1:]) <= 1_000
    assert built.context[1].startswith("part 17 ")  # closest to the question first
    assert "of its 50 parts" in built.context[0]
    assert built.notice == f"Answered from {len(built.sources)} of 50 parts of big.txt."


def test_two_versions_are_compared_as_a_diff(store_path, mock_server_embedder):
    llm = _ScriptedLLM()
    body = _ask(
        store_path,
        mock_server_embedder,
        llm,
        "what changed between law_2023.txt and law_2024.txt?",
    )
    assert body["mode"] == "compare"
    assert body["documents"] == ["/docs/law_2023.txt", "/docs/law_2024.txt"]
    assert len(llm.calls) == 1 and llm.raw_prompts == []
    header, *blocks = llm.calls[0]["context"]
    totals = "1 passages changed, 1 only in law_2024.txt, 0 only in law_2023.txt"
    assert totals in header and totals in body["notice"]
    # Only the differences reach the model — none of the shared articles.
    assert len(blocks) == 2
    changed = next(b for b in blocks if b.startswith("[changed]"))
    assert "30 days" in changed and "14 days" in changed
    added = next(b for b in blocks if b.startswith("[only in law_2024.txt]"))
    assert "appeals" in added
    assert not any("Unchanged provision" in block for block in blocks)


def test_diff_larger_than_the_budget_reports_what_is_shown():
    entries = [
        {"id": f"/docs/{name}", "metadata": {}, "chunks": 40} for name in ("a.txt", "b.txt")
    ]
    shared = [{"text": f"the same paragraph number {i}", "metadata": {}} for i in range(30)]
    only_b = [{"text": f"new {i} " + "y" * 300, "metadata": {}} for i in range(10)]
    built = docmode.compare_context(
        entries, [shared, shared + only_b], fake_embedding("q"), budget=1_000
    )
    assert "0 passages changed, 10 only in b.txt, 0 only in a.txt" in built.notice
    shown = len(built.context) - 1
    assert 0 < shown < 10
    assert f"Showing {shown} of 10 differences" in built.notice
    assert sum(len(block) for block in built.context[1:]) <= 1_000


def _windows(text: str, size: int) -> list[dict]:
    """Chunk ``text`` the way a token-window splitter does: fixed-size
    overlapping pieces that ignore line boundaries."""

    return [
        {"text": text[i : i + size], "metadata": {}}
        for i in range(0, len(text), size // 2)
    ]


def test_versions_are_recognized_across_shifted_chunks_and_extraction_noise():
    """Real versions never share whole chunks: an insertion shifts every
    later window, and two PDFs of one text extract with different spacing,
    soft hyphens and page headers. The diff must still be the actual edits."""

    paragraphs = [
        f"Section {i}. The classification of substance number {i} follows annex {i}."
        for i in range(60)
    ]
    old = "\n".join(
        [f"02008R1272 - FR - 01.05.2026 - 029.001 - {n}\n{p}" for n, p in enumerate(paragraphs)]
    )
    edited = list(paragraphs)
    edited[40] = edited[40].replace("follows annex 40", "follows the new annex 41")
    edited.insert(3, "Section 2a. A wholly new provision on digital labelling is inserted.")
    new = "\n".join(
        [
            f"02008R1272 - FR - 01.01.2027 - 032.001 - {n}\n{p}"
            for n, p in enumerate(edited)
        ]
    )
    # the newer file extracts with a soft hyphen and doubled spaces
    new = new.replace("classification", "classi\u00adfication").replace(" of ", "  of ")

    entries = [{"id": f"/docs/{n}", "metadata": {}, "chunks": 1} for n in ("old.pdf", "new.pdf")]
    chunks = [_windows(old, 500), _windows(new, 500)]
    assert not {c["text"] for c in chunks[0]} & {c["text"] for c in chunks[1]}

    built = docmode.compare_context(entries, chunks, fake_embedding("q"), budget=10_000)
    assert "share most of their text" in built.context[0]
    body = "\n".join(built.context[1:])
    assert "digital labelling" in body
    assert "follows the new annex 41" in body
    # page headers and untouched sections are not differences
    assert "029.001" not in body and "032.001" not in body
    assert "substance number 7 " not in body


def test_request_budget_caps_everything_sent_to_the_llm(store_path, mock_server_embedder):
    """max_request_chars bounds the request as a whole: the adaptive loop
    stops growing once the next attempt no longer fits."""

    llm = _ScriptedLLM(route="MODE: search")
    rag = RagConfig(
        adaptive={"factor": 2, "max_iterations": 4},
        documents={"max_request_chars": 2_000},
    )
    body = _ask(store_path, mock_server_embedder, llm, "who?", rag=rag, k=1)
    sent = sum(
        len(c["system_prompt"] or "") + sum(map(len, c["context"])) for c in llm.calls
    ) + sum(map(len, llm.raw_prompts))
    assert sent <= 2_000
    assert len(llm.calls) < 4
    assert "LLM budget" in body["notice"]


def test_budget_fit_trims_then_refuses():
    budget = docmode.Budget(1_000)
    assert budget.fit(100, ["a" * 300, "b" * 300, "c" * 400]) == ["a" * 300, "b" * 300]
    assert budget.exhausted and budget.left == 300
    assert budget.fit(100, ["d" * 300]) is None
    assert not budget.take(400) and budget.take(300)


def test_unrelated_documents_share_the_budget(store_path, mock_server_embedder):
    llm = _ScriptedLLM()
    body = _ask(
        store_path, mock_server_embedder, llm, "compare menu.md and law_2024.txt"
    )
    assert body["mode"] == "compare"
    header, *excerpts = llm.calls[0]["context"]
    assert "share little or no identical text" in header
    assert any(e.startswith("[menu.md] ") for e in excerpts)
    assert any(e.startswith("[law_2024.txt] ") for e in excerpts)


def test_unknown_document_is_reported_with_suggestions(
    store_path, mock_server_embedder
):
    llm = _ScriptedLLM(route="MODE: documents\nFILES: menu_old.md")
    body = _ask(store_path, mock_server_embedder, llm, "what is in the old menu?")
    assert body["mode"] == "unresolved" and body["sources"] == []
    assert 'No indexed document is named "menu_old.md"' in body["notice"]
    assert "menu.md" in body["notice"]
    # The LLM only words the system's facts; it is not asked to answer.
    assert llm.calls[-1]["system_prompt"] == docmode.EXPLAIN_SYSTEM_PROMPT
    assert llm.calls[-1]["context"] == [body["notice"]]


def test_named_file_that_is_not_indexed_is_reported_without_a_search(
    store_path, mock_server_embedder
):
    llm = _ScriptedLLM()
    body = _ask(store_path, mock_server_embedder, llm, "что написано в журнал.pdf?")
    assert body["mode"] == "unresolved"
    assert 'No indexed document is named "журнал.pdf"' in body["notice"]
    assert "journal.pdf" in body["notice"]  # what there is, since nothing is close
    assert len(llm.calls) == 1 and llm.raw_prompts == []


def test_context_chunks_carry_their_file_name(store_path, mock_server_embedder):
    llm = _ScriptedLLM(answers=True)
    _ask(store_path, mock_server_embedder, llm, "Soup of the day is borscht.", k=1)
    assert llm.calls[0]["context"] == ["[file: menu.md]\nSoup of the day is borscht."]

    llm = _ScriptedLLM(answers=True)
    rag = RagConfig(documents={"source_labels": False})
    _ask(store_path, mock_server_embedder, llm, "Soup of the day is borscht.", rag=rag, k=1)
    assert llm.calls[0]["context"] == ["Soup of the day is borscht."]


def test_ambiguous_name_asks_for_the_path(store_path, mock_server_embedder):
    llm = _ScriptedLLM()
    body = _ask(store_path, mock_server_embedder, llm, "summarize journal.pdf")
    assert body["mode"] == "unresolved"
    assert "/docs/journal.pdf" in body["notice"]
    assert "/archive/journal.pdf" in body["notice"]


def test_full_path_disambiguates(store_path, mock_server_embedder):
    llm = _ScriptedLLM()
    body = _ask(store_path, mock_server_embedder, llm, "summarize /archive/journal.pdf")
    assert body["mode"] == "document"
    assert body["documents"] == ["/archive/journal.pdf"]


def test_backend_without_catalog_keeps_the_ordinary_path(
    store_path, mock_server_embedder
):
    class NoCatalogAccessor(DuckDbAccessor):
        supports_catalog = False

        async def list_documents(self):
            return await AsyncVectorAccessor.list_documents(self)

    config = ServietteConfig(
        vector_db=DuckDbConfig(type="duckdb", path=str(store_path)),
        embedder=EmbedderConfig(type="openai"),
        server=ServerConfig(serve_frontend=False),
    )
    llm = _ScriptedLLM(route="MODE: catalog")
    app = create_app(
        config,
        embedder=mock_server_embedder,
        accessor=NoCatalogAccessor(config.vector_db),
        llm=llm,
    )
    with TestClient(app) as client:
        resp = client.post("/api/v1/rag", json={"query": "see menu.md", "k": 1})
    assert resp.json()["mode"] == "search"
    assert llm.calls[0]["system_prompt"] is None and llm.raw_prompts == []
