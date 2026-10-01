"""Document-level questions for ``/rag`` (``rag.documents``).

Top-k chunk retrieval answers "what does the corpus say about X". It cannot
answer questions about documents as objects: how many there are, which ones
exist, what a named file says as a whole, how two files differ. Everything
here is the deterministic half of that mode — the server decides *when* to
use it (see ``create_app``):

* the catalog helpers turn the accessor's per-document listing into names,
  resolve the names a question mentions, and render the listing as context;
* :func:`arbitrate` is the one short LLM call made after a failed ordinary
  attempt, deciding between "search wider" and this mode;
* the context builders pick what the answering call sees for one document
  or for a comparison, always within a fixed character budget — the cost of
  a request does not grow with the size of the documents.

There is no notion of a document *version*: two versions are simply two
documents that share most of their chunks, which :func:`compare_context`
detects and turns into a diff.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import math
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from serviette.server.accessors.abstract import document_key
from serviette.server.llm import AsyncLLM

logger = logging.getLogger(__name__)

CATALOG_SYSTEM_PROMPT = (
    "You answer questions about a collection of indexed documents using only "
    "the catalog below. The catalog is computed by the system and is "
    "authoritative: never count or list documents in any other way. It covers "
    "indexed documents only — files that produced no text are not in it, so "
    "say \"indexed documents\" rather than claiming to know every file in a "
    "folder. Answer in the language of the question."
)

DOCUMENT_SYSTEM_PROMPT = (
    "Answer the user's question about the document below using only the "
    "provided excerpts, all of which come from that document. Their order is "
    "not guaranteed to match the document. If the header says only part of "
    "the document is shown, say that the answer is based on a part of it. "
    "Answer in the language of the question."
)

COMPARE_SYSTEM_PROMPT = (
    "Answer the user's question about the documents below using only the "
    "provided material, which the system prepared by comparing the documents. "
    "The first item states what was compared and the exact totals — report "
    "those totals as given, and if not all differences are shown, say so. "
    "Answer in the language of the question."
)

# Goes with :func:`labeled`. Without it the model answers "which documents
# do you have" from the handful of file names it happens to see — a confident,
# incomplete list. The marker hands such questions to the catalog instead.
SOURCE_LABELS_NOTE = (
    "Each context excerpt starts with the name of the file it comes from. "
    "The excerpts are a small sample: the file names on them are not the "
    "list of indexed documents. If the question asks which documents exist, "
    "asks to list them, or asks about their dates or sizes, reply exactly "
    '"{marker}".'
)

EXPLAIN_SYSTEM_PROMPT = (
    "The user's question could not be answered as asked. Using only the facts "
    "below, tell the user briefly what the problem is and what they can do "
    "instead. Do not attempt to answer the question itself. Answer in the "
    "language of the question."
)

_ARBITER_PROMPT = """\
You route a question asked to a document-search assistant. A first search \
over document contents found no answer.

Indexed documents ({shown}):
{names}

Reply with exactly two lines:
MODE: search | catalog | documents
FILES: names separated by ";" (empty unless MODE is documents)

search — the answer would be inside document contents and a wider search \
may still find it. Also choose this for a question on a topic that has \
nothing to do with these documents, and whenever unsure.
catalog — the question is about the collection itself: how many documents \
there are, which ones exist, their names, dates, sizes or types.
documents — the question is about one or more specific documents as a \
whole: what a document says, a summary of it, or how documents differ. Put \
them in FILES, copying names from the list when they match, otherwise as \
the user wrote them.

Question: {query}"""

# Extensions that mark a word in a question as a file name even when no such
# file is indexed (the extensions of indexed files are added at run time).
_FILE_EXTENSIONS = {
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "rtf", "txt",
    "md", "csv", "html", "htm", "json", "xml", "eml", "epub",
}
# A misspelled name resolves to a document only at this similarity, and only
# when no other document comes within the margin.
_CLOSE_NAME = 0.85
_CLOSE_NAME_MARGIN = 0.05
# Two documents sharing at least this fraction of their chunks are compared
# as versions of one text (a diff); below it they are different documents.
_VERSION_OVERLAP = 0.3
# An unmatched chunk whose best counterpart in the other document is at least
# this similar is the same passage edited, not a removal plus an addition.
_CHANGED_SIMILARITY = 0.8
# Pairing is quadratic in the unmatched chunks; past this many pairs they are
# reported as plain additions/removals instead.
_MAX_PAIRINGS = 20_000


def document_name(entry: dict[str, Any]) -> str:
    """The name a person would use for a catalog entry: the source's own
    ``name`` (gdrive) or the last component of its path."""

    metadata = entry.get("metadata") or {}
    name = metadata.get("name")
    if isinstance(name, str) and name:
        return name
    key = str(entry["id"])
    return posixpath.basename(key.replace("\\", "/").rstrip("/")) or key


def _fold(value: str) -> str:
    return value.casefold().strip()


def _stem(name: str) -> str:
    return posixpath.splitext(name)[0]


def mentioned_names(query: str, catalog: list[dict[str, Any]]) -> list[str]:
    """Document names the question spells out literally, in question order.

    Deliberately strict — this check runs before any search and bypasses it:
    only a full path or a file name *with its extension* counts, bounded by
    non-word characters. A bare stem ("report") is too often an ordinary
    word; such references are left to :func:`arbitrate`. Names of files that
    are not indexed are returned as well (as written), for
    :func:`resolve_names` to report.
    """

    folded = _fold(query)
    found: dict[str, int] = {}
    for entry in catalog:
        name = document_name(entry)
        for candidate in (str(entry["id"]), name):
            needle = _fold(candidate)
            if "." not in posixpath.basename(needle):
                continue
            # Not preceded by a path separator either: "archive/journal.pdf"
            # must not also count as a mention of another folder's
            # "journal.pdf".
            match = re.search(
                r"(?<![\w./\\])" + re.escape(needle) + r"(?![\w])", folded
            )
            if match:
                found.setdefault(candidate, match.start())
                break
    # A file name that is not in the catalog is a mention too: the person
    # asked about a specific file, and "there is no such file" is the answer
    # — a search over other documents' contents would not be.
    extensions = _FILE_EXTENSIONS | {
        posixpath.splitext(_fold(document_name(e)))[1].lstrip(".") for e in catalog
    }
    extensions.discard("")
    known = {_fold(name) for name in found} | {
        _fold(document_name(e)) for e in catalog
    }
    file_like = re.compile(
        r"(?<![\w./\\])[\w\-]+\.(?:"
        + "|".join(re.escape(ext) for ext in sorted(extensions))
        + r")(?![\w])"
    )
    for match in file_like.finditer(folded):
        if match.group(0) not in known:
            found.setdefault(query[match.start() : match.end()], match.start())
    return sorted(found, key=found.__getitem__)


@dataclass
class Resolution:
    """Catalog entries for the names a question refers to."""

    found: list[dict[str, Any]] = field(default_factory=list)
    # name -> close catalog names offered instead
    missing: dict[str, list[str]] = field(default_factory=dict)
    # name -> ids of every entry it matches
    ambiguous: dict[str, list[str]] = field(default_factory=dict)


def resolve_names(names: list[str], catalog: list[dict[str, Any]]) -> Resolution:
    """Match ``names`` against the catalog: exactly (path, name, then stem),
    else by a single close match. Anything less certain is reported as
    missing with suggestions rather than guessed."""

    resolution = Resolution()
    seen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        if entry["id"] not in seen:
            seen.add(entry["id"])
            resolution.found.append(entry)

    labels = {entry["id"]: document_name(entry) for entry in catalog}
    for name in names:
        wanted = _fold(name)
        if not wanted:
            continue
        matches: list[dict[str, Any]] = []
        for key in (
            lambda e: _fold(str(e["id"])),
            lambda e: _fold(labels[e["id"]]),
            lambda e: _fold(_stem(labels[e["id"]])),
        ):
            matches = [e for e in catalog if key(e) == wanted]
            if matches:
                break
        if len(matches) == 1:
            add(matches[0])
            continue
        if len(matches) > 1:
            resolution.ambiguous[name] = [str(e["id"]) for e in matches]
            continue
        # No exact match: a typo is accepted only when one document is
        # clearly the closest — versions are named alike ("law_2023",
        # "law_2024"), and picking the wrong one silently is worse than asking.
        scored = sorted(
            (
                (
                    max(
                        difflib.SequenceMatcher(None, wanted, form).ratio()
                        for form in (
                            _fold(labels[entry["id"]]),
                            _fold(_stem(labels[entry["id"]])),
                        )
                    ),
                    labels[entry["id"]],
                    entry,
                )
                for entry in catalog
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best = scored[0] if scored else None
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if (
            best is not None
            and best[0] >= _CLOSE_NAME
            and best[0] - runner_up >= _CLOSE_NAME_MARGIN
        ):
            add(best[2])
            continue
        resolution.missing[name] = [
            label for score, label, _ in scored[:5] if score >= 0.5
        ]
    return resolution


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------


@dataclass
class Route:
    mode: str = "search"  # "search" | "catalog" | "documents"
    files: list[str] = field(default_factory=list)


_MODE_LINE = re.compile(r"^\s*MODE\s*:\s*([a-z]+)", re.IGNORECASE | re.MULTILINE)
_FILES_LINE = re.compile(r"^\s*FILES\s*:(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_route(reply: str) -> Route:
    mode_match = _MODE_LINE.search(reply)
    mode = mode_match.group(1).lower() if mode_match else "search"
    if mode not in ("catalog", "documents"):
        return Route()
    files: list[str] = []
    files_match = _FILES_LINE.search(reply)
    if files_match and mode == "documents":
        files = [
            part.strip().strip("\"'`")
            for part in files_match.group(1).split(";")
            if part.strip().strip("\"'`")
        ]
    return Route(mode=mode, files=files)


async def arbitrate(
    llm: AsyncLLM, query: str, catalog: list[dict[str, Any]], max_listed: int
) -> Route:
    """One short LLM call: does the question need a wider search or the
    document mode? Sees the question and document names only — never the
    retrieved context — so its cost is fixed. Degrades to "search" (the
    ordinary path) on any LLM hiccup or unparseable reply."""

    names = sorted(document_name(entry) for entry in catalog)
    shown = (
        f"{len(names)} total"
        if len(names) <= max_listed
        else f"{len(names)} total, first {max_listed} shown"
    )
    prompt = _ARBITER_PROMPT.format(
        shown=shown,
        names="\n".join(f"- {name}" for name in names[:max_listed]),
        query=query,
    )
    try:
        reply = await llm.raw(prompt)
    except Exception:  # noqa: BLE001 - routing must never break /rag
        logger.warning("document-mode arbiter failed; continuing with search")
        return Route()
    return parse_route(reply)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _timestamp(value: Any) -> str | None:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.datetime.fromtimestamp(
        seconds, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def _describe(entry: dict[str, Any]) -> str:
    """One catalog line: identity plus whatever the source reported."""

    metadata = entry.get("metadata") or {}
    name = document_name(entry)
    key = str(entry["id"])
    parts = [name if key == name else f"{name} ({key})"]
    modified = _timestamp(metadata.get("modified_at"))
    if modified:
        parts.append(f"modified {modified}")
    size = metadata.get("size")
    if isinstance(size, (int, float)) and size >= 0:
        parts.append(f"{int(size)} bytes")
    parts.append(f"{entry.get('chunks', 0)} indexed parts")
    return " | ".join(parts)


def corpus_card(catalog: list[dict[str, Any]]) -> str:
    """The one fact the answering prompt always carries in document mode."""

    return (
        f"The index currently holds {len(catalog)} documents "
        "(files that produced text). Never derive the number of documents "
        "from the context."
    )


def labeled(hits: list[dict[str, Any]]) -> list[str]:
    """Chunk texts prefixed with the name of the file they come from, so
    the model can tell documents apart — two versions of a file otherwise
    read as one self-contradicting text."""

    out = []
    for hit in hits:
        key = document_key(hit.get("metadata"))
        if key is None:
            out.append(hit["text"])
            continue
        name = document_name({"id": key, "metadata": hit.get("metadata")})
        out.append(f"[file: {name}]\n{hit['text']}")
    return out


def catalog_context(catalog: list[dict[str, Any]], budget: int) -> list[str]:
    """The catalog as context: exact totals first (computed here, never by
    the model), then as many document lines as the budget allows, newest
    first."""

    types = Counter(
        posixpath.splitext(document_name(e))[1].lower() or "(no extension)"
        for e in catalog
    )
    totals = [
        f"Indexed documents: {len(catalog)}",
        f"Indexed parts (chunks): {sum(e.get('chunks', 0) for e in catalog)}",
        "By type: "
        + ", ".join(f"{ext} {n}" for ext, n in sorted(types.items())),
    ]

    def modified(entry: dict[str, Any]) -> int:
        try:
            return int((entry.get("metadata") or {}).get("modified_at") or 0)
        except (TypeError, ValueError):
            return 0

    ordered = sorted(catalog, key=lambda e: (-modified(e), document_name(e)))
    lines: list[str] = []
    used = sum(len(t) for t in totals)
    for entry in ordered:
        line = "- " + _describe(entry)
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    if len(lines) == len(ordered):
        header = "Documents (complete list, most recently modified first):"
    else:
        header = (
            f"Documents (showing {len(lines)} of {len(ordered)}, most recently "
            "modified first — the list is incomplete, the totals are exact):"
        )
    return ["\n".join(totals), "\n".join([header, *lines])]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _unit(vector: list[float] | None) -> list[float] | None:
    if not vector:
        return None
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return None
    return [x / norm for x in vector]


def _dot(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _rank(
    chunks: list[dict[str, Any]], query_embedding: list[float]
) -> list[dict[str, Any]]:
    """``chunks`` as hits scored by similarity to the question, best first."""

    query_unit = _unit(query_embedding)
    hits = [
        {
            "text": chunk["text"],
            "metadata": chunk.get("metadata") or {},
            "score": _dot(query_unit, _unit(chunk.get("embedding"))),
        }
        for chunk in chunks
    ]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def _take(hits: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """The leading hits that fit ``budget`` characters (always at least one)."""

    taken: list[dict[str, Any]] = []
    used = 0
    for hit in hits:
        if taken and used + len(hit["text"]) > budget:
            break
        taken.append(hit)
        used += len(hit["text"])
    return taken


@dataclass
class DocumentContext:
    """What the answering call sees, and what the response reports."""

    context: list[str]
    sources: list[dict[str, Any]]
    notice: str | None = None


def single_document_context(
    entry: dict[str, Any],
    chunks: list[dict[str, Any]],
    query_embedding: list[float],
    budget: int,
) -> DocumentContext:
    """One document: all of it when it fits the budget, otherwise the parts
    closest to the question — and the header says which."""

    hits = _rank(chunks, query_embedding)
    taken = _take(hits, budget)
    header = "Document: " + _describe(entry)
    notice = None
    if len(taken) < len(hits):
        header += (
            f"\nOnly {len(taken)} of its {len(hits)} parts are shown (the ones "
            "closest to the question); the document is too large to show whole."
        )
        notice = (
            f"Answered from {len(taken)} of {len(hits)} parts of "
            f"{document_name(entry)}."
        )
    else:
        header += "\nThe whole document is shown."
    return DocumentContext(
        context=[header, *(h["text"] for h in taken)], sources=taken, notice=notice
    )


def _pair_changed(
    only_a: list[dict[str, Any]], only_b: list[dict[str, Any]]
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Split unmatched chunks into (changed pairs, removed, added) by greedy
    best-similarity matching on the stored embeddings."""

    if not only_a or not only_b or len(only_a) * len(only_b) > _MAX_PAIRINGS:
        return [], only_a, only_b
    units_a = [_unit(c.get("embedding")) for c in only_a]
    units_b = [_unit(c.get("embedding")) for c in only_b]
    scored = sorted(
        (
            (_dot(ua, ub), i, j)
            for i, ua in enumerate(units_a)
            for j, ub in enumerate(units_b)
            if ua is not None and ub is not None
        ),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for similarity, i, j in scored:
        if similarity < _CHANGED_SIMILARITY:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((only_a[i], only_b[j]))
    removed = [c for i, c in enumerate(only_a) if i not in used_a]
    added = [c for j, c in enumerate(only_b) if j not in used_b]
    return pairs, removed, added


def compare_context(
    entries: list[dict[str, Any]],
    chunks: list[list[dict[str, Any]]],
    query_embedding: list[float],
    budget: int,
) -> DocumentContext:
    """Two or more documents within one budget.

    Exactly two documents that share most of their chunks are versions of
    one text: identical chunks are dropped and the model sees only the
    differences (edited passages paired up, then additions and removals),
    with the exact totals computed here. Anything else — unrelated documents,
    or more than two — gets each document's parts closest to the question,
    the budget split evenly.
    """

    names = [document_name(e) for e in entries]
    if len(entries) == 2:
        texts_a = {_normalize(c["text"]) for c in chunks[0]}
        texts_b = {_normalize(c["text"]) for c in chunks[1]}
        shared = texts_a & texts_b
        total = len(texts_a) + len(texts_b)
        if total and 2 * len(shared) / total >= _VERSION_OVERLAP:
            return _diff_context(entries, chunks, shared, budget)

    share = max(budget // max(len(entries), 1), 1)
    context = [
        "Documents compared (they share little or no identical text, so each "
        "is represented by its parts closest to the question):\n"
        + "\n".join("- " + _describe(e) for e in entries)
    ]
    sources: list[dict[str, Any]] = []
    partial: list[str] = []
    for name, doc_chunks in zip(names, chunks):
        hits = _rank(doc_chunks, query_embedding)
        taken = _take(hits, share)
        if len(taken) < len(hits):
            partial.append(f"{len(taken)} of {len(hits)} parts of {name}")
        context.extend(f"[{name}] {h['text']}" for h in taken)
        sources.extend(taken)
    notice = "Compared using " + ", ".join(partial) + "." if partial else None
    if partial:
        context[0] += "\nShown: " + ", ".join(partial) + "."
    return DocumentContext(context=context, sources=sources, notice=notice)


def _diff_context(
    entries: list[dict[str, Any]],
    chunks: list[list[dict[str, Any]]],
    shared: set[str],
    budget: int,
) -> DocumentContext:
    name_a, name_b = document_name(entries[0]), document_name(entries[1])

    def unmatched(doc_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out = []
        for chunk in doc_chunks:
            text = _normalize(chunk["text"])
            if text not in shared and text not in seen:
                seen.add(text)
                out.append(chunk)
        return out

    pairs, removed, added = _pair_changed(unmatched(chunks[0]), unmatched(chunks[1]))

    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for old, new in pairs:
        blocks.append(
            (
                f"[changed]\n{name_a}: {old['text']}\n{name_b}: {new['text']}",
                [old, new],
            )
        )
    blocks.extend((f"[only in {name_b}] {c['text']}", [c]) for c in added)
    blocks.extend((f"[only in {name_a}] {c['text']}", [c]) for c in removed)

    shown: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0
    for text, block_chunks in blocks:
        if shown and used + len(text) > budget:
            break
        shown.append(text)
        used += len(text)
        sources.extend(
            {"text": c["text"], "metadata": c.get("metadata") or {}, "score": 0.0}
            for c in block_chunks
        )

    totals = (
        f"{len(shared)} identical parts, {len(pairs)} changed, "
        f"{len(added)} only in {name_b}, {len(removed)} only in {name_a}"
    )
    header = (
        "Comparison of two documents that share most of their text:\n"
        f"- {_describe(entries[0])}\n- {_describe(entries[1])}\n"
        f"Totals: {totals}."
    )
    notice = f"Compared {name_a} and {name_b}: {totals}."
    if len(shown) < len(blocks):
        partial = f" Showing {len(shown)} of {len(blocks)} differences."
        header += partial
        notice += partial
    if not blocks:
        header += " No differences were found in the indexed text."
    return DocumentContext(context=[header, *shown], sources=sources, notice=notice)
