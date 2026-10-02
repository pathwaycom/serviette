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
documents that share most of their lines, which :func:`compare_context`
detects and turns into a diff.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import math
import posixpath
import re
import unicodedata
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
    "provided material, which the system prepared by comparing the documents: "
    "the first item says what was compared and gives exact totals, the rest "
    "are the differences themselves. Describe what actually differs — what "
    "was added, removed or changed, with specifics (section numbers, names, "
    "figures), grouped by topic; never answer with the totals alone. Ignore "
    "differences that are only spacing, hyphenation or punctuation. End with "
    "the totals exactly as given and, if not all differences are shown, say "
    "that the description covers only part of them. Answer in the language "
    "of the question."
)

# Goes with :func:`labeled`. Without it the model answers "which documents
# do you have" from the handful of file names it happens to see — a confident,
# incomplete list. The marker hands such questions to the catalog instead.
SOURCE_LABELS_NOTE = (
    "Each context excerpt starts with the name of the file it comes from. "
    "The excerpts are a small sample: the file names on them are not the "
    "list of indexed documents, and a few excerpts cannot show how whole "
    "documents differ. If the question asks which documents exist, asks to "
    "list them, asks about their dates or sizes, or asks what differs or "
    "changed between documents or versions of a document, reply exactly "
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
# Two documents sharing at least this fraction of their lines are compared
# as versions of one text (a diff); below it they are different documents.
_VERSION_OVERLAP = 0.3
# Lines shorter than this (after squashing) are not compared: codes and
# table cells repeat throughout a document and say nothing on their own.
_MIN_LINE = 12
# From this many lines on, a chunk's first and last line are taken to be cut
# by the splitter and are not compared (shorter chunks are whole paragraphs).
_EDGE_LINES = 3
# A line whose digit-masked form occurs at least this often in *both*
# documents is running matter (page headers, numbered rows): its digits are
# ignored, so a page number or an edition date in a header is not a change.
_RUNNING_LINES = 20
# A passage only in one document and a passage only in the other that share
# at least this fraction of their words are one passage edited.
_CHANGED_SIMILARITY = 0.5
# Pairing is quadratic in the differing passages; past this many pairs they
# are reported as plain additions/removals instead.
_MAX_PAIRINGS = 400_000
# Differences shorter than this are shown after the longer ones: a stray
# line is more often an artifact of text extraction than a change.
_SUBSTANTIAL = 80


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


class Budget:
    """The ceiling on what one ``/rag`` request may send to the LLM, in
    characters, across all of its calls (attempts, routing, the answer).

    Every call is charged before it is made; one that does not fit is cut
    down to what is left, and once too little is left no call is made at
    all. The number of calls is bounded elsewhere (each step of the document
    mode runs at most once) — this bounds their combined size, whatever the
    documents, ``k`` or the adaptive loop ask for.
    """

    # Below this a call cannot carry a useful context any more.
    _MIN_CALL = 500

    def __init__(self, limit: int) -> None:
        self.left = limit
        self.exhausted = False

    def take(self, size: int) -> bool:
        """Charge a call of a fixed ``size``; False when it does not fit."""

        if size > self.left:
            self.exhausted = True
            return False
        self.left -= size
        return True

    def fit(self, fixed: int, context: list[str]) -> list[str] | None:
        """Charge a call with ``fixed`` characters of prompt plus as many
        leading ``context`` items as still fit. ``None`` — no call — when
        not even one item does."""

        room = self.left - fixed
        kept: list[str] = []
        used = 0
        for item in context:
            if used + len(item) > room:
                break
            kept.append(item)
            used += len(item)
        if len(kept) < len(context):
            self.exhausted = True
        if (context and not kept) or room < 0 or self.left < self._MIN_CALL:
            self.exhausted = True
            return None
        self.left -= fixed + used
        return kept


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
    llm: AsyncLLM,
    query: str,
    catalog: list[dict[str, Any]],
    max_listed: int,
    budget: Budget | None = None,
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
    if budget is not None and not budget.take(len(prompt)):
        return Route()
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


def _squash(text: str) -> str:
    """``text`` reduced to what must match for two extractions of the same
    passage to be equal: letters and digits only, one case, no diacritics.
    PDF text extraction varies between two files of one document in exactly
    the rest — spacing, soft hyphens, the kind of apostrophe or dash."""

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if ch.isalnum()).casefold()


_DIGITS = re.compile(r"\d+")


class _Lines:
    """One document as comparable lines, independent of how it was chunked.

    Chunk boundaries move when text is inserted (token windows), so whole
    chunks of two versions rarely match; lines do. The store keeps no chunk
    order, and none is needed: a line is looked up in the other document,
    not aligned with it.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        # squashed line -> index of the first chunk containing it
        self.lines: dict[str, int] = {}
        # per chunk: its lines as (squashed, raw), in order
        self.by_chunk: list[list[tuple[str, str]]] = []
        squashed_chunks = []
        for index, chunk in enumerate(chunks):
            rows = [(_squash(raw), raw.strip()) for raw in chunk["text"].splitlines()]
            squashed_chunks.append("".join(row[0] for row in rows))
            if len(rows) >= _EDGE_LINES:
                # A window splitter cuts the first and last line of a chunk
                # mid-line; such fragments match nothing. The chunk overlap
                # carries the same lines whole in the neighbouring chunk.
                rows = rows[1:-1]
            for squashed, _raw in rows:
                if len(squashed) >= _MIN_LINE:
                    self.lines.setdefault(squashed, index)
            self.by_chunk.append(rows)
        # The whole text, for finding a line that the other file wraps
        # differently (so it is no line of its own there).
        self.text = "\x00".join(squashed_chunks)
        self.masked = Counter(_DIGITS.sub("#", line) for line in self.lines)
        # Running matter also gets glued to the front of the line after it
        # ("<page header> <first line of the page>").
        running = [
            mask for mask, count in self.masked.items() if count >= _RUNNING_LINES
        ]
        self._running_prefix = (
            re.compile(
                "|".join(
                    re.escape(mask).replace("\\#", r"\d+")
                    for mask in sorted(running, key=len, reverse=True)
                )
            )
            if running
            else None
        )

    def body(self, squashed: str) -> str:
        """``squashed`` without a running header glued to its front."""

        if self._running_prefix is not None:
            match = self._running_prefix.match(squashed)
            if match and len(squashed) - match.end() >= _MIN_LINE:
                return squashed[match.end() :]
        return squashed

    def overlap(self, other: _Lines) -> float:
        total = len(self.lines) + len(other.lines)
        if not total:
            return 0.0
        return 2 * len(self.lines.keys() & other.lines.keys()) / total

    def passages_missing_from(self, other: _Lines) -> list[tuple[str, int]]:
        """Runs of consecutive lines of this document that the other does
        not contain, as ``(text, chunk index)``."""

        def present(squashed: str) -> bool:
            if squashed in other.lines:
                return True
            mask = _DIGITS.sub("#", squashed)
            if (
                self.masked[mask] >= _RUNNING_LINES
                and other.masked[mask] >= _RUNNING_LINES
            ):
                return True
            if squashed in other.text:
                return True
            body = self.body(squashed)
            return body is not squashed and body in other.text

        verdicts: dict[str, bool] = {}
        reported: set[str] = set()
        passages: list[tuple[str, int]] = []
        for index, rows in enumerate(self.by_chunk):
            run: list[str] = []
            for squashed, raw in [*rows, ("", "")]:
                missing = False
                if len(squashed) >= _MIN_LINE and squashed not in reported:
                    if squashed not in verdicts:
                        verdicts[squashed] = present(squashed)
                    missing = not verdicts[squashed]
                if missing:
                    reported.add(squashed)  # chunk overlap repeats lines
                    run.append(raw)
                elif run:
                    passages.append(("\n".join(run), index))
                    run = []
        return passages


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w{3,}", text.casefold()))


def _pair_changed(
    only_a: list[tuple[str, int]], only_b: list[tuple[str, int]]
) -> tuple[list[tuple[tuple[str, int], tuple[str, int]]], list, list]:
    """Split differing passages into (changed pairs, removed, added) by
    greedy best word-overlap matching."""

    if not only_a or not only_b or len(only_a) * len(only_b) > _MAX_PAIRINGS:
        return [], only_a, only_b
    words_a = [_words(text) for text, _ in only_a]
    words_b = [_words(text) for text, _ in only_b]
    scored = []
    for i, wa in enumerate(words_a):
        if not wa:
            continue
        for j, wb in enumerate(words_b):
            if not wb:
                continue
            similarity = len(wa & wb) / len(wa | wb)
            if similarity >= _CHANGED_SIMILARITY:
                scored.append((similarity, i, j))
    scored.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _similarity, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((only_a[i], only_b[j]))
    removed = [p for i, p in enumerate(only_a) if i not in used_a]
    added = [p for j, p in enumerate(only_b) if j not in used_b]
    return pairs, removed, added


def compare_context(
    entries: list[dict[str, Any]],
    chunks: list[list[dict[str, Any]]],
    query_embedding: list[float],
    budget: int,
) -> DocumentContext:
    """Two or more documents within one budget.

    Exactly two documents that share most of their lines are versions of
    one text: what they share is dropped and the model sees only the
    differences (edited passages paired up, then additions and removals),
    with the exact totals computed here. Anything else — unrelated documents,
    or more than two — gets each document's parts closest to the question,
    the budget split evenly.
    """

    names = [document_name(e) for e in entries]
    if len(entries) == 2:
        lines = [_Lines(chunks[0]), _Lines(chunks[1])]
        if lines[0].overlap(lines[1]) >= _VERSION_OVERLAP:
            return _diff_context(entries, chunks, lines, query_embedding, budget)

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
    lines: list[_Lines],
    query_embedding: list[float],
    budget: int,
) -> DocumentContext:
    name_a, name_b = document_name(entries[0]), document_name(entries[1])
    pairs, removed, added = _pair_changed(
        lines[0].passages_missing_from(lines[1]),
        lines[1].passages_missing_from(lines[0]),
    )

    # Which differences to show when not all fit: the substantial ones
    # first, and among those the ones in the parts of the documents closest
    # to the question.
    query_unit = _unit(query_embedding)
    relevance: dict[tuple[int, int], float] = {}

    def score(doc: int, chunk: int) -> float:
        if (doc, chunk) not in relevance:
            relevance[doc, chunk] = _dot(
                query_unit, _unit(chunks[doc][chunk].get("embedding"))
            )
        return relevance[doc, chunk]

    # (text, [(doc, chunk)], relevance)
    blocks: list[tuple[str, list[tuple[int, int]], float]] = []
    for (old, old_chunk), (new, new_chunk) in pairs:
        blocks.append(
            (
                f"[changed]\n{name_a}: {old}\n{name_b}: {new}",
                [(0, old_chunk), (1, new_chunk)],
                max(score(0, old_chunk), score(1, new_chunk)),
            )
        )
    blocks.extend(
        (f"[only in {name_b}] {text}", [(1, chunk)], score(1, chunk))
        for text, chunk in added
    )
    blocks.extend(
        (f"[only in {name_a}] {text}", [(0, chunk)], score(0, chunk))
        for text, chunk in removed
    )
    blocks.sort(key=lambda block: (len(block[0]) >= _SUBSTANTIAL, block[2]), reverse=True)

    shown: list[str] = []
    sources: list[dict[str, Any]] = []
    cited: set[tuple[int, int]] = set()
    used = 0
    for text, origins, _relevance in blocks:
        if used + len(text) > budget:
            if shown:
                continue  # a shorter difference further down may still fit
            text = text[:budget]
        shown.append(text)
        used += len(text)
        for doc, chunk in origins:
            if (doc, chunk) not in cited:
                cited.add((doc, chunk))
                source = chunks[doc][chunk]
                sources.append(
                    {
                        "text": source["text"],
                        "metadata": source.get("metadata") or {},
                        "score": score(doc, chunk),
                    }
                )

    totals = (
        f"{len(pairs)} passages changed, {len(added)} only in {name_b}, "
        f"{len(removed)} only in {name_a}"
    )
    header = (
        "Comparison of two documents that share most of their text. "
        "Everything they share was left out; below are the differences.\n"
        f"- {_describe(entries[0])}\n- {_describe(entries[1])}\n"
        f"Totals: {totals}."
    )
    notice = f"Compared {name_a} and {name_b}: {totals}."
    if len(shown) < len(blocks):
        partial = (
            f" Showing {len(shown)} of {len(blocks)} differences (the ones "
            "closest to the question)."
        )
        header += partial
        notice += partial
    if not blocks:
        header += " No differences were found in the indexed text."
    return DocumentContext(context=[header, *shown], sources=sources, notice=notice)
