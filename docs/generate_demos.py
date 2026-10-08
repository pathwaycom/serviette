#!/usr/bin/env python3
"""Generate the README demo GIF (terminal walkthrough + chat UI).

The frames are rendered with Pillow — no terminal recorder or browser needed —
so they regenerate deterministically anywhere:

    python docs/generate_demos.py

Output:
    docs/assets/demo.gif   `serviette demo` with its real output, the chat UI
                           answering a question, a live edit of a document in a
                           second terminal, and the same question answered anew

What is shown is what the programs print and answer. The terminal lines are
taken verbatim from a recorded `serviette demo` run (only the 5-second
heartbeats are thinned out), the chat answers are what gpt-4o-mini produced on
the bundled corpus before and after the edit, and the header statistics are the
real `/api/v1/stats` values. When the output of `demo`/`up` or the UI changes,
update the scripts below rather than inventing lines.

Requires Pillow (``pip install pillow``).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"

_FONT_CANDIDATES = {
    "mono": [
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
    "mono_bold": [
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ],
    "sans": [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "sans_bold": [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Fall back to regular mono/sans if a bold face is missing.
    base = kind.replace("_bold", "")
    if base != kind:
        return font(base, size)
    return ImageFont.load_default()


def save_gif(path: Path, frames: list[Image.Image], durations: list[int], colors: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    pal[0].save(
        path,
        save_all=True,
        append_images=pal[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {path}  ({path.stat().st_size // 1024} KB, {len(frames)} frames)")


# ---------------------------------------------------------------------------
# Terminal demo
# ---------------------------------------------------------------------------

T_W, T_H = 920, 560
T_BG = (13, 17, 23)
T_BAR = (32, 37, 46)
T_PROMPT = (88, 200, 120)
T_CMD = (236, 239, 244)
T_OUT = (148, 158, 170)
T_ACCENT = (129, 140, 248)
T_PAD = 24
T_LINE_H = 26
T_FONT = 17
T_FIRST_Y = 56

# Verbatim output of `serviette demo` (local embeddings, OPENAI_API_KEY set),
# thinned: the heartbeat repeats every 5 s, three of them are kept.
_INFO = "INFO "
SCRIPT_UP = [
    ("cmd", "serviette demo"),
    ("out", _INFO + "Using local sentence-transformers embeddings; answers are "
            "generated with OpenAI (OPENAI_API_KEY is set).", T_OUT),
    ("out", "", T_OUT),
    ("out", "─" * 72, T_OUT),
    ("out", " serviette demo — Lumina Coffee Systems", T_CMD),
    ("out", " Chat UI:  http://localhost:8989", T_CMD),
    ("out", "           (opens after the first indexing pass — wait for the", T_OUT),
    ("out", '            "Ready — open http://..." line below; the first run', T_OUT),
    ("out", "            also downloads the embedding model, ~1-2 min)", T_OUT),
    ("out", "", T_OUT),
    ("out", " Your documents live in:", T_OUT),
    ("out", "   /home/you/serviette-demo/docs", T_CMD),
    ("out", " Anything you do there — edit a file, drop in new documents (PDF, DOCX,", T_OUT),
    ("out", " scans, …), delete one — is reflected in the answers within seconds.", T_OUT),
    ("out", "", T_OUT),
    ("out", " A scripted moment to try first:", T_OUT),
    ("out", "   1. Ask in the chat: How much does the Team tier cost?   (→ 129 EUR)", T_OUT),
    ("out", "   2. Open /home/you/serviette-demo/docs/pricing.md and change 129 EUR → 199 EUR", T_OUT),
    ("out", "   3. Ask again — the answer follows the file", T_OUT),
    ("out", "      (watch the “indexed … ago” counter in the header)", T_OUT),
    ("out", "", T_OUT),
    ("out", " Ctrl-C stops everything.", T_OUT),
    ("out", "─" * 72, T_OUT),
    ("out", "", T_OUT),
    ("out", _INFO + "up: indexer started (pid 1986416)", T_OUT),
    ("out", _INFO + "up: showing progress, warnings and errors only — run with --verbose "
            "for the full indexer and server log", T_OUT),
    ("out", _INFO + "up: indexing in progress — the chat/API server starts once the first "
            "documents are ready (0s elapsed)", T_OUT),
    ("out", _INFO + "up: indexing in progress — the chat/API server starts once the first "
            "documents are ready (16s elapsed)", T_OUT),
    ("out", _INFO + "up: indexing in progress — the chat/API server starts once the first "
            "documents are ready (32s elapsed)", T_OUT),
    ("out", _INFO + "up: index is ready — starting the server (pid 2019552)", T_OUT),
    ("out", _INFO + "up: server starting — loading the query embedder; the URL appears once "
            "it answers (1s elapsed)", T_OUT),
    ("out", _INFO + "up: server starting — loading the query embedder; the URL appears once "
            "it answers (11s elapsed)", T_OUT),
    ("out", _INFO, T_OUT),
    ("out", "  Ready — open http://localhost:8989", T_ACCENT),
    ("out", "  (Ctrl-C stops the indexer and the server)", T_OUT),
]

# `demo` keeps the first terminal; the edit happens in a second one. `sed`
# prints nothing, and in quiet mode neither does `up` — the change shows up
# in the chat header ("indexed 3s ago") and in the answer.
SCRIPT_EDIT = [
    ("cmd", "sed -i 's/129 EUR/199 EUR/' serviette-demo/docs/pricing.md"),
]


def _wrap_mono(text: str, fnt: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Wrap one printed line the way a terminal of this width would."""

    if fnt.getlength(text) <= max_w:
        return [text]
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}" if cur else w
        if fnt.getlength(trial) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_terminal(lines, typing, title="serviette — terminal") -> Image.Image:
    img = Image.new("RGB", (T_W, T_H), T_BG)
    d = ImageDraw.Draw(img)
    mono = font("mono", T_FONT)
    mono_b = font("mono_bold", T_FONT)
    # title bar
    d.rounded_rectangle([0, 0, T_W, 40], radius=0, fill=T_BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 22, 14, 32 + i * 22, 26], fill=c)
    d.text((T_W // 2, 20), title, font=font("sans", 14), fill=T_OUT, anchor="mm")

    # Lay out every printed line (wrapped like a terminal would), then show
    # the tail that fits — older lines scroll off the top, as they really do.
    max_w = T_W - 2 * T_PAD
    rows: list[tuple[str, str, tuple]] = []
    for kind, text, color in lines:
        if kind == "cmd":
            rows.append(("cmd", text, T_CMD))
        else:
            for piece in _wrap_mono(text, mono, max_w):
                rows.append(("out", piece, color))
    prompt_rows = 1 if typing is not None else 0
    capacity = (T_H - T_FIRST_Y - 8) // T_LINE_H - prompt_rows
    rows = rows[-capacity:] if capacity > 0 else []

    y = T_FIRST_Y
    for kind, text, color in rows:
        if kind == "cmd":
            d.text((T_PAD, y), "$", font=mono_b, fill=T_PROMPT)
            d.text((T_PAD + 18, y), " " + text, font=mono_b, fill=T_CMD)
        else:
            d.text((T_PAD, y), text, font=mono, fill=color)
        y += T_LINE_H

    if typing is not None:
        d.text((T_PAD, y), "$", font=mono_b, fill=T_PROMPT)
        d.text((T_PAD + 18, y), " " + typing, font=mono_b, fill=T_CMD)
        w = mono_b.getlength(" " + typing)
        d.rectangle([T_PAD + 18 + w + 2, y + 2, T_PAD + 18 + w + 12, y + 20], fill=T_CMD)
    return img


def build_terminal(
    script, preprinted=(), title="serviette — terminal", hold=1500
) -> tuple[list[Image.Image], list[int], list[tuple]]:
    frames: list[Image.Image] = []
    durs: list[int] = []
    printed: list[tuple] = list(preprinted)

    for item in script:
        kind = item[0]
        if kind == "cmd":
            text = item[1]
            cur = ""
            for i, ch in enumerate(text):
                cur += ch
                if i % 3 == 0 or i == len(text) - 1:
                    frames.append(_draw_terminal(printed, cur, title))
                    durs.append(55)
            printed.append(("cmd", text, T_CMD))
            frames.append(_draw_terminal(printed, None, title))
            durs.append(350)
        elif kind == "out":
            printed.append(("out", item[1], item[2]))
            frames.append(_draw_terminal(printed, None, title))
            # The banner is printed at once; the `up:` progress lines arrive
            # over time, so they get a longer beat each.
            durs.append(700 if item[1].startswith(_INFO) or item[1].startswith("  ") else 120)

    frames.append(_draw_terminal(printed, None, title))
    durs.append(hold)  # hold before switching to the UI scene
    return frames, durs, printed


# ---------------------------------------------------------------------------
# Frontend (chat UI) demo
# ---------------------------------------------------------------------------

F_W, F_H = 760, 560
F_BG = (255, 255, 255)
F_BORDER = (230, 230, 233)
F_TEXT = (31, 32, 35)
F_SOFT = (107, 114, 128)
F_ACCENT = (79, 70, 229)
F_USER_BG = (31, 32, 35)
F_ASSIST_BG = (244, 244, 246)
F_SOFTBG = (247, 247, 248)

# What gpt-4o-mini answered on the bundled corpus, before and after the edit
# (recorded with `serviette demo`, local embeddings, k=5).
QUESTION = "How much does the Team tier cost?"
ANSWER_BEFORE = "The Team tier costs 129 EUR/month."
ANSWER_AFTER = "The Team tier costs 199 EUR/month."
# The header's live statistics line, as /api/v1/stats reported them.
STATS_BEFORE = "duckdb · 5 chunks · 5 docs · indexed 2m ago"
STATS_AFTER = "duckdb · 5 chunks · 5 docs · indexed 3s ago"
SOURCES_LABEL = "5 sources"


def _wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_frontend(exchanges, composer_text, typing, indexed_note) -> Image.Image:
    """Render the chat with a list of (question, shown_answer_chars|None,
    show_sources) exchanges; ``typing`` draws the assistant dots after the
    last question instead of an answer."""

    img = Image.new("RGB", (F_W, F_H), F_BG)
    d = ImageDraw.Draw(img)
    sans = font("sans", 16)
    sans_b = font("sans_bold", 16)
    small = font("sans", 13)

    # header
    d.line([0, 53, F_W, 53], fill=F_BORDER)
    d.rounded_rectangle([20, 16, 46, 42], radius=8, fill=F_ACCENT)
    d.text((56, 29), "serviette", font=sans_b, fill=F_TEXT, anchor="lm")
    # right side: live statistics, then the "Documents" and settings buttons
    x = F_W - 20
    d.rounded_rectangle([x - 30, 15, x, 43], radius=8, outline=F_BORDER, fill=F_SOFTBG)
    d.text((x - 15, 29), "⚙", font=font("sans", 14), fill=F_SOFT, anchor="mm")
    x -= 38
    d.rounded_rectangle([x - 86, 15, x, 43], radius=8, outline=F_BORDER, fill=F_SOFTBG)
    d.text((x - 43, 29), "Documents", font=small, fill=F_SOFT, anchor="mm")
    x -= 98
    d.text((x, 29), indexed_note, font=small, fill=F_SOFT, anchor="rm")

    y = 78
    if not exchanges and not composer_text:
        d.text((F_W // 2, 200), "Ask anything about your documents",
               font=font("sans_bold", 22), fill=F_TEXT, anchor="mm")
        d.text((F_W // 2, 232), "Answers are grounded in your indexed knowledge base.",
               font=sans, fill=F_SOFT, anchor="mm")
        d.text((F_W // 2, 256), "Each question is answered on its own, without memory of earlier ones.",
               font=sans, fill=F_SOFT, anchor="mm")

    for i, (question, answer_chars, show_sources) in enumerate(exchanges):
        # user bubble (right)
        lines = _wrap(d, question, sans, 360)
        bw = max(d.textlength(ln, font=sans) for ln in lines) + 32
        bh = len(lines) * 24 + 20
        bx2 = F_W - 24
        d.rounded_rectangle([bx2 - bw, y, bx2, y + bh], radius=18, fill=F_USER_BG)
        for j, ln in enumerate(lines):
            d.text((bx2 - bw + 16, y + 12 + j * 24), ln, font=sans, fill=(255, 255, 255))
        y += bh + 14

        # assistant side
        ax = 24
        d.ellipse([ax, y, ax + 30, y + 30], fill=F_ACCENT)
        d.text((ax + 15, y + 15), "AI", font=font("sans_bold", 11),
               fill=(255, 255, 255), anchor="mm")
        bx = ax + 42
        is_last = i == len(exchanges) - 1
        if is_last and typing:
            d.rounded_rectangle([bx, y, bx + 70, y + 34], radius=16, fill=F_ASSIST_BG)
            for k, on in enumerate(typing):
                col = F_SOFT if on else (200, 203, 209)
                d.ellipse([bx + 16 + k * 16, y + 14, bx + 23 + k * 16, y + 21], fill=col)
            y += 44
        elif answer_chars is not None:
            full = ANSWER_AFTER if i == 1 else ANSWER_BEFORE
            shown = full[:answer_chars]
            lines = _wrap(d, shown, sans, 470)
            bh = len(lines) * 24 + 20
            bw = (max((d.textlength(ln, font=sans) for ln in lines), default=0)) + 32
            d.rounded_rectangle([bx, y, bx + max(bw, 90), y + bh], radius=18, fill=F_ASSIST_BG)
            for j, ln in enumerate(lines):
                d.text((bx + 16, y + 12 + j * 24), ln, font=sans, fill=F_TEXT)
            y += bh + 8
            if show_sources:
                # The UI's collapsed <details> block listing the passages used.
                d.rounded_rectangle([bx, y, bx + 130, y + 28], radius=10,
                                    fill=F_SOFTBG, outline=F_BORDER)
                d.text((bx + 12, y + 14), "▸  " + SOURCES_LABEL, font=small,
                       fill=F_SOFT, anchor="lm")
                y += 38
        y += 10

    # composer
    cy = F_H - 74
    d.rounded_rectangle([24, cy, F_W - 24, cy + 50], radius=22,
                        outline=F_BORDER, width=1, fill=F_BG)
    text = composer_text if composer_text else "Message…"
    color = F_TEXT if composer_text else F_SOFT
    d.text((44, cy + 25), text, font=sans, fill=color, anchor="lm")
    d.ellipse([F_W - 24 - 46, cy + 6, F_W - 24 - 8, cy + 44], fill=F_ACCENT)
    cx, cyy = F_W - 24 - 27, cy + 25
    d.line([cx - 6, cyy + 5, cx + 6, cyy - 6], fill=(255, 255, 255), width=2)
    d.line([cx + 6, cyy - 6, cx + 1, cyy - 6], fill=(255, 255, 255), width=2)
    d.line([cx + 6, cyy - 6, cx + 6, cyy - 1], fill=(255, 255, 255), width=2)
    d.text((F_W // 2, F_H - 12),
           "Enter to send · Shift+Enter for a new line · Each question is answered on its own",
           font=font("sans", 11), fill=F_SOFT, anchor="mm")
    return img


def build_chat(prior, answer, indexed_note) -> tuple[list[Image.Image], list[int]]:
    """One question/answer beat: type the question, dots, reveal ``answer``.

    ``prior`` is a list of completed exchanges rendered above (the history).
    """

    frames: list[Image.Image] = []
    durs: list[int] = []

    def add(frame, dur):
        frames.append(frame)
        durs.append(dur)

    done = [(q, chars, True) for q, chars in prior]
    add(_draw_frontend(done, "", None, indexed_note), 900)
    cur = ""
    for i, ch in enumerate(QUESTION):
        cur += ch
        if i % 2 == 0 or i == len(QUESTION) - 1:
            add(_draw_frontend(done, cur, None, indexed_note), 40)
    add(_draw_frontend(done, QUESTION, None, indexed_note), 300)

    pending = done + [(QUESTION, None, False)]
    for _ in range(2):
        for pat in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            add(_draw_frontend(pending, "", pat, indexed_note), 200)

    for frac in (0.4, 0.8, 1.0):
        shown = done + [(QUESTION, int(len(answer) * frac), False)]
        add(_draw_frontend(shown, "", None, indexed_note), 380)
    add(_draw_frontend(done + [(QUESTION, len(answer), True)], "", None, indexed_note), 2600)
    return frames, durs


# ---------------------------------------------------------------------------
# Combined demo (terminal scene -> chat UI scene, one GIF)
# ---------------------------------------------------------------------------

C_W, C_H = 920, 600


def _onto_canvas(img: Image.Image, bg: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", (C_W, C_H), bg)
    canvas.paste(img, ((C_W - img.width) // 2, (C_H - img.height) // 2))
    return canvas


def render_combined_gif() -> None:
    # Beat 1: `serviette demo` up to "Ready — open". Beat 2: ask about the
    # Team tier -> 129 EUR. Beat 3: sed edits pricing.md in a second terminal.
    # Beat 4: the same question -> 199 EUR, header says "indexed 3s ago".
    t1_frames, t1_durs, _ = build_terminal(SCRIPT_UP)
    c1_frames, c1_durs = build_chat([], ANSWER_BEFORE, STATS_BEFORE)
    t2_frames, t2_durs, _ = build_terminal(
        SCRIPT_EDIT, title="serviette — second terminal", hold=1200
    )
    c2_frames, c2_durs = build_chat(
        [(QUESTION, len(ANSWER_BEFORE))], ANSWER_AFTER, STATS_AFTER
    )

    frames = [_onto_canvas(f, T_BG) for f in t1_frames]
    frames += [_onto_canvas(f, F_BG) for f in c1_frames]
    frames += [_onto_canvas(f, T_BG) for f in t2_frames]
    frames += [_onto_canvas(f, F_BG) for f in c2_frames]
    durs = t1_durs + c1_durs + t2_durs + c2_durs

    save_gif(ASSETS / "demo.gif", frames, durs, colors=96)


if __name__ == "__main__":
    render_combined_gif()
