#!/usr/bin/env python3
"""Generate the README demo GIFs (terminal walkthrough + chat UI).

These are *emulated* demos rendered with Pillow — no terminal recorder or
browser needed — so they regenerate deterministically anywhere:

    python docs/generate_demos.py

Output:
    docs/assets/demo.gif   the four `serviette` commands + output, then the chat UI
                           answering a question (two scenes, one GIF)

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
T_LINE_H = 30

SCRIPT_UP = [
    ("cmd", "serviette wizard"),
    ("out", "  ? Where are your documents?   › ./docs", T_OUT),
    ("out", "  ? Vector database             › DuckDB (embedded, zero setup)", T_OUT),
    ("out", "  ? Embeddings                  › local · no API key needed", T_OUT),
    ("out", "  ✓ Wrote config.yaml", T_OUT),
    ("gap", ""),
    ("cmd", "serviette up --config config.yaml"),
    ("out", "  [indexer] watching ./docs  (fs · streaming)", T_OUT),
    ("out", "  [indexer] parsed 128 docs · embedded 1,544 chunks → duckdb", T_ACCENT),
    ("out", "  [server]  chat UI + API on http://localhost:8989", T_ACCENT),
]

SCRIPT_EDIT = [
    ("gap", ""),
    ("out", "  # change the price — the index follows in seconds:", T_OUT),
    ("cmd", "sed -i 's/129 EUR/199 EUR/' docs/pricing.md"),
    ("out", "  [indexer] docs/pricing.md changed · re-embedded 3 chunks", T_ACCENT),
]


def _draw_terminal(lines, typing) -> Image.Image:
    img = Image.new("RGB", (T_W, T_H), T_BG)
    d = ImageDraw.Draw(img)
    mono = font("mono", 19)
    mono_b = font("mono_bold", 19)
    # title bar
    d.rounded_rectangle([0, 0, T_W, 40], radius=0, fill=T_BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 22, 14, 32 + i * 22, 26], fill=c)
    d.text((T_W // 2, 20), "serviette — terminal", font=font("sans", 14), fill=T_OUT, anchor="mm")

    y = 56
    for kind, text, color in lines:
        if kind == "cmd":
            d.text((T_PAD, y), "$", font=mono_b, fill=T_PROMPT)
            d.text((T_PAD + 18, y), " " + text, font=mono_b, fill=T_CMD)
        elif kind == "out":
            d.text((T_PAD, y), text, font=mono, fill=color)
        y += T_LINE_H

    if typing is not None:
        d.text((T_PAD, y), "$", font=mono_b, fill=T_PROMPT)
        d.text((T_PAD + 18, y), " " + typing, font=mono_b, fill=T_CMD)
        w = mono_b.getlength(" " + typing)
        d.rectangle([T_PAD + 18 + w + 2, y + 2, T_PAD + 18 + w + 12, y + 22], fill=T_CMD)
    return img


def build_terminal(script, preprinted=()) -> tuple[list[Image.Image], list[int], list[tuple]]:
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
                    frames.append(_draw_terminal(printed, cur))
                    durs.append(55)
            printed.append(("cmd", text, T_CMD))
            frames.append(_draw_terminal(printed, None))
            durs.append(350)
        elif kind == "out":
            printed.append(("out", item[1], item[2]))
            frames.append(_draw_terminal(printed, None))
            durs.append(550)
        elif kind == "gap":
            printed.append(("out", "", T_OUT))
            frames.append(_draw_terminal(printed, None))
            durs.append(150)

    frames.append(_draw_terminal(printed, None))
    durs.append(1500)  # hold before switching to the UI scene
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

QUESTION = "How much does the Team tier cost?"
ANSWER_BEFORE = (
    "The Team tier costs 129 EUR per seat per month, billed annually. "
    "It includes fleet monitoring for up to 50 machines."
)
ANSWER_AFTER = (
    "The Team tier costs 199 EUR per seat per month, billed annually. "
    "It includes fleet monitoring for up to 50 machines."
)


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
    d.text((F_W - 20, 29), f"API · http://localhost:8989 · {indexed_note}",
           font=small, fill=F_SOFT, anchor="rm")

    y = 78
    if not exchanges and not composer_text:
        d.text((F_W // 2, 210), "Ask anything about your documents",
               font=font("sans_bold", 22), fill=F_TEXT, anchor="mm")
        d.text((F_W // 2, 242), "Answers are grounded in your indexed knowledge base.",
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
                d.rounded_rectangle([bx, y, bx + 190, y + 28], radius=10,
                                    fill=F_SOFTBG, outline=F_BORDER)
                d.text((bx + 12, y + 14), "▸  docs/pricing.md", font=small,
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
    # Beat 1: wizard + up. Beat 2: ask about the Team tier -> 129 EUR.
    # Beat 3: sed edits pricing.md. Beat 4: same question -> 199 EUR.
    t1_frames, t1_durs, printed = build_terminal(SCRIPT_UP)
    c1_frames, c1_durs = build_chat([], ANSWER_BEFORE, "indexed 41 min ago")
    t2_frames, t2_durs, _ = build_terminal(SCRIPT_EDIT, preprinted=printed)
    c2_frames, c2_durs = build_chat(
        [(QUESTION, len(ANSWER_BEFORE))], ANSWER_AFTER, "indexed 2 s ago"
    )

    frames = [_onto_canvas(f, T_BG) for f in t1_frames]
    frames += [_onto_canvas(f, F_BG) for f in c1_frames]
    frames += [_onto_canvas(f, T_BG) for f in t2_frames]
    frames += [_onto_canvas(f, F_BG) for f in c2_frames]
    durs = t1_durs + c1_durs + t2_durs + c2_durs

    save_gif(ASSETS / "demo.gif", frames, durs, colors=96)


if __name__ == "__main__":
    render_combined_gif()
