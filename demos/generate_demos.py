#!/usr/bin/env python3
"""Generate the README demo GIFs (terminal walkthrough + chat UI).

These are *emulated* demos rendered with Pillow — no terminal recorder or
browser needed — so they regenerate deterministically anywhere:

    python demos/generate_demos.py

Output:
    docs/assets/demo.gif   the four `vetosh` commands + output, then the chat UI
                           answering a question (two scenes, one GIF)

Requires Pillow (``pip install pillow``).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

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

SCRIPT = [
    ("cmd", "vetosh quickstart"),
    ("out", "  Step 1: What to generate?   › Universal config", T_OUT),
    ("out", "  …  Wrote universal config to ./config.yaml", T_OUT),
    ("gap", ""),
    ("cmd", "vetosh indexer --config config.yaml"),
    ("out", "  [indexer] watching /data/documents  (fs · only_metadata)", T_OUT),
    ("out", "  [indexer] parsed 128 docs · embedded 1,544 chunks → pgvector", T_ACCENT),
    ("gap", ""),
    ("cmd", "vetosh server --config config.yaml"),
    ("out", "  INFO  Uvicorn running on http://0.0.0.0:8000", T_OUT),
    ("gap", ""),
    ("cmd", "vetosh frontend --config config.yaml"),
    ("out", "  INFO  Uvicorn running on http://0.0.0.0:3000  →  open in browser", T_ACCENT),
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
    d.text((T_W // 2, 20), "vetosh — terminal", font=font("sans", 14), fill=T_OUT, anchor="mm")

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


def build_terminal() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durs: list[int] = []
    printed: list[tuple] = []

    for item in SCRIPT:
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
    return frames, durs


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

QUESTION = "How does vetosh keep the vector DB in sync?"
ANSWER = (
    "vetosh runs on Pathway's live data framework. When a file changes, the "
    "indexer re-embeds only the diff and updates the vector DB in real time — "
    "additions, edits and deletions all propagate automatically."
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


def _draw_frontend(composer_text, show_user, typing, answer_chars, show_sources) -> Image.Image:
    img = Image.new("RGB", (F_W, F_H), F_BG)
    d = ImageDraw.Draw(img)
    sans = font("sans", 16)
    sans_b = font("sans_bold", 16)
    small = font("sans", 13)

    # header
    d.line([0, 53, F_W, 53], fill=F_BORDER)
    d.rounded_rectangle([20, 16, 46, 42], radius=8, fill=F_ACCENT)
    d.text((56, 29), "vetosh", font=sans_b, fill=F_TEXT, anchor="lm")
    d.text((F_W - 20, 29), "API · http://localhost:8000", font=small, fill=F_SOFT, anchor="rm")

    y = 86
    if not show_user:
        d.text((F_W // 2, 210), "Ask anything about your documents",
               font=font("sans_bold", 22), fill=F_TEXT, anchor="mm")
        d.text((F_W // 2, 242), "Answers are grounded in your indexed knowledge base.",
               font=sans, fill=F_SOFT, anchor="mm")
    else:
        # user bubble (right)
        lines = _wrap(d, QUESTION, sans, 360)
        bw = max(d.textlength(ln, font=sans) for ln in lines) + 32
        bh = len(lines) * 24 + 20
        bx2 = F_W - 24
        d.rounded_rectangle([bx2 - bw, y, bx2, y + bh], radius=18, fill=F_USER_BG)
        for i, ln in enumerate(lines):
            d.text((bx2 - bw + 16, y + 12 + i * 24), ln, font=sans, fill=(255, 255, 255))
        d.ellipse([F_W - 24 + 8, y, F_W - 24 + 8, y], fill=F_USER_BG)
        y += bh + 18

        # assistant area
        ax = 24
        d.ellipse([ax, y, ax + 30, y + 30], fill=F_ACCENT)
        d.text((ax + 15, y + 15), "AI", font=font("sans_bold", 11), fill=(255, 255, 255), anchor="mm")
        bx = ax + 42
        if typing:
            d.rounded_rectangle([bx, y, bx + 70, y + 34], radius=16, fill=F_ASSIST_BG)
            for i, on in enumerate(typing):
                col = F_SOFT if on else (200, 203, 209)
                d.ellipse([bx + 16 + i * 16, y + 14, bx + 23 + i * 16, y + 21], fill=col)
        else:
            shown = ANSWER[:answer_chars]
            lines = _wrap(d, shown, sans, 470)
            bh = len(lines) * 24 + 20
            bw = (max((d.textlength(ln, font=sans) for ln in lines), default=0)) + 32
            d.rounded_rectangle([bx, y, bx + max(bw, 90), y + bh], radius=18, fill=F_ASSIST_BG)
            for i, ln in enumerate(lines):
                d.text((bx + 16, y + 12 + i * 24), ln, font=sans, fill=F_TEXT)
            y += bh + 10
            if show_sources:
                d.rounded_rectangle([bx, y, bx + 150, y + 30], radius=10,
                                    fill=F_SOFTBG, outline=F_BORDER)
                d.text((bx + 12, y + 15), "▸  2 sources", font=small, fill=F_SOFT, anchor="lm")

    # composer
    cy = F_H - 74
    d.rounded_rectangle([24, cy, F_W - 24, cy + 50], radius=22, outline=F_BORDER, width=1, fill=F_BG)
    text = composer_text if composer_text else "Message…"
    color = F_TEXT if composer_text else F_SOFT
    d.text((44, cy + 25), text, font=sans, fill=color, anchor="lm")
    d.ellipse([F_W - 24 - 46, cy + 6, F_W - 24 - 8, cy + 44], fill=F_ACCENT)
    # send arrow
    cx, cyy = F_W - 24 - 27, cy + 25
    d.line([cx - 6, cyy + 5, cx + 6, cyy - 6], fill=(255, 255, 255), width=2)
    d.line([cx + 6, cyy - 6, cx + 1, cyy - 6], fill=(255, 255, 255), width=2)
    d.line([cx + 6, cyy - 6, cx + 6, cyy - 1], fill=(255, 255, 255), width=2)
    return img


def build_frontend() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durs: list[int] = []

    def add(frame: Image.Image, dur: int) -> None:
        frames.append(frame)
        durs.append(dur)

    # 1) empty + type the question
    add(_draw_frontend("", False, None, 0, False), 900)
    cur = ""
    for i, ch in enumerate(QUESTION):
        cur += ch
        if i % 2 == 0 or i == len(QUESTION) - 1:
            add(_draw_frontend(cur, False, None, 0, False), 40)
    add(_draw_frontend(QUESTION, False, None, 0, False), 350)

    # 2) send -> user bubble + typing dots
    for _ in range(2):
        for pat in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            add(_draw_frontend("", True, pat, 0, False), 220)

    # 3) reveal answer progressively
    for frac in (0.35, 0.7, 1.0):
        add(_draw_frontend("", True, None, int(len(ANSWER) * frac), False), 380)

    # 4) show sources + hold
    add(_draw_frontend("", True, None, len(ANSWER), True), 3000)

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
    t_frames, t_durs = build_terminal()
    f_frames, f_durs = build_frontend()

    frames = [_onto_canvas(f, T_BG) for f in t_frames]
    frames += [_onto_canvas(f, F_BG) for f in f_frames]
    durs = t_durs + f_durs

    save_gif(ASSETS / "demo.gif", frames, durs, colors=96)


if __name__ == "__main__":
    render_combined_gif()
