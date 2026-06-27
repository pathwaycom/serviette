# Demo assets

The README GIFs are **emulated** (rendered with Pillow — no terminal recorder or
browser needed), so they regenerate deterministically anywhere:

```bash
pip install pillow
python demos/generate_demos.py
```

This writes `docs/assets/demo.gif` — a single animation that plays the
`quickstart → indexer → server → frontend` terminal walkthrough and then the
chat UI answering a question (drawn to match the real `vetosh frontend` page).

Edit the `SCRIPT` / `QUESTION` / `ANSWER` constants in `generate_demos.py` to
change the demo content. The two scenes are produced by `build_terminal()` and
`build_frontend()` and stitched onto a shared canvas by `render_combined_gif()`.
