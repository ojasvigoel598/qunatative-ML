#!/usr/bin/env python3
"""
Build a SELF-CONTAINED demo player page: demo/output/video_player.html

Why: the Freebuff preview server (and many static hosts) serve only the single
registered HTML file — sibling .mp4 files come back 404, so <video> controls
are disabled and nothing plays. Embedding each video as a base64 data URI makes
the page work from ANY static host, or even from file://.

Each video is re-encoded to 1280x720 (CRF 26) before embedding to keep the
page size sane.

Usage:
    python demo/build_embedded_player.py
"""

import base64
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TMP = OUT / ".embed_tmp"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CARDS = [
    {
        "file": "project_explainer.mp4",
        "title": "Whole-project explainer — 4:39 narrated tour (start here)",
        "tags": ["AI voice narration", "architecture diagram", "real code snippets",
                 "data sources", "how the simulation proves the methodology"],
        "note": ("The full picture, not just the simulation: the architecture "
                 "diagram, actual code from all three model layers, the data "
                 "ingestion source, backtest plots, real-data season validation, "
                 "the $1M Monte-Carlo simulation, the confidence-aware layer, and "
                 "why the honest losses prove the methodology works."),
    },
    {
        "file": "demo_narrated.mp4",
        "title": "Narrated demo — 4:40 simulation-focused walkthrough",
        "tags": ["AI voice narration", "edge-tts", "flat + dynamic simulations", "Serie A replay"],
        "note": ("A polished product-style demo with natural AI narration: the $1M "
                 "flat simulation, the confidence-aware dynamic layer, a real Serie "
                 "A replay, and the honest bottom line. Narration only states what "
                 "the repo actually does."),
    },
    {
        "file": "simulation_live_flat.mp4",
        "title": "$1,000,000 Simulation — live replay",
        "tags": ["synthetic world", "flat $10K/bet", "trial 0: final $1,569,700 (+57%)"],
        "note": ("One Monte-Carlo draw of 1,200 matches (~3.3 yrs). "
                 "639 bets, 235 W / 404 L. Watch the ML thinking panel on the right."),
    },
    {
        "file": "simulation_live_dynamic.mp4",
        "title": "Dynamic Thinking Layer — $1M live (confidence-aware decisions)",
        "tags": ["confidence-aware", "market-split signal",
                 "trial 0: final $1,170,489 (+17.0%)"],
        "note": ("The ML THINKING panel shows the live decision trace: "
                 "model-vs-market weight, confidence, dispersion, rest days. "
                 "The layer refits its base model when confidence decays and "
                 "scales stakes with how sure it is."),
    },
    {
        "file": "serie_a_live.mp4",
        "title": "Real Serie A 2025/26 — point-in-time replay",
        "tags": ["real matches", "real B365 odds", "adaptive model",
                 "final $377,276 (−62.3%)"],
        "note": ("The honest real-world result: 95 bets at real prices, "
                 "32 W / 63 L. Real odds are hard to beat — that is the "
                 "finding, not a bug."),
    },
]


def data_uri(path: Path) -> str:
    """Re-encode to 1280x720 and return a base64 data URI."""
    TMP.mkdir(exist_ok=True)
    small = TMP / path.name
    subprocess.run(
        [FFMPEG, "-y", "-i", str(path),
         "-vf", "scale=1280:720",
         "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-an", str(small)],
        check=True, capture_output=True)
    b64 = base64.b64encode(small.read_bytes()).decode()
    return f"data:video/mp4;base64,{b64}"


def main() -> int:
    missing = [c["file"] for c in CARDS if not (OUT / c["file"]).exists()]
    if missing:
        print("MISSING VIDEO FILES:", missing)
        print("Re-render them first (demo/make_*_video.py) before building.")
        return 1

    cards_html = []
    for c in CARDS:
        print(f"  embedding {c['file']} ...")
        uri = data_uri(OUT / c["file"])
        tags = "".join(f'<span class="tag">{t}</span>' for t in c["tags"])
        cards_html.append(f"""
  <div class="card">
    <h2>{c['title']}</h2>
    <div class="meta">{tags}</div>
    <video controls preload="auto" src="{uri}"></video>
    <div class="warn">{c['note']}</div>
  </div>""")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ML Betting Demo Videos</title>
<style>
  body {{ background:#0E1420; color:#E8EDF2; font-family:'Segoe UI',system-ui,sans-serif;
         margin:0; padding:32px 24px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  p.sub {{ color:#9FB0C3; margin:0 0 24px; font-size:14px; }}
  .card {{ background:#1B2536; border-radius:12px; padding:20px; margin-bottom:28px;
          border:1px solid #2A3547; }}
  .card h2 {{ margin:0 0 4px; font-size:18px; }}
  .card .meta {{ color:#9FB0C3; font-size:13px; margin-bottom:14px; }}
  video {{ width:100%; max-width:960px; border-radius:8px; background:#000; display:block; }}
  .tag {{ display:inline-block; background:#2E86AB; color:#fff; font-size:12px;
         border-radius:999px; padding:2px 10px; margin-right:6px; }}
  .warn {{ color:#F18F01; font-size:13px; margin-top:10px; }}
</style>
</head>
<body>
  <h1>🎬 ML Betting — Demo Videos</h1>
  <p class="sub">Rendered from the real run outputs (matplotlib + ffmpeg). No AI-generated
  footage; every number is the actual simulation result. Self-contained page — videos are
  embedded, so it plays from any host.</p>
{''.join(cards_html)}
</body>
</html>"""

    out = OUT / "video_player.html"
    out.write_text(page, encoding="utf-8")
    print(f"\n[OK] {out}  ({out.stat().st_size / 1e6:.1f} MB — self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
