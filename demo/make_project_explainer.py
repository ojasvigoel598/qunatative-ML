#!/usr/bin/env python3
"""
Build a narrated WHOLE-PROJECT explainer video (not just the simulation).

A self-contained MP4 that walks a stranger through the entire project using
REAL artifacts from the repository: the architecture diagram, actual code
snippets from the three model layers, the data-ingestion source, the backtest
plots, the season-by-season real-data validation, the $1M Monte-Carlo
simulation, the confidence-aware dynamic layer, and the honest bottom line.

Narration is AI-generated with Microsoft Edge neural voices (edge-tts, free,
no API key). Every claim in the narration is true of the committed repo; every
image is a real file produced by running the project.

Run:
    python demo/make_project_explainer.py

Output: demo/output/project_explainer.mp4  (+ intermediates under .embed_tmp/)
Dependencies: edge-tts (pip), matplotlib, imageio-ffmpeg.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TMP = OUT / ".embed_tmp"
TMP.mkdir(parents=True, exist_ok=True)

VOICE = "en-US-BrianNeural"
RATE = "-4%"
PITCH = "+0Hz"

IMG = {
    "arch": OUT / ".." / ".." / "assets" / "architecture.png",
    "backtest_analysis": ROOT.parent / "backtests" / "results" / "backtest_analysis.png",
    "backtest_summary": ROOT.parent / "backtests" / "results" / "backtest_summary.png",
    "season_within": ROOT.parent / "backtests" / "results" / "season_backtest_within.png",
    "season_cross": ROOT.parent / "backtests" / "results" / "season_backtest_cross.png",
    "stress": ROOT.parent / "backtests" / "results" / "staking_stress_test.png",
    "sim_1m": OUT / "simulation_1m.png",
    "sim_dynamic": OUT / "simulation_1m_dynamic.png",
    "real_multi": OUT / "real_simulation_multi.png",
}

# (label, media, narration)  media: None=title card, ("image", key),
# ("code", path, start_line, end_line), ("clip", filename)
SECTIONS = [
    ("intro", None, (
        "This is a quantitative machine-learning system for sports betting. "
        "Three layers: a Poisson and Elo hybrid that prices football matches, "
        "a calibrated gradient-boosting layer that hunts for value, and a "
        "Q-learning agent that decides how much to stake. Every number and "
        "every image in this video comes from actually running the code in "
        "this repository."
    )),
    ("arch", ("image", "arch"), (
        "Here is the whole pipeline. Raw match data with real bookmaker odds "
        "flows into the Poisson-Elo engine, which produces win-draw-loss "
        "probabilities. Those become features for the machine-learning layer, "
        "and a staking agent converts edges into bet sizes. The pipeline is "
        "validated with a chronological train, validation, test split, so the "
        "test set is never tuned on."
    )),
    ("data", ("code", "scripts/01_data_ingestion.py", 22, 60), (
        "The system is built on real football data. Results and odds come "
        "from football-data dot co dot uk across five European leagues, with "
        "Pinnacle closing odds preserved so closing line value can be "
        "measured. An openfootball World Cup dataset adds international "
        "matches. When downloads are unavailable, a seeded generator produces "
        "a fully reproducible synthetic dataset instead."
    )),
    ("layer1", ("code", "models/poisson_elo_model.py", 62, 100), (
        "Layer one is the Poisson-Elo hybrid. Elo ratings give each team a "
        "dynamic strength, updated sequentially match by match, with no "
        "look-ahead. Two Poisson regressions model expected home and away "
        "goals, the score grid is summed into win, draw, loss probabilities, "
        "and those convert to fair odds. Edge is probability times the "
        "bookmaker odds, minus one."
    )),
    ("layer2", ("code", "models/ml_layer.py", 36, 80), (
        "Layer two is a calibrated gradient-boosting model. It trains on the "
        "same Elo features plus shifted rolling goal averages, so a match's "
        "own goals never leak into its features. The classifier is "
        "calibrated, because edge calculation is extremely sensitive to "
        "overconfident probabilities."
    )),
    ("layer3", ("code", "models/rl_staking_agent.py", 39, 75), (
        "Layer three is the staking agent. A Q-learning agent learns Kelly "
        "multipliers from validation bets, and quarter-Kelly staking keeps "
        "the bankroll alive through losing streaks."
    )),
    ("backtest", ("image", "backtest_analysis"), (
        "The canonical backtest is fully reproducible from a clean checkout. "
        "It prices two hundred and forty held-out test matches. The model "
        "beats the majority baseline at fifty-four point six percent "
        "accuracy, with a log loss of point nine eight. But here is the "
        "honest finding: both configurations lose money against a bookmaker "
        "with a real positive margin, at minus fourteen and minus thirteen "
        "percent return on investment. The estimated edges overstate realised "
        "returns. That is the winner's curse."
    )),
    ("season", ("image", "season_within"), (
        "On real data, the methodology shows its strength. A season-by-season "
        "backtest on La Liga reaches fifty-three to fifty-five percent "
        "accuracy on unseen seasons, against a forty-four to forty-nine "
        "percent majority baseline, with stable calibration. This is "
        "out-of-sample, time-aware validation with online features."
    )),
    ("sim", ("image", "sim_1m"), (
        "The one million dollar Monte-Carlo simulation is the honest "
        "summary. Across twenty-five trials, the median path loses about six "
        "percent, with only a thirty-two percent probability of profit. The "
        "point is not that it wins. The point is that the methodology "
        "measures what would actually happen."
    )),
    ("dynamic", ("image", "sim_dynamic"), (
        "The confidence-aware dynamic thinking layer improves the risk "
        "profile. It fuses the model probability with the sharp market line, "
        "watches the public-versus-sharp split, refits when its confidence "
        "decays, and scales stakes with certainty. It loses the least of "
        "every policy, at minus one point one percent, with the tightest "
        "downside range and a forty-four percent probability of profit."
    )),
    ("stress", ("image", "stress"), (
        "So how does the simulation prove the theory? It stress-tests staking "
        "under real margins, reproduces the same honest result across "
        "hundreds of random seeds in a walk-forward audit, and flags leakage "
        "at every step: point-in-time features, chronological splits, and "
        "independent closing odds for line value. A system that reports its "
        "losses honestly is far more credible than one that claims to beat "
        "the bookmaker."
    )),
    ("outro", None, (
        "The project is a rigorous, leak-free testing methodology for "
        "quantitative sports betting. Clone the repository, install the "
        "requirements, and run the pipelines yourself. The commands are in "
        "the readme, and the narrated simulation demo shows the model "
        "thinking and betting in real time."
    )),
]


def _prep_static(src: Path) -> Path:
    """Re-save any image as an exact 1600x900 PNG (aspect-preserved, black
    bars) so ffmpeg decodes one small frame instead of a huge source PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    img = mpimg.imread(str(src))
    h, w = img.shape[:2]
    tw, th = 1600, 900
    fig = plt.figure(figsize=(tw / 100, th / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, tw)
    ax.set_ylim(0, th)
    scale = min(tw / w, th / h)
    nw, nh = w * scale, h * scale
    x0, y0 = (tw - nw) / 2, (th - nh) / 2
    ax.imshow(img, extent=[x0, x0 + nw, y0, y0 + nh], aspect="auto")
    dst = TMP / f"prepped_{src.name}"
    fig.savefig(dst, dpi=100, facecolor="black")
    plt.close(fig)
    return dst


def _resolve_media(media):
    if media is None:
        return None
    kind = media[0]
    if kind == "image":
        p = IMG[media[1]].resolve()
        if not p.exists():
            raise SystemExit(f"MISSING IMAGE: {p}")
        return _prep_static(p)
    if kind == "clip":
        p = (OUT / media[1]).resolve()
        if not p.exists():
            raise SystemExit(f"MISSING CLIP: {p}")
        return p
    if kind == "code":
        path = (ROOT.parent / media[1])
        _, _, start, end = media
        lines = path.read_text(encoding="utf-8").splitlines()
        snippet = lines[start - 1:end]
        return _render_code(path.name, snippet, start)
    raise SystemExit(f"unknown media: {media}")


def _render_code(filename: str, lines: list[str], start_line: int) -> Path:
    """Render real code lines to a PNG with a file-header bar (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrapped = []
    for ln in lines:
        while len(ln) > 100:
            cut = ln[:100]
            wrapped.append(cut)
            ln = ln[100:]
        wrapped.append(ln)
    n = len(wrapped) + 2  # header + blank

    fig = plt.figure(figsize=(16, 9), facecolor="#0D1117")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, n + 1)

    # header bar
    ax.add_patch(plt.Rectangle((0, n - 1), 100, 2, facecolor="#161B22",
                               edgecolor="none"))
    ax.text(2, n - 0.05, filename, color="#58A6FF", fontsize=13,
            fontweight="bold", family="DejaVu Sans Mono", va="top")
    ax.text(98, n - 0.05, f"lines {start_line}-{start_line + len(lines) - 1}",
            color="#8B949E", fontsize=10, family="DejaVu Sans Mono",
            ha="right", va="top")

    for i, ln in enumerate(wrapped):
        y = n - 3 - i
        if y < 0.5:
            break
        color = "#8B949E" if ln.strip().startswith("#") else "#E6EDF3"
        safe = ln.replace("\\", "\\\\").replace("$", "\\$")
        ax.text(2, y, safe, color=color, fontsize=12,
                family="DejaVu Sans Mono", va="top")

    p = TMP / f"code_{filename.replace('/', '_').replace('.', '_')}.png"
    fig.savefig(p, dpi=100)  # figsize 16x9 @ dpi 100 -> exactly 1600x900
    plt.close(fig)
    return p


def card_path(label: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "intro": ("QUANTITATIVE SPORTS-BETTING ML SYSTEM",
                  ["a narrated tour of the whole project",
                   "data  ->  Poisson + Elo  ->  gradient boosting",
                   "->  Q-learning staking  ->  dynamic thinking",
                   "->  leak-free walk-forward validation",
                   "every image is a real file from this repository"]),
        "outro": ("WHY THIS PROJECT MATTERS",
                  ["a rigorous, leak-free testing methodology",
                   "real odds · calibrated models · audited walk-forward",
                   "it reports its losses honestly",
                   "clone it, pip install, and run it yourself"]),
    }
    title, lines = titles[label]
    fig = plt.figure(figsize=(16, 9), facecolor="#101623")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.78, title, color="white", fontsize=30, fontweight="bold",
            ha="center", va="center")
    ax.plot([0.30, 0.70], [0.66, 0.66], color="#06D6A0", linewidth=3,
            transform=ax.transAxes)
    for i, ln in enumerate(lines):
        ax.text(0.5, 0.56 - i * 0.08, ln, color="#DCE4EE", fontsize=15,
                ha="center", va="center", family="DejaVu Sans Mono",
                transform=ax.transAxes)
    p = TMP / f"card_{label}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


async def synth(label: str, text: str) -> Path:
    import edge_tts
    out = TMP / f"explainer_narr_{label}.mp3"
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save(str(out))
    return out


def probe_duration(path: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(path), "-f", "null", "-"],
                         capture_output=True, text=True)
    for line in out.stderr.splitlines():
        if "Duration:" in line:
            tok = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = tok.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def build_video(card_seconds: float = 8.0):
    # 1) narration + resolve media for each section
    segments = []
    for label, media, text in SECTIONS:
        audio = asyncio.run(synth(label, text))
        dur = probe_duration(audio)
        segments.append((label, _resolve_media(media), audio, dur))
    print("Narration segments (label, media, seconds):")
    for label, media, audio, dur in segments:
        name = str(media.name) if media else "CARD"
        print(f"  {label:<9} {name:<40} {dur:6.1f}s")

    # 2) video timeline: static media hold long enough for narration,
    #    clips play at real speed.
    timeline = []
    t = 0.0
    for label, media, audio, dur in segments:
        if media is None or media.suffix != ".mp4":
            seg_dur = max(card_seconds, dur + 1.5)
        else:
            seg_dur = probe_duration(media)
        timeline.append((label, t, seg_dur))
        t += seg_dur
    print("\nVideo timeline (label, start, duration):")
    for label, t0, d in timeline:
        print(f"  {label:<9} {t0:7.1f}s  {d:6.1f}s")

    # 3) inputs: static media as looped inputs (-t), clips as mp4
    inputs = []
    n_video = 0
    for (label, media, audio, dur), (_, t0, seg_dur) in zip(segments, timeline):
        if media is None:
            card = card_path(label)
            inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-i", str(card)]
        elif media.suffix == ".mp4":
            inputs += ["-i", str(media)]
        else:
            inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-i", str(media)]
        n_video += 1

    # 4) filtergraph: static media is already exactly 1600x900, clips get
    #    scaled/padded.  setsar=1 must come AFTER scale/pad so every chain
    #    ends with SAR 1:1 (concat requires matching SAR on all inputs).
    filter_parts = []
    for i in range(n_video):
        media = segments[i][1]
        is_clip = media is not None and media.suffix == ".mp4"
        if is_clip:
            chain = (f"[{i}:v]fps=24,"
                     f"scale=1600:900:force_original_aspect_ratio=decrease,"
                     f"pad=1600:900:(ow-iw)/2:(oh-ih)/2,"
                     f"setsar=1,format=yuv420p[v{i}]")
        else:
            chain = (f"[{i}:v]fps=24,"
                     f"scale=1600:900:force_original_aspect_ratio=decrease,"
                     f"pad=1600:900:(ow-iw)/2:(oh-ih)/2,"
                     f"setsar=1,format=yuv420p[v{i}]")
        filter_parts.append(chain)
    concat_in = "".join(f"[v{i}]" for i in range(n_video))
    filter_parts.append(f"{concat_in}concat=n={n_video}:v=1:a=0[outv]")

    # 5) narration: each "-i path" pair is one input file, audio k is at
    #    input-file index n_video + k; delay to its segment start, then mix.
    audio_inputs = []
    for k, (label, media, audio, dur) in enumerate(segments):
        audio_inputs += ["-i", str(audio)]
        stream_idx = n_video + k
        t0 = timeline[k][1]
        delay_ms = int(t0 * 1000)
        filter_parts.append(
            f"[{stream_idx}:a]aresample=24000,adelay={delay_ms}|{delay_ms},"
            f"apad[a{k}]")

    mix_in = "".join(f"[a{k}]" for k in range(len(segments)))
    filter_parts.append(
        f"{mix_in}amix=inputs={len(segments)}:normalize=0,"
        f"atrim=0:{t:.2f},asetpts=PTS-STARTPTS[outa]")
    filter_complex = ";".join(filter_parts)

    out_path = OUT / "project_explainer.mp4"
    cmd = [FFMPEG, "-y"] + inputs + audio_inputs
    cmd += ["-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            str(out_path)]
    print("\nRunning ffmpeg composition (can take a few minutes)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-3000:])
        raise SystemExit(f"ffmpeg failed: {res.returncode}")
    print(f"\n[OK] Project explainer saved: {out_path}")
    print(f"     duration: {probe_duration(out_path):.1f}s")
    return out_path


if __name__ == "__main__":
    build_video()
