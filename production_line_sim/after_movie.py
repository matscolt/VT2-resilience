#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""after_movie.py

Opdateret efter dine ønsker:

1) Produktions-ikon er fjernet.
   - Vi bruger carrier.png alle steder (queue, transport og i produktion).
   - Loading bar over unit i produktion bibeholdes.

2) Farver/font-størrelser er konsekvent implementeret:
   - Samme fontfamilie loader (Calibri hvis muligt), men forskellige størrelser og farver
     for unit-id og disruption.
   - Disruption farve/konstant er kun defineret ét sted.

3) Transport er ændret til input/output pr station:
   - Hver station har input_pos (slutpunkt for incoming) og output_pos (startpunkt for outgoing).
   - Transport interpolerer fra from_station.output_pos -> to_station.input_pos.

4) OPTION A (top-left koordinater) bibeholdes:
   - Du indtaster top-left for queue slots + process subbox.
   - Scriptet konverterer til center via SUBBOX_W/H.

5) Din måde at beregne koordinater på bibeholdes:
   - x1_1, x_space, y_space osv. er med og kan bruges til at definere positioner.

6) clear_folder() bibeholdes og KØRES før rendering.

Folderstruktur:
production_line_sim/
  after_movie.py
  data/Layouts/
    1_LAYOUT.png
    carrier.png
  output/
    <run_folder>/
      station_schedule.csv
      transport_schedule.csv
      movie/
        after_movie.mp4
        frames/
          1.png, 2.png, ...

Kør:
  python after_movie.py
  python after_movie.py render --run <run_folder>

MP4 backend (Windows, hvis nødvendigt):
  pip install imageio[ffmpeg]
"""

from __future__ import annotations
import time
import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Paths
# ----------------------------
ROOTDIR = Path(__file__).resolve().parent
LAYOUTDIR = ROOTDIR / "data" / "Layouts"
OUTPUTDIR = ROOTDIR / "output"

BACKGROUND_PNG = LAYOUTDIR / "1_LAYOUT.png"
CARRIER_PNG = LAYOUTDIR / "carrier.png"

STATION_SCHEDULE_CSV = "station_schedule.csv"
TRANSPORT_SCHEDULE_CSV = "transport_schedule.csv"


# ----------------------------
# Video / rendering settings
# ----------------------------
FPS = 30
SIM_SECONDS_PER_FRAME = 0.5

CARRIER_SIZE_PX = 63

# Unit-ID style (inside carrier)
UNIT_FONT_SIZE = 14
UNIT_TEXT_COLOR = (0, 0, 0, 255)

# Disruption style (next to station)
DISRUPTION_FONT_SIZE = 24
DISRUPTION_TEXT_COLOR = (255, 80, 80, 255)
DISRUPTION_PREFIX = "DISR"

# Optional time label
DRAW_TIME_LABEL = True
TIME_LABEL_POS = (20, 20)
TIME_LABEL_COLOR = (255, 255, 255, 255)
TIME_LABEL_FONT_SIZE = 16

# Loading bar shown above the unit in production
BAR_W = 50
BAR_H = 8
BAR_GAP = 6
BAR_BG_COLOR = (255, 255, 255, 255)
BAR_BORDER_COLOR = (255, 255, 255, 255)
BAR_FILL_COLOR = (0, 200, 0, 255)


# ----------------------------
# OPTION A: TOP-LEFT -> CENTER conversion
# ----------------------------
SUBBOX_W = 71
SUBBOX_H = 83


def tl_to_center(xy_tl: Tuple[int, int], w: int = SUBBOX_W, h: int = SUBBOX_H) -> Tuple[float, float]:
    """Convert a top-left point of a rectangle to its center point."""
    x, y = xy_tl
    return (x + w / 2.0, y + h / 2.0)


# ----------------------------
# Station geometry
# - queue_slots_tl / process_pos_tl are TOP-LEFT corners of subboxes
# - input_pos / output_pos are transport anchor points (pixel coords)
# ----------------------------

@dataclass(frozen=True)
class StationGeom:
    station_name_in_csv: str
    queue_slots_tl: List[Tuple[int, int]]
    process_pos_tl: Tuple[int, int]
    input_pos: Tuple[int, int]
    output_pos: Tuple[int, int]
    label_pos: Optional[Tuple[int, int]] = None

x_space = 71
y_space = 83

x1_1 = 414
x1_2 = x1_1 + x_space
x1_3 = x1_2 + x_space
x1_4 = x1_3 + x_space

y1_1 = 899
y1_2 = y1_1 - y_space

# Station 2 offset ift station 1
x_12_gap = 477
y_12_gap = 0

x2_1 = x1_1 + x_12_gap
x2_2 = x2_1 + x_space
x2_3 = x2_2 + x_space
x2_4 = x2_3 + x_space

y2_1 = y1_1
y2_2 = y2_1 - y_space

x_23_gap = 787
y_23_gap = -266

x3_1 = x2_1 + x_23_gap
x3_2 = x3_1 + x_space

y3_1 = y2_1 + y_23_gap
y3_2 = y3_1 + y_space
y3_3 = y3_2 + y_space
y3_4 = y3_3 + y_space

x_14_gap = 725
y_14_gap = -200

x4_1 = x1_1 + x_14_gap
x4_2 = x4_1 + x_space
x4_3 = x4_2 + x_space
x4_4 = x4_3 + x_space

y4_1 = y1_1 + y_14_gap
y4_2 = y4_1 - y_space

x_45_gap = -493
y_45_gap = 0

x5_1 = x4_1 + x_45_gap
x5_2 = x5_1 + x_space
x5_3 = x5_2 + x_space
x5_4 = x5_3 + x_space

y5_1 = y4_1
y5_2 = y5_1 - y_space

x_56_gap = -500
y_56_gap = -524

x6_1 = x5_1 + x_56_gap
x6_2 = x6_1 + x_space

y6_1 = y5_1 + y_56_gap
y6_2 = y6_1 + y_space
y6_3 = y6_2 + y_space
y6_4 = y6_3 + y_space


# NOTE: input_pos/output_pos er bevidst sat til fornuftige defaults.
# Du bør måle/justere dem så de rammer dine "tubes" rigtigt.
STATION_GEOMETRY_BY_INDEX: Dict[int, StationGeom] = {
    1: StationGeom(
        station_name_in_csv="Station 1: Bottom cover",
        # bottom row (lavest på skærmen) først, derefter top row
        queue_slots_tl=[
            (x1_3, y1_1), (x1_2, y1_1), (x1_1, y1_1),
            (x1_4, y1_2), (x1_3, y1_2), (x1_2, y1_2), (x1_1, y1_2),
        ],
        process_pos_tl=(x1_4, y1_1),
        input_pos=(x1_1 - 30, y1_1-6),
        output_pos=(x1_4 + 102, y1_1-6),
        label_pos=(650, 1010),
    ),

    2: StationGeom(
        station_name_in_csv="Station 2: Drill station",
        queue_slots_tl=[
            (x2_3, y2_1), (x2_2, y2_1), (x2_1, y2_1),
            (x2_4, y2_2), (x2_3, y2_2), (x2_2, y2_2), (x2_1, y2_2),
        ],
        process_pos_tl=(x2_4, y2_1),
        input_pos=(x2_1 - 30, y2_1-6),
        output_pos=(x2_4 + 102, y2_1-6),
        label_pos=(1150, 1010),
    ),

    3: StationGeom(
        station_name_in_csv="Station 3: Robot cell",
        queue_slots_tl=[
            (x3_1, y3_4),
            (x3_1, y3_3), (x3_2, y3_3),
            (x3_1, y3_2), (x3_2, y3_2),
            (x3_1, y3_1), (x3_2, y3_1),
        ],
        process_pos_tl=(x3_2, y3_4),
        input_pos=(1640, 891),
        output_pos=(1640, y4_1 - 6),
        label_pos=(1830, 600),
    ),

    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots_tl=[
            (x4_3, y4_1), (x4_2, y4_1), (x4_1, y4_1),
            (x4_4, y4_2), (x4_3, y4_2), (x4_2, y4_2), (x4_1, y4_2),
        ],
        process_pos_tl=(x4_4, y4_1),
        input_pos=(x4_4 + 102, y4_1 - 6),
        output_pos=(x4_1-30, y4_1 - 6),
        label_pos=(1180, 430),
    ),

    5: StationGeom(
        station_name_in_csv="Station 5: Top cover",
        queue_slots_tl=[
            (x5_3, y5_1), (x5_2, y5_1), (x5_1, y5_1),
            (x5_4, y5_2), (x5_3, y5_2), (x5_2, y5_2), (x5_1, y5_2),
        ],
        process_pos_tl=(x5_4, y5_1),
        input_pos=(x5_4 + 102, y5_1 - 6),
        output_pos=(x5_1 - 30, y5_1 - 6),
        label_pos=(700, 430),
    ),

    6: StationGeom(
        station_name_in_csv="Station 6: Packaging",
        queue_slots_tl=[
            (x6_1, y6_4),
            (x6_1, y6_3), (x6_2, y6_3),
            (x6_1, y6_2), (x6_2, y6_2),
            (x6_1, y6_1), (x6_2, y6_1),
        ],
        process_pos_tl=(x6_2, y6_4),
        input_pos=(x6_2 + 60, y6_4 - 63),
        output_pos=(x6_2 - 60, y6_4 + 30),
        label_pos=(130, 120),
    ),
}


# ----------------------------
# Folder helpers
# ----------------------------

def list_output_runs(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    runs = [p for p in output_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def build_movie_paths(order_dir: Path) -> Tuple[Path, Path, Path]:
    movie_dir = order_dir / "movie"
    frames_dir = movie_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = movie_dir / "after_movie.mp4"
    return movie_dir, frames_dir, mp4_path


def clear_folder(order_dir: Path) -> None:
    """Sletter movie/frames (og evt. gamle frames) før rendering.

    Bevidst robust på Windows (OneDrive/Explorer kan låse filer).
    """
    movie_dir = order_dir / "movie"
    frames_dir = movie_dir / "frames"

    if not frames_dir.exists():
        return

    def _onerror(func, path, excinfo):
        # prøv at fjerne readonly
        try:
            os.chmod(path, 0o666)
            func(path)
        except Exception:
            print(f">> Could not delete (locked): {path}")

    try:
        shutil.rmtree(frames_dir, onerror=_onerror)
    except Exception:
        print(f">> Could not delete (locked): {frames_dir}")

    frames_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Prompt helpers
# ----------------------------

def prompt_menu() -> str:
    print("\nWhat do you want to do?")
    print("  1) render (create frames + mp4)")
    print("  2) pick-coords (click to get pixel coordinates)")
    print("  3) quit")
    while True:
        choice = input("Select 1/2/3: ").strip().lower()
        if choice in {"1", "render"}:
            return "render"
        if choice in {"2", "pick", "pick-coords"}:
            return "pick-coords"
        if choice in {"3", "q", "quit", "exit"}:
            return "quit"
        print("Please type 1, 2 or 3.")


def prompt_for_run_folder(output_dir: Path) -> Path:
    runs = list_output_runs(output_dir)
    if not runs:
        raise FileNotFoundError(f"No run folders found in: {output_dir}")

    print("\nAvailable run folders in ./output (newest first):")
    for i, p in enumerate(runs, start=1):
        print(f"  {i:2d}) {p.name}")

    default = 1
    while True:
        s = input(f"Choose run folder number (Enter = {default}): ").strip()
        if s == "":
            return runs[default - 1]
        if s.isdigit() and 1 <= int(s) <= len(runs):
            return runs[int(s) - 1]
        candidate = output_dir / s
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("Invalid selection. Enter a number from the list or paste the folder name.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        s = input(f"{prompt} (Enter = {default}): ").strip()
        if s == "":
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Please enter a number.")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        s = input(f"{prompt} (Enter = {default}): ").strip()
        if s == "":
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


# ----------------------------
# Terminal progress bar (2% steps)
# ----------------------------

def progress_update(frame_idx: int, total_frames: int, next_pct: int, bar_width: int = 70) -> int:
    if total_frames <= 0:
        return next_pct

    pct = int((frame_idx / total_frames) * 100)
    if pct < next_pct and frame_idx != total_frames:
        return next_pct

    pct = min(100, max(0, pct))
    filled = int(round((pct / 100) * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    msg = f"Rendering: [{bar}] {pct:3d}%  ({frame_idx}/{total_frames})"
    print("\r" + msg, end="", flush=True)

    if pct >= 100:
        return 101
    return next_pct + 2


# ----------------------------
# Drawing helpers
# ----------------------------

def load_font_calibri_prefer(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prøver Calibri, ellers fallback. Samme fontfamilie; størrelser styres af size."""
    candidates = [
        str(ROOTDIR / "calibri.ttf"),
        str(ROOTDIR / "Calibri.ttf"),
        os.path.join(os.getcwd(), "calibri.ttf"),
        os.path.join(os.getcwd(), "Calibri.ttf"),
        "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def paste_icon_with_centered_text(
    frame_rgba: Image.Image,
    icon_rgba_resized: Image.Image,
    center_xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int, int],
):
    """Paste icon centered at center_xy and draw centered text."""
    cx, cy = center_xy
    w, h = icon_rgba_resized.size
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))

    frame_rgba.alpha_composite(icon_rgba_resized, dest=(x0, y0))

    draw = ImageDraw.Draw(frame_rgba)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(round(cx - tw / 2))
    ty = int(round(cy - th / 2))
    draw.text((tx, ty), text, font=font, fill=text_color)


def draw_loading_bar(frame_rgba: Image.Image, center_xy: Tuple[float, float], progress_0_1: float):
    """White bar filled green above the icon."""
    progress = float(max(0.0, min(1.0, progress_0_1)))
    cx, cy = center_xy

    x0 = int(round(cx - BAR_W / 2))
    y0 = int(round(cy - CARRIER_SIZE_PX / 2 - BAR_GAP - BAR_H))
    x1 = x0 + BAR_W
    y1 = y0 + BAR_H

    draw = ImageDraw.Draw(frame_rgba)
    draw.rectangle([x0, y0, x1, y1], fill=BAR_BG_COLOR, outline=BAR_BORDER_COLOR)

    fill_w = int(round(BAR_W * progress))
    if fill_w > 0:
        draw.rectangle([x0, y0, x0 + fill_w, y1], fill=BAR_FILL_COLOR)


def pick_coords_interactive(image_path: Path):
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)

    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.set_title("Click points; close window when done. Coordinates print in console.")
    ax.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f"Clicked: ({x}, {y})")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)


# ----------------------------
# Data preparation (event lists)
# ----------------------------

@dataclass
class ProcRecord:
    unit_id: str
    start: float
    finish: float
    disruption_end: float
    proc_start: float
    proc_dur: float


@dataclass
class TransportRecord:
    unit_id: str
    start: float
    finish: float
    from_name: str
    to_name: str


@dataclass
class Prepared:
    t_end: float
    arrivals: Dict[int, List[Tuple[float, str]]]
    starts: Dict[int, List[Tuple[float, ProcRecord]]]
    transports: List[TransportRecord]
    station_name_to_index: Dict[str, int]
    station_name_to_geom: Dict[str, StationGeom]


def disruption_and_processing_times_row(row: pd.Series) -> Tuple[float, float, float, float, float]:
    """Split processing into disruption + normal."""
    ts = float(row["start_time_s"])
    tf = float(row["finish_time_s"])
    total = max(tf - ts, 0.0)

    disruption = 0.0
    if "process_time_s" in row.index and "base_process_time_s" in row.index:
        try:
            pt = float(row["process_time_s"])
            bt = float(row["base_process_time_s"])
            disruption = max(pt - bt, 0.0)
        except Exception:
            disruption = 0.0

    disruption = min(disruption, total)
    proc_start = ts + disruption
    proc_dur = max(tf - proc_start, 0.0)
    return ts, tf, disruption, proc_start, proc_dur


def prepare_data(order_dir: Path) -> Prepared:
    station_df = pd.read_csv(order_dir / STATION_SCHEDULE_CSV)
    transport_df = pd.read_csv(order_dir / TRANSPORT_SCHEDULE_CSV)

    needed_station = {"unit_id", "station_index", "station_name", "arrival_time_s", "start_time_s", "finish_time_s"}
    needed_transport = {"unit_id", "from_station", "to_station", "start_time_s", "finish_time_s"}

    ms = needed_station - set(station_df.columns)
    mt = needed_transport - set(transport_df.columns)
    if ms:
        raise ValueError(f"station_schedule.csv missing columns: {sorted(ms)}")
    if mt:
        raise ValueError(f"transport_schedule.csv missing columns: {sorted(mt)}")

    station_df["unit_id"] = station_df["unit_id"].astype(str)
    transport_df["unit_id"] = transport_df["unit_id"].astype(str)

    station_name_to_index = (
        station_df[["station_name", "station_index"]]
        .drop_duplicates()
        .set_index("station_name")["station_index"]
        .to_dict()
    )

    station_name_to_geom = {g.station_name_in_csv: g for g in STATION_GEOMETRY_BY_INDEX.values()}

    t_end = float(np.nanmax([station_df["finish_time_s"].max(), transport_df["finish_time_s"].max()]))

    arrivals: Dict[int, List[Tuple[float, str]]] = {k: [] for k in STATION_GEOMETRY_BY_INDEX.keys()}
    starts: Dict[int, List[Tuple[float, ProcRecord]]] = {k: [] for k in STATION_GEOMETRY_BY_INDEX.keys()}

    station_df = station_df.sort_values(["station_index", "arrival_time_s", "start_time_s", "unit_id"]).reset_index(drop=True)

    for _, row in station_df.iterrows():
        idx = int(row["station_index"])
        unit_id = str(row["unit_id"])
        arr_t = float(row["arrival_time_s"])

        ts, tf, disruption, proc_start, proc_dur = disruption_and_processing_times_row(row)

        arrivals.setdefault(idx, []).append((arr_t, unit_id))

        pr = ProcRecord(
            unit_id=unit_id,
            start=ts,
            finish=tf,
            disruption_end=ts + disruption,
            proc_start=proc_start,
            proc_dur=proc_dur,
        )
        starts.setdefault(idx, []).append((ts, pr))

    for idx in arrivals:
        arrivals[idx].sort(key=lambda x: (x[0], x[1]))
        starts[idx].sort(key=lambda x: (x[0], x[1].unit_id))

    transport_df = transport_df.sort_values(["start_time_s", "unit_id"]).reset_index(drop=True)
    transports: List[TransportRecord] = [
        TransportRecord(
            unit_id=str(r["unit_id"]),
            start=float(r["start_time_s"]),
            finish=float(r["finish_time_s"]),
            from_name=str(r["from_station"]),
            to_name=str(r["to_station"]),
        )
        for _, r in transport_df.iterrows()
    ]

    return Prepared(
        t_end=t_end,
        arrivals=arrivals,
        starts=starts,
        transports=transports,
        station_name_to_index=station_name_to_index,
        station_name_to_geom=station_name_to_geom,
    )


# ----------------------------
# Transport helper: output -> input
# ----------------------------

def resolve_transport_points(prepared: Prepared, from_name: str, to_name: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    from_geom = prepared.station_name_to_geom.get(from_name)
    to_geom = prepared.station_name_to_geom.get(to_name)

    if from_geom is None and from_name in prepared.station_name_to_index:
        from_geom = STATION_GEOMETRY_BY_INDEX.get(int(prepared.station_name_to_index[from_name]))
    if to_geom is None and to_name in prepared.station_name_to_index:
        to_geom = STATION_GEOMETRY_BY_INDEX.get(int(prepared.station_name_to_index[to_name]))

    if from_geom is None or to_geom is None:
        return None

    return (from_geom.output_pos, to_geom.input_pos)


# ----------------------------
# Rendering
# ----------------------------

def render_after_movie(order_dir: Path, fps: int = FPS, sim_seconds_per_frame: float = SIM_SECONDS_PER_FRAME):
    starttime = time.perf_counter()
    order_dir = order_dir.resolve()
    if not order_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {order_dir}")

    # Ensure folder tree exists
    movie_dir, frames_dir, mp4_path = build_movie_paths(order_dir)

    # Clean frames folder before render (as requested)
    clear_folder(order_dir)

    prepared = prepare_data(order_dir)

    background = Image.open(BACKGROUND_PNG).convert("RGBA")
    carrier_icon = Image.open(CARRIER_PNG).convert("RGBA").resize(
        (CARRIER_SIZE_PX, CARRIER_SIZE_PX), Image.Resampling.LANCZOS
    )

    # Same icon for production and queue/transport (production icon removed)
    unit_icon = carrier_icon

    unit_font = load_font_calibri_prefer(UNIT_FONT_SIZE)
    disruption_font = load_font_calibri_prefer(DISRUPTION_FONT_SIZE)
    time_font = load_font_calibri_prefer(TIME_LABEL_FONT_SIZE)

    # Per-station pointers + state
    arr_ptr = {idx: 0 for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    start_ptr = {idx: 0 for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    queues: Dict[int, List[str]] = {idx: [] for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    active_proc: Dict[int, List[ProcRecord]] = {idx: [] for idx in STATION_GEOMETRY_BY_INDEX.keys()}

    # Transport pointers/state
    tr_ptr = 0
    active_tr: List[TransportRecord] = []

    writer = None
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(mp4_path), fps=fps, macro_block_size=1)
    except Exception as e:
        writer = None
        print("WARNING: Could not create MP4 writer. Frames will still be saved.")
        print("Reason:", e)

    total_frames = int(np.floor(prepared.t_end / sim_seconds_per_frame)) + 1
    next_pct = 0

    t = 0.0
    frame_idx = 0

    try:
        while frame_idx < total_frames:
            # --- advance station states ---
            for idx, geom in STATION_GEOMETRY_BY_INDEX.items():
                # arrivals
                arr_list = prepared.arrivals.get(idx, [])
                while arr_ptr[idx] < len(arr_list) and arr_list[arr_ptr[idx]][0] <= t:
                    _, u = arr_list[arr_ptr[idx]]
                    queues[idx].append(u)
                    arr_ptr[idx] += 1

                # starts
                start_list = prepared.starts.get(idx, [])
                while start_ptr[idx] < len(start_list) and start_list[start_ptr[idx]][0] <= t:
                    _, pr = start_list[start_ptr[idx]]
                    if pr.unit_id in queues[idx]:
                        queues[idx].remove(pr.unit_id)
                    active_proc[idx].append(pr)
                    start_ptr[idx] += 1

                # remove finished
                if active_proc[idx]:
                    active_proc[idx] = [pr for pr in active_proc[idx] if pr.finish > t]

            # --- advance transport state ---
            tr_list = prepared.transports
            while tr_ptr < len(tr_list) and tr_list[tr_ptr].start <= t:
                active_tr.append(tr_list[tr_ptr])
                tr_ptr += 1
            if active_tr:
                active_tr = [tr for tr in active_tr if tr.finish > t]

            # --- draw frame ---
            frame = background.copy()
            draw = ImageDraw.Draw(frame)

            for idx, geom in STATION_GEOMETRY_BY_INDEX.items():
                # queue
                q = queues[idx]
                for i, u in enumerate(q[:len(geom.queue_slots_tl)]):
                    center_xy = tl_to_center(geom.queue_slots_tl[i])
                    paste_icon_with_centered_text(frame, unit_icon, center_xy, u, unit_font, UNIT_TEXT_COLOR)

                # processing
                for k, pr in enumerate(active_proc[idx]):
                    base_center = tl_to_center(geom.process_pos_tl)
                    offset = k * (CARRIER_SIZE_PX * 0.35)
                    center = (base_center[0] + offset, base_center[1])

                    # disruption phase
                    if pr.disruption_end > pr.start and pr.start <= t < pr.disruption_end:
                        remaining = max(pr.disruption_end - t, 0.0)
                        label_pos = geom.label_pos or (int(center[0] + 25), int(center[1] - 35))
                        draw.text(
                            label_pos,
                            f"{DISRUPTION_PREFIX} {remaining:0.1f}s",
                            fill=DISRUPTION_TEXT_COLOR,
                            font=disruption_font,
                        )
                        paste_icon_with_centered_text(frame, unit_icon, center, pr.unit_id, unit_font, UNIT_TEXT_COLOR)
                    else:
                        # normal processing phase
                        if t < pr.proc_start:
                            progress = 0.0
                        elif pr.proc_dur <= 1e-9:
                            progress = 1.0
                        else:
                            progress = (t - pr.proc_start) / pr.proc_dur

                        draw_loading_bar(frame, center, progress)
                        paste_icon_with_centered_text(frame, unit_icon, center, pr.unit_id, unit_font, UNIT_TEXT_COLOR)

            # transports
            for tr in active_tr:
                pts = resolve_transport_points(prepared, tr.from_name, tr.to_name)
                if pts is None:
                    continue
                (x0, y0), (x1, y1) = pts

                denom = max(tr.finish - tr.start, 1e-9)
                alpha = max(0.0, min(1.0, (t - tr.start) / denom))
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                paste_icon_with_centered_text(frame, unit_icon, (x, y), tr.unit_id, unit_font, UNIT_TEXT_COLOR)

            if DRAW_TIME_LABEL:
                draw.text(TIME_LABEL_POS, f"t = {t:7.2f} s", fill=TIME_LABEL_COLOR, font=time_font)

            frame_path = frames_dir / f"{frame_idx + 1}.png"
            frame.save(frame_path)

            if writer is not None:
                writer.append_data(np.asarray(frame))

            frame_idx += 1
            t += sim_seconds_per_frame
            next_pct = progress_update(frame_idx, total_frames, next_pct)

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        print()  # newline after progress bar

    print(f"Rendered {frame_idx} frames")
    print(f"Frames saved in: {frames_dir}")
    if writer is not None:
        print(f"MP4 saved to:    {mp4_path}")
    return starttime, total_frames


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="After-movie renderer (top-left coords + IO transport points)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_pick = sub.add_parser("pick-coords", help="Click on the layout image to print pixel coordinates")
    p_pick.add_argument("--image", default=str(BACKGROUND_PNG), help="Path to layout PNG")

    p_render = sub.add_parser("render", help="Render MP4 + PNG frames")
    p_render.add_argument("--run", default=None, help="Run folder name inside ./output")
    p_render.add_argument("--fps", type=int, default=None, help="Video FPS")
    p_render.add_argument("--sim-seconds-per-frame", type=float, default=None,
                          help="Simulation seconds advanced per rendered frame")

    args = parser.parse_args()

    if args.cmd is None:
        cmd = prompt_menu()
        if cmd == "quit":
            return
        if cmd == "pick-coords":
            pick_coords_interactive(Path(BACKGROUND_PNG))
            return
        order_dir = prompt_for_run_folder(OUTPUTDIR)
        fps = prompt_int("FPS", FPS)
        spf = prompt_float("Simulation seconds per frame", SIM_SECONDS_PER_FRAME)
        #clear_folder(order_dir)
        starttime, total_frames = render_after_movie(order_dir, fps=fps, sim_seconds_per_frame=spf)
        endtime = time.perf_counter()
        fps = (endtime - starttime) / total_frames if total_frames > 0 else 0.0
        print(total_frames, "frames rendered.")
        print(f"Rendered {fps} frames per second.")
        print(f"Total rendering time: {endtime - starttime:.2f} seconds")
        return

    if args.cmd == "pick-coords":
        pick_coords_interactive(Path(args.image))
        return

    if args.cmd == "render":
        order_dir = prompt_for_run_folder(OUTPUTDIR) if args.run is None else (OUTPUTDIR / args.run)
        fps = args.fps if args.fps is not None else FPS
        spf = args.sim_seconds_per_frame if args.sim_seconds_per_frame is not None else SIM_SECONDS_PER_FRAME
        #clear_folder(order_dir)
        starttime, total_frames = render_after_movie(order_dir, fps=fps, sim_seconds_per_frame=spf)
        endtime = time.perf_counter()
        starttime, total_frames = render_after_movie(order_dir, fps=fps, sim_seconds_per_frame=spf)
        endtime = time.perf_counter()
        fps = int(np.round((endtime - starttime) / max(1e-9, total_frames)))
        print(f"Rendered {fps} frames per second.")
        print(f"Total rendering time: {endtime - starttime:.2f} seconds")
        return


if __name__ == "__main__":
    main()
