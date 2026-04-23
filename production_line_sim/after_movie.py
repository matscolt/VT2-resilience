#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""after_movie.py

✅ Interactive + fast after-movie renderer for a production line simulation.

Your folder tree:
production_line_sim/
  after_movie.py
  data/Layouts/
    1_LAYOUT.png
    carrier.png
    carrier_loading.png
  output/
    yyyymmdd_hhmmss.../
      station_schedule.csv
      transport_schedule.csv
      movie/
        after_movie.mp4
        frames/
          1.png
          2.png
          ...

What changed vs your last run:
- FIX for the KeyboardInterrupt hang: removed per-frame pandas sort/filter loops.
  We now use an event-driven simulation of queues/process/transport states.
  This makes rendering scale much better with many units/frames.
- MP4 creation:
  - If ffmpeg backend is installed, MP4 is written.
  - Writer is closed in a finally-block so the MP4 is finalized even if you Ctrl+C.
  - macro_block_size=1 avoids resizing warning (1920x1080 -> 1920x1088).

MP4 backend (Windows):
  pip install imageio[ffmpeg]
(or) pip install imageio[pyav]

Run:
  python after_movie.py
    (interactive prompts)

  python after_movie.py render
    (prompts for missing args)

  python after_movie.py render --run yyyymmdd_hhmmss
"""

from __future__ import annotations
import shutil
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Paths (match your tree)
# ----------------------------
ROOTDIR = Path(__file__).resolve().parent
LAYOUTDIR = ROOTDIR / "data" / "Layouts"
OUTPUTDIR = ROOTDIR / "output"

BACKGROUND_PNG = LAYOUTDIR / "1_LAYOUT.png"
CARRIER_PNG = LAYOUTDIR / "carrier.png"
PRODUCTION_ICON_PNG = LAYOUTDIR / "carrier_loading.png"

STATION_SCHEDULE_CSV = "station_schedule.csv"
TRANSPORT_SCHEDULE_CSV = "transport_schedule.csv"


# ----------------------------
# Video / rendering settings
# ----------------------------
FPS = 30
SIM_SECONDS_PER_FRAME = 0.5

CARRIER_SIZE_PX = 28
PRODUCTION_ICON_SIZE_PX = 40

FONT_SIZE = 12
TEXT_COLOR = (0, 0, 0, 255)

DRAW_TIME_LABEL = True
TIME_LABEL_POS = (20, 20)
TIME_LABEL_COLOR = (255, 255, 255, 255)

DISRUPTION_TEXT_COLOR = (255, 80, 80, 255)
DISRUPTION_PREFIX = "DISR"

BAR_W = 50
BAR_H = 8
BAR_GAP = 6
BAR_BG_COLOR = (255, 255, 255, 255)
BAR_BORDER_COLOR = (255, 255, 255, 255)
BAR_FILL_COLOR = (0, 200, 0, 255)


# ----------------------------
# Station geometry
# NOTE: first-pass center coordinates from your provided layout PNG.
# queue_slots order: bottom -> top (lowest on screen first)
# ----------------------------

@dataclass(frozen=True)
class StationGeom:
    station_name_in_csv: str
    queue_slots: List[Tuple[int, int]]
    process_pos: Tuple[int, int]
    center_pos: Tuple[int, int]
    label_pos: Optional[Tuple[int, int]] = None


STATION_GEOMETRY_BY_INDEX: Dict[int, StationGeom] = {
    1: StationGeom(
        station_name_in_csv="Station 1: Bottom cover",
        queue_slots=[(449, 851), (520, 851), (591, 851),
                    (449, 768), (520, 768), (591, 768), (662, 768)],
        process_pos=(662, 851),
        center_pos=(555, 809),
        label_pos=(375, 920),
    ),
    2: StationGeom(
        station_name_in_csv="Station 2: Drill station",
        queue_slots=[(823, 851), (894, 851), (981, 851),
                    (823, 768), (894, 768), (981, 768), (1068, 768)],
        process_pos=(1068, 851),
        center_pos=(945, 809),
        label_pos=(820, 920),
    ),
    3: StationGeom(
        station_name_in_csv="Station 3: Top cover",
        queue_slots=[(681, 651), (752, 651), (823, 651),
                    (681, 568), (752, 568), (823, 568), (894, 568)],
        process_pos=(894, 651),
        center_pos=(787, 609),
        label_pos=(700, 430),
    ),
    4: StationGeom(
        station_name_in_csv="Station 4: Inspection",
        queue_slots=[(1174, 652), (1245, 652), (1316, 652),
                    (1174, 559), (1245, 559), (1316, 559), (1387, 559)],
        process_pos=(1387, 652),
        center_pos=(1280, 600),
        label_pos=(1180, 430),
    ),
    5: StationGeom(
        station_name_in_csv="Station 5: Robot cell",
        queue_slots=[(1713, 834),
                    (1713, 751), (1784, 751),
                    (1713, 668), (1784, 668),
                    (1713, 567), (1784, 567)],
        process_pos=(1784, 834),
        center_pos=(1748, 691),
        label_pos=(1600, 430),
    ),
    6: StationGeom(
        station_name_in_csv="Station 6: Packaging",
        queue_slots=[(181, 460),
                    (181, 377), (252, 377),
                    (181, 294), (252, 294),
                    (181, 211), (252, 211)],
        process_pos=(252, 460),
        center_pos=(216, 335),
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

# ----------------------------
# clean folder
# ----------------------------

def clear_folder(folder):
    folder_path = folder / "movie"
    subfolder_path = folder_path / "frames"

    if not folder_path.exists():
        print(f">> Folder does not exist: {folder_path}")
        return

    for item in folder_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=False)
            else:
                item.unlink()
        except PermissionError:
            print(f">> Could not delete (locked): {item}")

    if not subfolder_path.exists():
        print(f">> Folder does not exist: {subfolder_path}")
        return

    for item in subfolder_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=False)
            else:
                item.unlink()
        except PermissionError:
            print(f">> Could not delete (locked): {item}")

    folder = str(folder).split("\\")[-1].split("__")[0]
    print(f"cleaned out {folder}/graph")


# ----------------------------
# Prompt helpers (interactive)
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
# Drawing helpers
# ----------------------------

def load_font_calibri_prefer(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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
    text_color: Tuple[int, int, int, int] = TEXT_COLOR,
):
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
    progress = float(max(0.0, min(1.0, progress_0_1)))
    cx, cy = center_xy

    x0 = int(round(cx - BAR_W / 2))
    y0 = int(round(cy - PRODUCTION_ICON_SIZE_PX / 2 - BAR_GAP - BAR_H))
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
# Data preparation
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
    # per station: arrivals and starts sorted
    arrivals: Dict[int, List[Tuple[float, str]]]
    starts: Dict[int, List[Tuple[float, ProcRecord]]]
    finishes: Dict[int, List[Tuple[float, str]]]
    # transports sorted by start
    transports: List[TransportRecord]
    # name mappings
    station_name_to_index: Dict[str, int]
    station_name_to_geom: Dict[str, StationGeom]


def disruption_and_processing_times_row(row: pd.Series) -> Tuple[float, float, float, float, float]:
    """Return (start, finish, disruption, proc_start, proc_dur)."""
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

    # Standardize unit_id to string for rendering and stable ordering
    station_df["unit_id"] = station_df["unit_id"].astype(str)
    transport_df["unit_id"] = transport_df["unit_id"].astype(str)

    # station name -> index
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
    finishes: Dict[int, List[Tuple[float, str]]] = {k: [] for k in STATION_GEOMETRY_BY_INDEX.keys()}

    # Build per-station event lists
    # Sort by arrival/start to ensure stable queue visualization
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
        finishes.setdefault(idx, []).append((tf, unit_id))

    # Sort lists
    for idx in arrivals:
        arrivals[idx].sort(key=lambda x: (x[0], x[1]))
        starts[idx].sort(key=lambda x: (x[0], x[1].unit_id))
        finishes[idx].sort(key=lambda x: (x[0], x[1]))

    # Transports
    # if transport_time_s missing, derive
    if "transport_time_s" not in transport_df.columns:
        transport_df["transport_time_s"] = transport_df["finish_time_s"] - transport_df["start_time_s"]

    transport_df = transport_df.sort_values(["start_time_s", "unit_id"]).reset_index(drop=True)
    transports: List[TransportRecord] = []
    for _, r in transport_df.iterrows():
        transports.append(TransportRecord(
            unit_id=str(r["unit_id"]),
            start=float(r["start_time_s"]),
            finish=float(r["finish_time_s"]),
            from_name=str(r["from_station"]),
            to_name=str(r["to_station"]),
        ))

    return Prepared(
        t_end=t_end,
        arrivals=arrivals,
        starts=starts,
        finishes=finishes,
        transports=transports,
        station_name_to_index=station_name_to_index,
        station_name_to_geom=station_name_to_geom,
    )


# ----------------------------
# Rendering (event-driven)
# ----------------------------

def print_mp4_backend_help(mp4_path: Path):
    print("\nMP4 writing failed.")
    print(f"Target file was: {mp4_path}")
    print("To enable MP4 output, install one backend:")
    print("  pip install imageio[ffmpeg]")
    print("or")
    print("  pip install imageio[pyav]")
    print("Frames are still saved as PNGs and can be converted with ffmpeg.")


def render_after_movie(order_dir: Path, fps: int = FPS, sim_seconds_per_frame: float = SIM_SECONDS_PER_FRAME):
    order_dir = order_dir.resolve()
    if not order_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {order_dir}")

    _, frames_dir, mp4_path = build_movie_paths(order_dir)

    prepared = prepare_data(order_dir)

    background = Image.open(BACKGROUND_PNG).convert("RGBA")
    carrier_icon = Image.open(CARRIER_PNG).convert("RGBA").resize((CARRIER_SIZE_PX, CARRIER_SIZE_PX), Image.Resampling.LANCZOS)
    prod_icon_path = PRODUCTION_ICON_PNG if PRODUCTION_ICON_PNG.exists() else CARRIER_PNG
    production_icon = Image.open(prod_icon_path).convert("RGBA").resize((PRODUCTION_ICON_SIZE_PX, PRODUCTION_ICON_SIZE_PX), Image.Resampling.LANCZOS)
    font = load_font_calibri_prefer(FONT_SIZE)

    # Per-station pointers + state
    arr_ptr = {idx: 0 for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    start_ptr = {idx: 0 for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    # visible queue content (unit_ids)
    queues: Dict[int, List[str]] = {idx: [] for idx in STATION_GEOMETRY_BY_INDEX.keys()}
    # active processing records
    active_proc: Dict[int, List[ProcRecord]] = {idx: [] for idx in STATION_GEOMETRY_BY_INDEX.keys()}

    # Transport pointers/state
    tr_ptr = 0
    active_tr: List[TransportRecord] = []

    # MP4 writer
    writer = None
    try:
        import imageio.v2 as imageio
        # macro_block_size=1 avoids 1080->1088 resizing warning.
        writer = imageio.get_writer(str(mp4_path), fps=fps, macro_block_size=1)
    except Exception as e:
        print("WARNING: Could not create MP4 writer. Frames will still be saved.")
        print("Reason:", e)
        print_mp4_backend_help(mp4_path)

    t = 0.0
    frame_idx = 0

    try:
        while t <= prepared.t_end + 1e-9:
            # ---- advance station states by events up to time t ----
            for idx, geom in STATION_GEOMETRY_BY_INDEX.items():
                # arrivals
                arr_list = prepared.arrivals.get(idx, [])
                while arr_ptr[idx] < len(arr_list) and arr_list[arr_ptr[idx]][0] <= t:
                    _, u = arr_list[arr_ptr[idx]]
                    queues[idx].append(u)
                    arr_ptr[idx] += 1

                # starts: remove from queue and add to processing
                start_list = prepared.starts.get(idx, [])
                while start_ptr[idx] < len(start_list) and start_list[start_ptr[idx]][0] <= t:
                    _, pr = start_list[start_ptr[idx]]
                    # remove that unit from queue if present
                    if pr.unit_id in queues[idx]:
                        queues[idx].remove(pr.unit_id)
                    active_proc[idx].append(pr)
                    start_ptr[idx] += 1

                # remove finished processing
                if active_proc[idx]:
                    active_proc[idx] = [pr for pr in active_proc[idx] if pr.finish > t]

            # ---- advance transport state ----
            while tr_ptr < len(prepared.transports) and prepared.transports[tr_ptr].start <= t:
                active_tr.append(prepared.transports[tr_ptr])
                tr_ptr += 1
            if active_tr:
                active_tr = [tr for tr in active_tr if tr.finish > t]

            # ---- draw frame ----
            frame = background.copy()
            draw = ImageDraw.Draw(frame)

            # Stations: draw queue + processing
            for idx, geom in STATION_GEOMETRY_BY_INDEX.items():
                # queue (bottom->top slots)
                q = queues[idx]
                # show first N in the visible queue order
                for i, u in enumerate(q[:len(geom.queue_slots)]):
                    paste_icon_with_centered_text(frame, carrier_icon, geom.queue_slots[i], u, font)

                # processing: can be multiple, draw with slight offset
                for k, pr in enumerate(active_proc[idx]):
                    offset = k * (PRODUCTION_ICON_SIZE_PX * 0.35)
                    center = (geom.process_pos[0] + offset, geom.process_pos[1])

                    if pr.disruption_end > pr.start and pr.start <= t < pr.disruption_end:
                        remaining = max(pr.disruption_end - t, 0.0)
                        label_pos = geom.label_pos or (int(center[0] + 25), int(center[1] - 35))
                        draw.text(label_pos, f"{DISRUPTION_PREFIX} {remaining:0.1f}s", fill=DISRUPTION_TEXT_COLOR, font=font)
                        paste_icon_with_centered_text(frame, production_icon, center, pr.unit_id, font)
                    else:
                        # loading bar runs for the "normal" part
                        if t < pr.proc_start:
                            progress = 0.0
                        elif pr.proc_dur <= 1e-9:
                            progress = 1.0
                        else:
                            progress = (t - pr.proc_start) / pr.proc_dur
                        draw_loading_bar(frame, center, progress)
                        paste_icon_with_centered_text(frame, production_icon, center, pr.unit_id, font)

            # Transports: draw moving carriers
            for tr in active_tr:
                # resolve geometries
                from_geom = prepared.station_name_to_geom.get(tr.from_name)
                to_geom = prepared.station_name_to_geom.get(tr.to_name)

                if from_geom is None and tr.from_name in prepared.station_name_to_index:
                    from_geom = STATION_GEOMETRY_BY_INDEX.get(int(prepared.station_name_to_index[tr.from_name]))
                if to_geom is None and tr.to_name in prepared.station_name_to_index:
                    to_geom = STATION_GEOMETRY_BY_INDEX.get(int(prepared.station_name_to_index[tr.to_name]))

                if from_geom is None or to_geom is None:
                    continue

                x0, y0 = from_geom.center_pos
                x1, y1 = to_geom.center_pos
                denom = max(tr.finish - tr.start, 1e-9)
                alpha = max(0.0, min(1.0, (t - tr.start) / denom))
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                paste_icon_with_centered_text(frame, carrier_icon, (x, y), tr.unit_id, font)

            # Time label
            if DRAW_TIME_LABEL:
                draw.text(TIME_LABEL_POS, f"t = {t:7.2f} s", fill=TIME_LABEL_COLOR, font=font)

            # Save frame PNG
            frame_path = frames_dir / f"{frame_idx + 1}.png"
            frame.save(frame_path)

            # Append to MP4
            if writer is not None:
                writer.append_data(np.asarray(frame))

            frame_idx += 1
            t += sim_seconds_per_frame

    finally:
        # Ensure MP4 is finalized even if user interrupts
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    print(f"Rendered {frame_idx} frames")
    print(f"Frames saved in: {frames_dir}")
    if writer is not None:
        print(f"MP4 saved to:    {mp4_path}")
    else:
        print("MP4 not created (missing backend).")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="After-movie renderer")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_pick = sub.add_parser("pick-coords", help="Click on the layout image to print pixel coordinates")
    p_pick.add_argument("--image", default=str(BACKGROUND_PNG), help="Path to layout PNG")

    p_render = sub.add_parser("render", help="Render MP4 + PNG frames")
    p_render.add_argument("--run", default=None, help="Run folder name inside ./output")
    p_render.add_argument("--fps", type=int, default=None, help="Video FPS")
    p_render.add_argument("--sim-seconds-per-frame", type=float, default=None,
                          help="Simulation seconds advanced per rendered frame")

    args = parser.parse_args()

    # No subcommand -> interactive
    if args.cmd is None:
        cmd = prompt_menu()
        if cmd == "quit":
            return
        if cmd == "pick-coords":
            pick_coords_interactive(BACKGROUND_PNG)
            return
        # render
        order_dir = prompt_for_run_folder(OUTPUTDIR)
        fps = prompt_int("FPS", FPS)
        spf = prompt_float("Simulation seconds per frame", SIM_SECONDS_PER_FRAME)
        clear_folder(order_dir)
        render_after_movie(order_dir, fps=fps, sim_seconds_per_frame=spf)
        return

    if args.cmd == "pick-coords":
        pick_coords_interactive(Path(args.image))
        return

    if args.cmd == "render":
        if args.run is None:
            order_dir = prompt_for_run_folder(OUTPUTDIR)
        else:
            order_dir = OUTPUTDIR / args.run

        fps = args.fps if args.fps is not None else FPS
        spf = args.sim_seconds_per_frame if args.sim_seconds_per_frame is not None else SIM_SECONDS_PER_FRAME
        clear_folder(order_dir)
        render_after_movie(order_dir, fps=fps, sim_seconds_per_frame=spf)
        return


if __name__ == "__main__":
    main()
